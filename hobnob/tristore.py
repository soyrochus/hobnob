from __future__ import annotations

import os
import uuid
import datetime as _dt
from typing import Any, Dict, List, Optional


class TriStore:
    """Graph-first storage using PostgreSQL + AGE (openCypher).

    Notes:
    - Uses AGE's `cypher(graph, query, params_json)` function.
    - Each public method opens a single transaction and executes one or more
      Cypher statements within it.
    - JSONB blobs are stored directly on vertices/edges via properties like
      `spec_jsonb`, `input_jsonb`, `output_jsonb`, etc.
    - Step identity is based on (flow_name, flow_semver, step name).
    """

    def __init__(
        self,
        dsn: Optional[str] = None,
        graph: str = "hob",
    ) -> None:
        self.graph = graph
        self.dsn = dsn or os.getenv("HOBNOB_DATABASE_URL") or os.getenv("DATABASE_URL") or ""
        try:
            import psycopg
            self._psycopg = psycopg
        except Exception as e:  # pragma: no cover - import path
            raise RuntimeError(
                "psycopg is required for TriStore. Please install 'psycopg[binary,pool]'"
            ) from e

    # --------------- internal helpers ---------------
    def _conn(self):  # type: ignore[override]
        return self._psycopg.connect(self.dsn)  # autocommit False by default

    def _set_graph_path(self, cur) -> None:
        cur.execute("SET graph_path = %s;", (self.graph,))

    def _cypher(self, cur, query: str, params: Optional[Dict[str, Any]] = None) -> List[Any]:
        """Execute a cypher query against AGE and return a list of `r` values.

        We wrap the query in a SELECT call to `cypher(graph, query, params)`
        and coerce the column to text for portability.
        """
        import json as _json
        # Build params JSON (AGE expects a JSON as the 3rd arg)
        params_json = _json.dumps(params or {})
        # Return each row as text form of agtype via casting
        cur.execute(
            "SELECT (r)::text FROM cypher(%s, %s, %s) AS (r agtype);",
            (self.graph, query, params_json),
        )
        return [row[0] for row in cur.fetchall()]

    def _now(self) -> str:
        return _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc).isoformat()

    # --------------- public API ---------------
    def create_flow(self, flow_json: Dict[str, Any]) -> str:
        name = flow_json.get("name")
        semver = flow_json.get("semver")
        if not name or not semver:
            raise ValueError("flow_json must include 'name' and 'semver'")

        steps: List[Dict[str, Any]] = list(flow_json.get("steps", []))
        transitions: List[Dict[str, Any]] = list(flow_json.get("transitions", []))
        initial_step: str = flow_json.get("initial_step")
        if not initial_step:
            raise ValueError("flow_json must include 'initial_step'")

        q = """
        WITH $flow AS flow, $steps AS steps, $trans AS trans, $initial AS initial
        MERGE (f:Flow {name: flow.name, semver: flow.semver})
        ON CREATE SET f.spec_jsonb = flow, f.created_at = $ts
        ON MATCH  SET f.updated_at = $ts
        WITH f, steps, trans, initial
        UNWIND steps AS st
        MERGE (s:Step {flow_name: f.name, flow_semver: f.semver, name: st.name})
        SET s.spec_jsonb = st, s.type = coalesce(st.type, 'llm')
        MERGE (f)-[:HAS_STEP]->(s)
        WITH f, trans, initial
        MATCH (s0:Step {flow_name: f.name, flow_semver: f.semver, name: initial})
        MERGE (f)-[:INITIAL]->(s0)
        WITH f, trans
        UNWIND trans AS tr
        MATCH (src:Step {flow_name: f.name, flow_semver: f.semver, name: tr.from})
        WITH f, src, tr
        OPTIONAL MATCH (dst:Step {flow_name: f.name, flow_semver: f.semver, name: tr.to})
        FOREACH (_ IN CASE WHEN tr.to IS NOT NULL THEN [1] ELSE [] END |
            MERGE (src)-[:NEXT {condition: tr.condition}]->(dst)
        )
        RETURN f
        """

        payload = {
            "flow": flow_json,
            "steps": steps,
            "trans": transitions,
            "initial": initial_step,
            "ts": self._now(),
        }

        with self._conn() as conn:
            with conn.cursor() as cur:
                self._set_graph_path(cur)
                self._cypher(cur, q, payload)
        return f"{name}@{semver}"

    def get_flow(self, name: str, semver: str) -> Dict[str, Any]:
        q = """
        MATCH (f:Flow {name: $name, semver: $semver})
        RETURN f.spec_jsonb AS spec
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                self._set_graph_path(cur)
                rows = self._cypher(cur, q, {"name": name, "semver": semver})
        if not rows:
            raise KeyError(f"Flow not found: {name}@{semver}")

        import json as _json
        # AGE returns agtype::text for map, which is valid JSON-ish
        spec = _json.loads(rows[0])
        return spec

    def start_instance(
        self,
        name: str,
        semver: str,
        state: Dict[str, Any],
        tenant: str,
    ) -> str:
        iid = str(uuid.uuid4())
        q = """
        MATCH (f:Flow {name: $name, semver: $semver})
        CREATE (i:Instance {id: $iid, tenant: $tenant, created_at: $ts})
        MERGE (i)-[:OF]->(f)
        SET i.state_jsonb = $state
        CREATE (c:Checkpoint {
            step_no: 0,
            input_jsonb: $state,
            output_jsonb: $state,
            error_jsonb: null,
            created_at: $ts,
            terminal: false
        })
        MERGE (c)-[:OF]->(i)
        RETURN i.id AS iid
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                self._set_graph_path(cur)
                self._cypher(
                    cur,
                    q,
                    {
                        "name": name,
                        "semver": semver,
                        "state": state,
                        "tenant": tenant,
                        "iid": iid,
                        "ts": self._now(),
                    },
                )
        return iid

    def current_checkpoint(self, instance_id: str) -> Optional[Dict[str, Any]]:
        q = """
        MATCH (i:Instance {id: $iid})
        MATCH (c:Checkpoint)-[:OF]->(i)
        RETURN properties(c) AS c
        ORDER BY c.step_no DESC
        LIMIT 1
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                self._set_graph_path(cur)
                rows = self._cypher(cur, q, {"iid": instance_id})
        if not rows:
            return None
        import json as _json
        return _json.loads(rows[0])

    def advance(
        self,
        instance_id: str,
        next_step_id: str,
        step_no: int,
        io: Dict[str, Any],
        fingerprints: Dict[str, Any],
        terminal: bool,
    ) -> Dict[str, Any]:
        q = """
        MATCH (i:Instance {id: $iid})-[:OF]->(f:Flow)
        MATCH (s:Step {flow_name: f.name, flow_semver: f.semver, name: $step})
        CREATE (c:Checkpoint {
            step_no: $no,
            input_jsonb: $io.input_jsonb,
            output_jsonb: $io.output_jsonb,
            error_jsonb: $io.error_jsonb,
            fingerprints: $fp,
            created_at: $ts,
            terminal: $terminal
        })
        MERGE (c)-[:OF]->(i)
        MERGE (c)-[:AT]->(s)
        WITH i, c
        OPTIONAL MATCH (p:Checkpoint {step_no: c.step_no - 1})-[:OF]->(i)
        FOREACH (_ IN CASE WHEN p IS NOT NULL THEN [1] ELSE [] END | MERGE (c)-[:PREV]->(p))
        MERGE (i)-[:RAN]->(s)
        RETURN properties(c) AS c
        """
        params = {
            "iid": instance_id,
            "step": next_step_id,
            "no": step_no,
            "io": io,
            "fp": fingerprints,
            "terminal": terminal,
            "ts": self._now(),
        }
        with self._conn() as conn:
            with conn.cursor() as cur:
                self._set_graph_path(cur)
                rows = self._cypher(cur, q, params)
        import json as _json
        return _json.loads(rows[0]) if rows else {}

    def append_event(self, instance_id: str, kind: str, payload: Dict[str, Any]) -> None:
        q = """
        MATCH (i:Instance {id: $iid})
        CREATE (e:Event {kind: $kind, payload: $payload, created_at: $ts})
        MERGE (i)-[:EMITTED]->(e)
        RETURN 1
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                self._set_graph_path(cur)
                self._cypher(
                    cur,
                    q,
                    {"iid": instance_id, "kind": kind, "payload": payload, "ts": self._now()},
                )

    def lineage(self, instance_id: str) -> List[Dict[str, Any]]:
        q = """
        MATCH (i:Instance {id: $iid})
        MATCH (c:Checkpoint)-[:OF]->(i)
        OPTIONAL MATCH (c)-[:AT]->(s:Step)
        RETURN c.step_no AS step_no,
               c.input_jsonb AS input_jsonb,
               c.output_jsonb AS output_jsonb,
               c.error_jsonb AS error_jsonb,
               c.fingerprints AS fingerprints,
               s.name AS step_name,
               s.spec_jsonb AS step_spec
        ORDER BY step_no ASC
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                self._set_graph_path(cur)
                rows = self._cypher(cur, q, {"iid": instance_id})
        import json as _json
        result: List[Dict[str, Any]] = []
        for row in rows:
            result.append(_json.loads(row))
        return result

    # ---------- Helpers for CLI (not part of public spec) ----------
    def _flow_identity_for_instance(self, instance_id: str) -> Dict[str, str]:
        q = """
        MATCH (i:Instance {id: $iid})-[:OF]->(f:Flow)
        RETURN f.name AS name, f.semver AS semver
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                self._set_graph_path(cur)
                rows = self._cypher(cur, q, {"iid": instance_id})
        import json as _json
        if not rows:
            raise KeyError(f"Instance not found: {instance_id}")
        return _json.loads(rows[0])
