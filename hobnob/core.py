from __future__ import annotations
from typing import Dict, Any, Optional, Set, TypedDict
import logging
import time
import re
import warnings
from langgraph.graph import StateGraph, END
from hobnob.executors import ExecutorRegistry
from hobnob.rendering import PromptRenderer
from hobnob.parsing import JsonParser
from hobnob.routers import ConditionRouter, RouterRegistry
from typing import cast

try:
    # Optional import to avoid circular dependency at import time
    from hobnob.tristore import TriStore  # type: ignore
except Exception:  # pragma: no cover - optional dependency path
    TriStore = None  # type: ignore

logger = logging.getLogger(__name__)

class FlowRunner:
    def __init__(
        self,
        flow_def,
        llm,
        state_schema: Optional[type] = None,
        on_step=None,
        condition_router: Optional[ConditionRouter] = None,
        infer_state: bool = True,
        persist_flow: bool = False,
        store: Optional["TriStore"] = None,
    ):
        self.flow_def = flow_def
        self.llm = llm
        self.on_step = on_step
        self.condition_router = condition_router or RouterRegistry.get("jmespath")
        self.infer_state = infer_state
        self.store = store
        self.persist_flow = persist_flow
        if state_schema is None:
            if infer_state:
                state_schema = self._infer_state_schema()
            else:
                raise ValueError("state_schema must be provided when infer_state is False")
        self.state_schema = state_schema
        self._known_fields: Set[str] = set(getattr(self.state_schema, "__annotations__", {}).keys())
        self._graph = self._build_graph()

        # Storage-related ephemeral fields (per-run)
        self._instance_id: Optional[str] = None
        self._step_no: int = 0
        self._by_src: Dict[str, list] = {}

        # Persist flow spec if requested
        if self.persist_flow and self.store is not None:
            try:
                self.store.create_flow(self.flow_def)
            except Exception as e:  # pragma: no cover - storage path
                logger.warning("Failed to persist flow: %s", e)

    def _infer_state_schema(self) -> type:
        fields: Set[str] = set()
        for step in self.flow_def.get("steps", []):
            prompt = step.get("prompt", "")
            fields.update(re.findall(r"{(.*?)}", prompt))
            for ex in step.get("examples", []):
                for part in ("input", "output"):
                    data = ex.get(part, {})
                    if isinstance(data, dict):
                        fields.update(data.keys())
        return TypedDict("InferredState", {f: Any for f in fields})

    def _validate_state(self, state: Dict[str, Any]):
        if not self.infer_state or not self._known_fields:
            return
        extra = set(state.keys()) - self._known_fields
        if extra:
            warnings.warn(f"Unexpected state fields: {sorted(extra)}")

    def _router_factory(self, conds):
        router = self.condition_router

        def _route(state):
            for tr in conds:
                cond = tr.get("condition")
                target = tr["to"]
                if cond is None:
                    return target or END
                try:
                    if router.check(cond, state):
                        return target or END
                except Exception as e:
                    logger.exception("Condition routing error: %s", e)
            return END
        return _route
    
    def _make_executor(self, step_cfg):
        step_name = step_cfg["name"]
        stype = step_cfg.get("type", "llm")
        on_step = self.on_step
        retry_cfg = step_cfg.get("retry", {})
        system_prompt = self.flow_def.get("system_prompt", "")

        factory = ExecutorRegistry.get(stype)

        def _run_once(state: Dict[str, Any]) -> Dict[str, Any]:
            # Create a fresh executor each attempt in case it has internal state.
            executor = factory(
                {**step_cfg, "system_prompt": system_prompt},  # pass through config
                self,
            )
            return executor(state)

        def _fn(state: Dict[str, Any]) -> Dict[str, Any]:
            logger.info("Step %s input: %s", step_name, state)
            max_attempts = int(retry_cfg.get("max_attempts", 1) or 1)
            backoff = float(retry_cfg.get("backoff", 0) or 0)
            attempt = 0
            last_exc: Optional[Exception] = None
            while attempt < max_attempts:
                attempt += 1
                try:
                    inp_snapshot = dict(state)
                    out = _run_once(state)
                    logger.info("Step %s output: %s", step_name, out)
                    if on_step:
                        on_step(step_name, out)

                    # Persist checkpoint if storage is configured
                    if self.store is not None:
                        try:
                            # Increment step counter per successful execution
                            self._step_no += 1
                            io = {
                                "input_jsonb": inp_snapshot,
                                "output_jsonb": out,
                                "error_jsonb": None,
                            }
                            # Basic fingerprints; can be refined as needed
                            try:
                                model_fp = getattr(self.llm, "model_name", None)
                            except Exception:  # pragma: no cover - llm attribute optional
                                model_fp = None
                            import hashlib, json as _json
                            rendered_prompt = PromptRenderer(self.flow_def.get("system_prompt", "")).render(step_cfg, inp_snapshot)
                            prompt_hash = hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest()
                            toolset_hash = hashlib.sha256((stype + ":" + step_name).encode("utf-8")).hexdigest()
                            fingerprints = {
                                "prompt_hash": prompt_hash,
                                "toolset_hash": toolset_hash,
                                "model_fingerprint": model_fp,
                            }
                            inst_id = cast(Optional[str], self._instance_id) or ""
                            if inst_id:
                                is_terminal = self._is_terminal_after(step_name, out)
                                self.store.advance(
                                    inst_id,
                                    step_name,
                                    self._step_no,
                                    io,
                                    fingerprints,
                                    terminal=is_terminal,
                                )
                        except Exception as se:  # pragma: no cover - storage path
                            logger.warning("Failed to persist checkpoint: %s", se)
                    return out
                except Exception as exc:  # pragma: no cover - error path
                    last_exc = exc
                    if attempt < max_attempts:
                        logger.warning(
                            "Retrying step %s after error: %s (attempt %s)",
                            step_name,
                            exc,
                            attempt,
                        )
                        if backoff > 0:
                            time.sleep(backoff)
                    else:
                        logger.exception("Step %s failed", step_name)
                        raise
            # Should not reach here
            raise last_exc if last_exc else RuntimeError("Unknown step failure")

        return _fn

    def _build_graph(self):
        sg = StateGraph(self.state_schema)

        # nodes
        for step in self.flow_def["steps"]:
            sg.add_node(step["name"], self._make_executor(step))

        # transitions
        by_src: Dict[str, list] = {}
        for tr in self.flow_def["transitions"]:
            by_src.setdefault(tr["from"], []).append(tr)

        # Keep for terminal detection during execution
        self._by_src = by_src

        for src, lst in by_src.items():
            if len(lst) == 1 and lst[0].get("condition") is None:
                sg.add_edge(src, lst[0]["to"] or END)
            else:
                targets = {tr["to"] or END for tr in lst}
                sg.add_conditional_edges(src,
                                         self._router_factory(lst),
                                         list(targets))
        sg.set_entry_point(self.flow_def["initial_step"])
        return sg.compile()

    def _is_terminal_after(self, step_name: str, state: Dict[str, Any]) -> bool:
        conds = self._by_src.get(step_name) or []
        if not conds:
            return True  # No outgoing edges means terminal
        if len(conds) == 1 and conds[0].get("condition") is None:
            return conds[0].get("to") is None
        # Evaluate conditions via the configured router
        for tr in conds:
            cond = tr.get("condition")
            if cond is None:
                return tr.get("to") is None
            try:
                if self.condition_router.check(cond, state):
                    return tr.get("to") is None
            except Exception:
                # On router error, treat as not matched and continue
                continue
        # No condition matched => END
        return True

    # ---------- public API ----------
    def run(self, initial_state: Dict[str, Any], tenant: Optional[str] = None) -> Dict[str, Any]:
        self._validate_state(initial_state)
        # Start a persisted instance if configured
        if self.store is not None:
            try:
                name = self.flow_def.get("name", "anonymous")
                semver = self.flow_def.get("semver", "0.0.0")
                self._instance_id = self.store.start_instance(
                    name,
                    semver,
                    initial_state,
                    tenant or "default",
                )
                self._step_no = 0
            except Exception as e:  # pragma: no cover - storage path
                logger.warning("Failed to start instance: %s", e)

        result = self._graph.invoke(initial_state)

        # Optionally mark terminal checkpoint or append event
        if self.store is not None and self._instance_id:
            try:
                self.store.append_event(self._instance_id, "end", {"final_state": result})
            except Exception as e:  # pragma: no cover - storage path
                logger.warning("Failed to append end event: %s", e)

        return result
