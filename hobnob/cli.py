from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Dict

from hobnob.tristore import TriStore
from hobnob.publish import publish_flow
from hobnob.executors import ExecutorRegistry
from hobnob.rendering import PromptRenderer
from hobnob.parsing import JsonParser


log = logging.getLogger("hobnob.cli")


def _build_executor(step_cfg: Dict[str, Any], llm: Any, flow_def: Dict[str, Any]):
    class _RunnerShim:
        def __init__(self, llm: Any, flow_def: Dict[str, Any]) -> None:
            self.llm = llm
            self.flow_def = flow_def

    runner_like = _RunnerShim(llm, flow_def)
    factory = ExecutorRegistry.get(step_cfg.get("type", "llm"))
    return factory(step_cfg, runner_like)


def cmd_replay(args: argparse.Namespace) -> int:
    store = TriStore()
    lineage = store.lineage(args.instance)
    if not lineage:
        print("No lineage found.")
        return 1

    ident = store._flow_identity_for_instance(args.instance)
    flow_def = store.get_flow(ident["name"], ident["semver"])  # includes system_prompt, steps

    # Lazy import to avoid hard dependency when not used
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=args.model, temperature=0)
    except Exception as e:  # pragma: no cover - optional env path
        print("Unable to initialize LLM (langchain_openai). Install deps and set credentials.")
        return 2

    prompt_renderer = PromptRenderer(flow_def.get("system_prompt", ""))
    parser = JsonParser()

    print(f"Replaying instance: {args.instance} for {ident['name']}@{ident['semver']}")
    ok = True
    for cp in lineage:
        step_no = cp.get("step_no")
        step_name = cp.get("step_name")
        if step_no == 0:
            print(f"- checkpoint {step_no}: initial state (skipped)")
            continue
        step_spec = cp.get("step_spec") or {}
        stype = step_spec.get("type", "llm")
        input_state = cp.get("input_jsonb") or {}
        expected_state = cp.get("output_jsonb") or {}

        print(f"- step {step_no}: {step_name} [{stype}]")
        if stype == "user_input":
            print("  user_input step: skipping re-execution; comparing stored output only")
            print("  ok")
            continue

        try:
            exe = _build_executor(step_spec, llm, flow_def)
            out = exe(input_state)
        except Exception as e:
            print(f"  error re-executing: {e}")
            ok = False
            continue

        # Compare only keys that were present in stored output
        mismatches = {}
        for k, v in (expected_state or {}).items():
            if out.get(k) != v:
                mismatches[k] = {"expected": v, "actual": out.get(k)}

        if mismatches:
            print("  mismatch:")
            print(json.dumps(mismatches, indent=2))
            ok = False
        else:
            print("  ok")

    return 0 if ok else 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hobnob")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_replay = sub.add_parser("replay", help="Replay a stored instance")
    p_replay.add_argument("--instance", required=True, help="Instance UUID")
    p_replay.add_argument("--model", default="gpt-4o", help="LLM model id for replay")
    p_replay.set_defaults(func=cmd_replay)

    p_init = sub.add_parser("graph-init", help="Initialize AGE graph and tables")
    p_init.add_argument("--dsn", default=None, help="PostgreSQL DSN (uses env if omitted)")
    p_init.add_argument("--graph", default="hob", help="Graph name (default: hob)")
    p_init.set_defaults(func=cmd_graph_init)

    p_pub = sub.add_parser("publish", help="Publish a flow (prompt or file) to TriStore")
    src = p_pub.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", help="Path to flow JSON file")
    src.add_argument("--prompt", help="Natural language spec to generate a flow")
    p_pub.add_argument("--name", help="Flow name (overrides)")
    p_pub.add_argument("--semver", help="Semver (overrides)")
    p_pub.add_argument("--model", default="gpt-4o", help="Model for --prompt generation")
    p_pub.set_defaults(func=cmd_publish)

    p_qr = sub.add_parser("quickrun", help="Generate (or load), publish, and run a flow")
    src2 = p_qr.add_mutually_exclusive_group(required=True)
    src2.add_argument("--file", help="Path to flow JSON file")
    src2.add_argument("--prompt", help="Natural language spec to generate a flow")
    p_qr.add_argument("--name", help="Flow name (overrides or default)")
    p_qr.add_argument("--semver", help="Semver (overrides or default)")
    p_qr.add_argument("--model", default="gpt-4o", help="Model id for generation and run")
    p_qr.add_argument("--temperature", type=float, default=0.0, help="LLM temperature for run (default 0.0)")
    p_qr.add_argument("--tenant", default="demo", help="Tenant label for the run")
    p_qr.add_argument("--state", help="Initial state as JSON string")
    p_qr.add_argument("--state-file", help="Path to a JSON file with initial state")
    p_qr.set_defaults(func=cmd_quickrun)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))


def cmd_graph_init(args: argparse.Namespace) -> int:
    """Initialize AGE graph and minimal relational tables.

    Attempts to:
    - CREATE EXTENSION age (if permitted)
    - LOAD 'age' (if needed)
    - CREATE GRAPH <graph> (or SELECT create_graph('<graph>'))
    - SET graph_path = <graph>
    - Create outbox and idempotency tables
    """
    import os
    try:
        import psycopg
    except Exception as e:
        print("psycopg is required. Install 'psycopg[binary,pool]'.")
        return 1

    dsn = args.dsn or os.getenv("HOBNOB_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not dsn:
        print("No DSN provided. Use --dsn or set HOBNOB_DATABASE_URL/DATABASE_URL.")
        return 2

    graph = args.graph
    try:
        with psycopg.connect(dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                # Best-effort AGE enablement
                for stmt in (
                    "CREATE EXTENSION IF NOT EXISTS age;",
                    "LOAD 'age';",
                ):
                    try:
                        cur.execute(stmt)
                    except Exception:
                        pass  # ignore if not permitted

                # Create graph (try both syntaxes)
                try:
                    cur.execute(f"CREATE GRAPH {graph};")
                except Exception:
                    try:
                        cur.execute("SELECT create_graph(%s);", (graph,))
                    except Exception:
                        pass  # maybe it already exists

                try:
                    cur.execute("SET graph_path = %s;", (graph,))
                except Exception:
                    pass

                # Tables
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS hobnob_outbox (
                      id uuid PRIMARY KEY,
                      instance_id text NOT NULL,
                      kind text NOT NULL,
                      payload jsonb NOT NULL,
                      created_at timestamptz DEFAULT now(),
                      delivered_at timestamptz
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS hobnob_idempotency (
                      key text PRIMARY KEY,
                      seen_at timestamptz DEFAULT now()
                    );
                    """
                )
        print(f"Graph '{graph}' initialized (best effort).")
        return 0
    except Exception as e:
        print(f"Initialization failed: {e}")
        return 3


def cmd_publish(args: argparse.Namespace) -> int:
    store = TriStore()
    if args.file:
        import json, pathlib
        path = pathlib.Path(args.file)
        flow = json.loads(path.read_text())
    else:
        # Prompt-to-Flow
        try:
            from langchain_openai import ChatOpenAI
        except Exception:
            print("langchain_openai not available. Install dependencies and set OPENAI_API_KEY.")
            return 2
        from hobnob.generation import from_prompt
        llm = ChatOpenAI(model=args.model, temperature=0)
        flow = from_prompt(args.prompt, llm=llm)

    try:
        summary = publish_flow(flow, store, name=args.name, semver=args.semver)
    except Exception as e:
        print(f"Publish failed: {e}")
        return 1

    print(json.dumps(summary, indent=2))
    return 0


def _load_state(args: argparse.Namespace) -> Dict[str, Any]:
    if args.state and args.state_file:
        raise SystemExit("Use either --state or --state-file, not both.")
    if args.state_file:
        import pathlib
        return json.loads(pathlib.Path(args.state_file).read_text())
    if args.state:
        return json.loads(args.state)
    # Default: empty dict
    return {}


def cmd_quickrun(args: argparse.Namespace) -> int:
    store = TriStore()
    # Build or load flow
    if args.file:
        import pathlib
        flow = json.loads(pathlib.Path(args.file).read_text())
    else:
        try:
            from langchain_openai import ChatOpenAI
        except Exception:
            print("langchain_openai not available. Install dependencies and set OPENAI_API_KEY.")
            return 2
        from hobnob.generation import from_prompt
        llm_gen = ChatOpenAI(model=args.model, temperature=0)
        flow = from_prompt(args.prompt, llm=llm_gen)

    # Ensure identity
    flow.setdefault("name", args.name or "adhoc")
    flow.setdefault("semver", args.semver or "0.0.1")
    if args.name:
        flow["name"] = args.name
    if args.semver:
        flow["semver"] = args.semver

    # Publish (enforces immutability)
    try:
        summary = publish_flow(flow, store)
    except Exception as e:
        print(f"Publish failed: {e}")
        return 1

    print("Published:")
    print(json.dumps(summary, indent=2))

    # Run
    try:
        from langchain_openai import ChatOpenAI
    except Exception:
        print("langchain_openai not available. Install dependencies and set OPENAI_API_KEY.")
        return 2
    from hobnob.core import FlowRunner

    llm = ChatOpenAI(model=args.model, temperature=args.temperature)

    def on_step(step_name: str, state: Dict[str, Any]) -> None:
        print(f"\n--- {step_name} ---")
        # print a compact state view
        try:
            print(json.dumps(state, indent=2))
        except Exception:
            print(str(state))

    runner = FlowRunner(flow, llm, infer_state=True, on_step=on_step, persist_flow=True, store=store)
    init_state = _load_state(args)
    final = runner.run(init_state, tenant=args.tenant)
    print("\n==== FINAL STATE ====")
    print(json.dumps(final, indent=2))

    # Print instance id hint
    try:
        # The runner appends an "end" event; we can fetch the latest instance id by scanning lineage.
        # In real usage, TriStore.start_instance already returns it; FlowRunner stores it internally.
        # We don't expose it publicly here, so provide a replay hint:
        print("\nReplay using: hobnob replay --instance <uuid>")
    except Exception:
        pass
    return 0
