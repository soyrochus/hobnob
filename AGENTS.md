# agents.md — Hobnob × TriStore (AGE/Cypher) Edition

**Goal**
Implement a **graph-first TriStore** for Hobnob so we can **create flows**, **store flows**, and **replay flows** while reusing the current API and the **Fibonacci with Limericks** example from the README as an end-to-end test. Exclude vectors for now.

---

## 0) Architecture and priorities

**Storage priorities**

1. **Graph (AGE/Cypher)** — primary source of truth for flows, steps, transitions, instances, checkpoints, and events.
2. **Relational tables** — minimal, for outbox/idempotency, quick indexes, and tenant policies.
3. **NoSQL (JSONB)** — ad-hoc blobs on vertices/edges (specs, inputs/outputs, state).

**Hobnob surface to preserve**

* Flow JSON format, step types, routing evaluators, `FlowRunner` execution model.
* Example: **“Fibonacci with Limericks”** flow in README.

---

## 1) Environment

* Python 3.13+, `uv` for deps.
* PostgreSQL 15+ with `age` extension enabled.
* Packages: `psycopg[binary,pool]`, `jsonschema`, `orjson`, `jmespath`, `tenacity`.
* AGE: `CREATE GRAPH hob; SET graph_path = hob;`.

---

## 2) Migrations

`sql/001_tristore.sql` (same as before).

---

## 3) Graph model

Vertices: `Flow`, `Step`, `Instance`, `Checkpoint`, `Event`.
Edges: `HAS_STEP`, `INITIAL`, `NEXT`, `OF`, `AT`, `RAN`, `PREV`, `EMITTED`.

---

## 4) Storage API

Create `hobnob/tristore.py`:

```python
class TriStore:
    def create_flow(self, flow_json: dict) -> str: ...
    def get_flow(self, name: str, semver: str) -> dict: ...
    def start_instance(self, name: str, semver: str,
                       state: dict, tenant: str) -> str: ...
    def current_checkpoint(self, instance_id: str) -> dict | None: ...
    def advance(self, instance_id: str, next_step_id: str, step_no: int,
                io: dict, fingerprints: dict, terminal: bool) -> dict: ...
    def append_event(self, instance_id: str, kind: str, payload: dict) -> None: ...
    def lineage(self, instance_id: str) -> list[dict]: ...
```

* Each method = single DB transaction with Cypher.
* `io` holds `{input_jsonb, output_jsonb, error_jsonb}`.
* `fingerprints` = `{prompt_hash, toolset_hash, model_fingerprint}`.

---

## 5) Runner integration

* Add `store: TriStore | None` and `persist_flow: bool`.
* When `persist_flow=True`, call `store.create_flow(flow_definition)`.
* Instances: `instance_id = store.start_instance(...)`.
* On each step, after execution → `store.advance(...)`.
* Replay CLI → `store.lineage(instance_id)`.

---

## 6) Example: Fibonacci with Limericks

```python
from hobnob import FlowRunner
from hobnob.tristore import TriStore
from langchain_openai import ChatOpenAI
from typing import TypedDict, List, Optional

class FibState(TypedDict):
    fib_sequence: List[int]
    last_number: int
    done: bool
    user_continue: str
    limerick: Optional[str]

FLOW_NAME = "fib_limericks"
FLOW_SEMVER = "1.0.0"

flow_definition = {
    "name": FLOW_NAME,
    "semver": FLOW_SEMVER,
    "system_prompt": "You are a creative mathematician and poet.",
    "steps": [
        {
            "name": "fib_and_limerick",
            "context": "The Fibonacci sequence: 1, 1, 2, 3, 5, 8, 13, ...",
            "instructions": "Create a limerick and calculate the next Fibonacci number",
            "output_format": "JSON with: limerick, fib_sequence, last_number, done",
            "prompt": (
                "Current sequence: {fib_sequence}\n"
                "Last number: {last_number}\n\n"
                "1. Write a limerick about {last_number}\n"
                "2. Calculate next Fibonacci number\n"
                "3. Set done=true if new number > 10"
            )
        },
        { "name": "ask_continue", "type": "user_input",
          "question": "Continue? (yes/no): " }
    ],
    "transitions": [
        {"from": "fib_and_limerick", "to": "ask_continue"},
        {"from": "ask_continue", "to": "fib_and_limerick",
         "condition": "user_continue == 'yes' and not done"},
        {"from": "ask_continue", "to": None,
         "condition": "user_continue == 'no' or done"}
    ],
    "initial_step": "fib_and_limerick"
}

def print_step(step_name, state):
    print(f"\n--- {step_name} ---")
    print(f"Sequence: {state.get('fib_sequence', [])}")
    if state.get("limerick"):
        print(f"Limerick:\n{state['limerick']}")

llm = ChatOpenAI(model="gpt-4o", temperature=0)
store = TriStore()  # new adapter
runner = FlowRunner(flow_definition, llm, FibState,
                    on_step=print_step,
                    persist_flow=True, store=store)

initial_state = {
    "fib_sequence": [1, 1],
    "last_number": 1,
    "done": False,
    "user_continue": "yes",
    "limerick": None
}

final_state = runner.run(initial_state, tenant="demo")
```

---

## 7) Replay

CLI:

```
hobnob replay --instance <uuid>
```

Uses `TriStore.lineage()` to walk checkpoints, re-execute steps, and compare outputs.

---

## 8) Acceptance criteria

* **Create flows**: Fibonacci flow persists as `:Flow` with `:Step`s, `:NEXT`, and `:INITIAL`.
* **Store flows**: JSON stored in `Flow.spec_jsonb`; transitions queryable via Cypher.
* **Replay flows**: Running `replay` reproduces outputs deterministically (with `temperature=0`).
* **Compatibility**: Old API works, only `store=` and `persist_flow=True` are added.


