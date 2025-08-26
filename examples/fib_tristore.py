from __future__ import annotations

import json
from typing import TypedDict, List, Optional

from langchain_openai import ChatOpenAI

from hobnob import FlowRunner
from hobnob.tristore import TriStore


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
            ),
        },
        {"name": "ask_continue", "type": "user_input", "question": "Continue? (yes/no): "},
    ],
    "transitions": [
        {"from": "fib_and_limerick", "to": "ask_continue"},
        {"from": "ask_continue", "to": "fib_and_limerick", "condition": "user_continue == 'yes' and not done"},
        {"from": "ask_continue", "to": None, "condition": "user_continue == 'no' or done"},
    ],
    "initial_step": "fib_and_limerick",
}


def print_step(step_name, state):
    print(f"\n--- {step_name} ---")
    print(f"Sequence: {state.get('fib_sequence', [])}")
    if state.get("limerick"):
        print(f"Limerick:\n{state['limerick']}")


def main() -> int:
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    store = TriStore()  # reads DSN from HOBNOB_DATABASE_URL or DATABASE_URL
    runner = FlowRunner(
        flow_definition,
        llm,
        FibState,
        on_step=print_step,
        persist_flow=True,
        store=store,
    )

    initial_state: FibState = {
        "fib_sequence": [1, 1],
        "last_number": 1,
        "done": False,
        "user_continue": "yes",
        "limerick": None,
    }

    final_state = runner.run(initial_state, tenant="demo")
    print("\n==== FINAL STATE ====")
    print(json.dumps(final_state, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

