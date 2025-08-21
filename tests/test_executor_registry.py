from typing import Any, Dict, TypedDict

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from hobnob import ExecutorRegistry, FlowRunner


class State(TypedDict):
    n: int


def test_custom_executor_registration() -> None:
    def add_one_factory(_cfg: Dict[str, Any], _runner: Any):
        def _step(state: Dict[str, Any]) -> Dict[str, Any]:
            return {**state, "n": state["n"] + 1}

        return _step

    ExecutorRegistry.register("add_one", add_one_factory)

    flow_def = {
        "steps": [{"name": "inc", "type": "add_one"}],
        "transitions": [{"from": "inc", "to": None}],
        "initial_step": "inc",
    }

    runner = FlowRunner(flow_def, llm=None, state_schema=State)
    result = runner.run({"n": 1})

    assert result["n"] == 2

