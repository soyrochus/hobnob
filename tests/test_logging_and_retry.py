import json
import logging
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, TypedDict

import pytest

from hobnob.core import FlowRunner


class DummyState(TypedDict):
    foo: str


class FakeLLM:
    """Simple LLM stub that returns predetermined results or raises errors."""

    def __init__(self, responses: List[Any]):
        self._responses: Iterator[Any] = iter(responses)
        self.calls = 0

    def invoke(self, prompt: str) -> Any:  # pragma: no cover - trivial wrapper
        self.calls += 1
        result = next(self._responses)
        if isinstance(result, Exception):
            raise result
        return SimpleNamespace(content=json.dumps(result))


def build_flow(retry: Dict[str, Any] | None = None) -> Dict[str, Any]:
    step: Dict[str, Any] = {"name": "test", "prompt": "say something"}
    if retry is not None:
        step["retry"] = retry
    return {
        "steps": [step],
        "transitions": [{"from": "test", "to": None}],
        "initial_step": "test",
    }


def test_step_logging(caplog: pytest.LogCaptureFixture) -> None:
    llm = FakeLLM([{"foo": "baz"}])
    runner = FlowRunner(build_flow(), llm, DummyState)

    with caplog.at_level(logging.INFO):
        result = runner.run({"foo": "bar"})

    assert result["foo"] == "baz"
    msgs = [r.message for r in caplog.records]
    assert any("Step test input" in m for m in msgs)
    assert any("Step test output" in m for m in msgs)


def test_retry_backoff(caplog: pytest.LogCaptureFixture) -> None:
    llm = FakeLLM([ValueError("boom"), {"foo": "ok"}])
    runner = FlowRunner(build_flow({"max_attempts": 2, "backoff": 0}), llm, DummyState)

    with caplog.at_level(logging.WARNING):
        result = runner.run({"foo": "bar"})

    assert result["foo"] == "ok"
    assert llm.calls == 2
    assert any("Retrying step test" in r.message for r in caplog.records)
