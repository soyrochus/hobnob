from typing import Any, Dict, TypedDict

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import hobnob.executors as executors
from hobnob import FlowRunner


class SearchState(TypedDict, total=False):
    query: str
    results: str


def test_web_search_step(monkeypatch: Any) -> None:
    def fake_get(url: str, params: Dict[str, Any]):
        class Resp:
            def json(self) -> Dict[str, Any]:
                return {"AbstractText": "summary"}

        return Resp()

    monkeypatch.setattr(executors.requests, "get", fake_get)

    flow_def = {
        "steps": [
            {"name": "search", "type": "web_search", "query_key": "query", "result_key": "results"}
        ],
        "transitions": [{"from": "search", "to": None}],
        "initial_step": "search",
    }

    runner = FlowRunner(flow_def, llm=None, state_schema=SearchState)
    result = runner.run({"query": "python"})

    assert result["results"] == "summary"


class APIState(TypedDict, total=False):
    joke: Any


def test_api_call_step(monkeypatch: Any) -> None:
    def fake_request(
        method: str,
        url: str,
        params: Dict[str, Any] | None = None,
        json: Dict[str, Any] | None = None,
    ):
        class Resp:
            def json(self) -> Dict[str, Any]:
                return {"joke": "funny"}

        return Resp()

    monkeypatch.setattr(executors.requests, "request", fake_request)

    flow_def = {
        "steps": [
            {"name": "joke", "type": "api_call", "url": "https://example.com", "result_key": "joke"}
        ],
        "transitions": [{"from": "joke", "to": None}],
        "initial_step": "joke",
    }

    runner = FlowRunner(flow_def, llm=None, state_schema=APIState)
    result = runner.run({})

    assert result["joke"] == {"joke": "funny"}

