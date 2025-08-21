from __future__ import annotations
from typing import Dict, Any, Protocol
from langchain_core.language_models import BaseChatModel
from hobnob.rendering import PromptRenderer
from hobnob.parsing import JsonParser
import requests  # type: ignore[import]


class Executor(Protocol):
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]: ...


class LLMStep:
    def __init__(self, cfg: Dict[str, Any],
                 llm: BaseChatModel,
                 renderer: PromptRenderer | None = None,
                 parser: JsonParser | None = None):
        self.cfg, self.llm = cfg, llm
        self.renderer = renderer or PromptRenderer()
        self.parser   = parser   or JsonParser()

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self.renderer.render(self.cfg, state)
        result = self.llm.invoke(prompt)
        updates = self.parser.parse(result.content)
        return {**state, **updates}


class UserInputStep:
    """Blocking console prompt.  Replace with UI handler in real apps."""
    def __init__(self, question: str):
        self.question = question

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        while True:
            ans = input(self.question).strip().lower()
            if ans in ("yes", "no"):
                return {**state, "user_continue": ans}
            print("Please answer 'yes' or 'no'.")


class ExecutorRegistry:
    """Registry for creating step executors by name."""

    _executors: Dict[str, Callable[[Dict[str, Any], Any], Executor]] = {}

    @classmethod
    def register(
        cls, name: str, factory: Callable[[Dict[str, Any], Any], Executor]
    ) -> None:
        cls._executors[name] = factory

    @classmethod
    def get(cls, name: str) -> Callable[[Dict[str, Any], Any], Executor]:
        try:
            return cls._executors[name]
        except KeyError as exc:
            raise KeyError(f"Unknown executor type: {name}") from exc


def _llm_factory(cfg: Dict[str, Any], runner: Any) -> Executor:
    return LLMStep(
        cfg,
        runner.llm,
        renderer=PromptRenderer(runner.flow_def.get("system_prompt", "")),
        parser=JsonParser(),
    )


def _user_input_factory(cfg: Dict[str, Any], _runner: Any) -> Executor:
    return UserInputStep(cfg.get("question", "Continue? (yes/no): "))


class WebSearchStep:
    """Example step that performs a DuckDuckGo search."""

    def __init__(self, cfg: Dict[str, Any]):
        self.query_key = cfg.get("query_key", "query")
        self.result_key = cfg.get("result_key", "results")

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = state.get(self.query_key, "")
        if not query:
            return state
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_redirect": 1, "no_html": 1},
        )
        data = resp.json()
        text = data.get("AbstractText") or ""
        return {**state, self.result_key: text}


def _web_search_factory(cfg: Dict[str, Any], _runner: Any) -> Executor:
    return WebSearchStep(cfg)


class APICallStep:
    """Generic HTTP API call step."""

    def __init__(self, cfg: Dict[str, Any]):
        self.url = cfg["url"]
        self.method = cfg.get("method", "get").lower()
        self.params = cfg.get("params", {})
        self.result_key = cfg.get("result_key", "api_result")

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        params = {
            k: v.format(**state) if isinstance(v, str) else v for k, v in self.params.items()
        }
        resp = requests.request(
            self.method,
            self.url,
            params=params if self.method == "get" else None,
            json=params if self.method != "get" else None,
        )
        try:
            data: Any = resp.json()
        except Exception:
            data = resp.text
        return {**state, self.result_key: data}


def _api_call_factory(cfg: Dict[str, Any], _runner: Any) -> Executor:
    return APICallStep(cfg)


# Register built-in executors
ExecutorRegistry.register("llm", _llm_factory)
ExecutorRegistry.register("user_input", _user_input_factory)
ExecutorRegistry.register("web_search", _web_search_factory)
ExecutorRegistry.register("api_call", _api_call_factory)
