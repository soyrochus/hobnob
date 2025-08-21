from __future__ import annotations
from typing import Any, Dict, Protocol
from langchain_core.language_models import BaseChatModel
from hobnob.rendering import PromptRenderer
from hobnob.parsing import JsonParser


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
