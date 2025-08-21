from __future__ import annotations

from typing import Any, Dict, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .parsing import JsonParser


def from_prompt(prompt: str, llm: Optional[BaseChatModel] = None) -> Dict[str, Any]:
    """Generate a Hobnob flow definition from natural language.

    Parameters
    ----------
    prompt:
        Natural language description of the workflow.
    llm:
        Optional LangChain chat model. If omitted, ``ChatOpenAI`` is used.

    Returns
    -------
    Dict[str, Any]
        Flow definition compatible with :class:`hobnob.core.FlowRunner`.
    """
    llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)
    system_prompt = (
        "You convert natural language workflow descriptions into a JSON object "
        "for the Hobnob framework. The JSON must contain 'system_prompt' (optional), "
        "'steps' (list of step objects), 'transitions' (list of transitions), and "
        "'initial_step'. Each step object requires a 'name' and may include 'type', "
        "'context', 'instructions', 'output_format', 'examples', 'prompt', or 'question'. "
        "Transitions need 'from' and 'to' (null to end) and optional 'condition'. "
        "Return ONLY valid JSON without additional commentary."
    )
    result = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt),
    ])
    return JsonParser().parse(result.content)
