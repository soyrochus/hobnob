from __future__ import annotations
from typing import Dict, Any, Callable, List
from langgraph.graph import StateGraph, END
from hobnob.executors import LLMStep, UserInputStep
from hobnob.rendering import PromptRenderer
from hobnob.parsing import JsonParser


class FlowRunner:
    def __init__(
        self,
        flow_def: Dict[str, Any],
        llm,
        state_schema,
        on_step: Callable[[str, Dict[str, Any]], None] = None  # NEW
    ):
        self.flow_def = flow_def
        self.llm = llm
        self.state_schema = state_schema
        self.on_step = on_step
        self._graph = self._build_graph()
    
    def _make_executor(self, step_cfg):
        step_name = step_cfg["name"]
        stype = step_cfg.get("type", "llm")
        on_step = self.on_step
        if stype == "user_input":
            def _fn(state):
                out = UserInputStep(step_cfg.get("question", "Continue? (yes/no): "))(state)
                if on_step:
                    on_step(step_name, out)
                return out
            return _fn
        else:
            def _fn(state):
                out = LLMStep(
                    step_cfg,
                    self.llm,
                    renderer=PromptRenderer(self.flow_def.get("system_prompt", "")),
                    parser=JsonParser()
                )(state)
                if on_step:
                    on_step(step_name, out)
                return out
            return _fn

    def _router_factory(self, conds: List[Dict[str, Any]]):
        """
        Very small interpreter for boolean expressions of the form
        'user_continue == "yes" and not done'
        """
        def router(state: Dict[str, Any]):
            for tr in conds:
                cond = tr.get("condition")
                target = tr["to"]
                if cond is None:  # unconditional
                    return target or END
                try:
                    if eval(cond, {}, state):   # pragma: no cover (demo only)
                        return target or END
                except Exception:
                    pass
            return END
        return router

    def _build_graph(self):
        sg = StateGraph(self.state_schema)

        # nodes
        for step in self.flow_def["steps"]:
            sg.add_node(step["name"], self._make_executor(step))

        # transitions
        by_src = {}
        for tr in self.flow_def["transitions"]:
            by_src.setdefault(tr["from"], []).append(tr)

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

    # ---------- public API ----------
    def run(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        return self._graph.invoke(initial_state)
