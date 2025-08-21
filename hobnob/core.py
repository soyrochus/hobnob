from __future__ import annotations
from typing import Any, Dict, Optional
from langgraph.graph import StateGraph, END  # type: ignore[import-not-found]
from hobnob.executors import ExecutorRegistry
from hobnob.routers import ConditionRouter, RouterRegistry

class FlowRunner:
    def __init__(self, flow_def, llm, state_schema, on_step=None, condition_router: Optional[ConditionRouter] = None):
        self.flow_def = flow_def
        self.llm = llm
        self.state_schema = state_schema
        self.on_step = on_step
        self.condition_router = condition_router or RouterRegistry.get("jmespath")
        self._graph = self._build_graph()

    def _router_factory(self, conds):
        router = self.condition_router

        def _route(state):
            for tr in conds:
                cond = tr.get("condition")
                target = tr["to"]
                if cond is None:
                    return target or END
                try:
                    if router.check(cond, state):
                        return target or END
                except Exception as e:
                    print(f"Condition routing error: {e}")
            return END
        return _route
    
    def _make_executor(self, step_cfg):
        step_name = step_cfg["name"]
        stype = step_cfg.get("type", "llm")
        on_step = self.on_step
        factory = ExecutorRegistry.get(stype)
        executor = factory(step_cfg, self)

        def _fn(state):
            out = executor(state)
            if on_step:
                on_step(step_name, out)
            return out

        return _fn

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
