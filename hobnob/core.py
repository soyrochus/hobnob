from __future__ import annotations
from typing import Dict, Any, Optional, TypedDict, Set
import re
import warnings
from langgraph.graph import StateGraph, END
from hobnob.executors import ExecutorRegistry
from hobnob.rendering import PromptRenderer
from hobnob.parsing import JsonParser
from hobnob.routers import ConditionRouter, RouterRegistry

class FlowRunner:
    def __init__(
        self,
        flow_def,
        llm,
        state_schema: Optional[type] = None,
        on_step=None,
        condition_router: Optional[ConditionRouter] = None,
        infer_state: bool = True,
    ):
        self.flow_def = flow_def
        self.llm = llm
        self.on_step = on_step
        self.condition_router = condition_router or RouterRegistry.get("jmespath")
        self.infer_state = infer_state
        if state_schema is None:
            if infer_state:
                state_schema = self._infer_state_schema()
            else:
                raise ValueError("state_schema must be provided when infer_state is False")
        self.state_schema = state_schema
        self._known_fields: Set[str] = set(getattr(self.state_schema, "__annotations__", {}).keys())
        self._graph = self._build_graph()

    def _infer_state_schema(self) -> type[TypedDict]:
        fields: Set[str] = set()
        for step in self.flow_def.get("steps", []):
            prompt = step.get("prompt", "")
            fields.update(re.findall(r"{(.*?)}", prompt))
            for ex in step.get("examples", []):
                for part in ("input", "output"):
                    data = ex.get(part, {})
                    if isinstance(data, dict):
                        fields.update(data.keys())
        return TypedDict("InferredState", {f: Any for f in fields})

    def _validate_state(self, state: Dict[str, Any]):
        if not self.infer_state or not self._known_fields:
            return
        extra = set(state.keys()) - self._known_fields
        if extra:
            warnings.warn(f"Unexpected state fields: {sorted(extra)}")

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
        # Use the ExecutorRegistry to obtain a factory for this step type.
        # The factory signature is (cfg, runner) -> Executor.
        factory = ExecutorRegistry.get(stype)

        def _fn(state):
            executor = factory(step_cfg, self)
            out = executor(state)
            if on_step:
                on_step(step_name, out)
            self._validate_state(out)
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
        self._validate_state(initial_state)
        return self._graph.invoke(initial_state)
