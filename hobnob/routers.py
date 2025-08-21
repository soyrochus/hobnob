# routers.py

from typing import Any, Dict, Protocol
import jmespath

class ConditionRouter(Protocol):
    def check(self, condition: str, state: Dict[str, Any]) -> bool: ...

class JMESPathRouter:
    """Safe condition router using JMESPath expressions."""
    def check(self, condition: str, state: Dict[str, Any]) -> bool:
        # JMESPath returns actual value, so we treat truthy as True
        result = jmespath.search(condition, state)
        return bool(result)

class EvalRouter:
    """UNSAFE: Only for trusted inputs! Must be explicitly enabled."""
    enabled = False  # set to True to allow usage

    def check(self, condition: str, state: Dict[str, Any]) -> bool:
        if not self.enabled:
            raise RuntimeError(
                "EvalRouter is disabled for safety. Set EvalRouter.enabled = True to use (not recommended for production)."
            )
        return bool(eval(condition, {}, state))

# routers.py (continued)

class RouterRegistry:
    _routers = {
        "jmespath": JMESPathRouter(),
        "eval": EvalRouter(),
    }

    @classmethod
    def get(cls, name: str) -> ConditionRouter:
        if name not in cls._routers:
            raise ValueError(f"Unknown router: {name}")
        return cls._routers[name]

    @classmethod
    def register(cls, name: str, router: ConditionRouter):
        cls._routers[name] = router
