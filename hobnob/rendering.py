from __future__ import annotations
import json
from typing import Dict, Any


class PromptRenderer:
    """Renders a step prompt given its config and the live state."""
    def __init__(self, system_prompt: str | None = None) -> None:
        self.system_prompt = system_prompt or ""

    def render(self, step_cfg: Dict[str, Any], state: Dict[str, Any]) -> str:
        parts: list[str] = []
        if self.system_prompt:
            parts.append(f"SYSTEM: {self.system_prompt}\n")
        if ctx := step_cfg.get("context"):
            parts.append(f"CONTEXT: {ctx}\n")
        if exs := step_cfg.get("examples"):
            parts.append("EXAMPLES:")
            for i, ex in enumerate(exs, 1):
                parts.append(f"\nExample {i}:")
                parts.append(f"\nInput: {json.dumps(ex['input'], indent=2)}")
                parts.append(f"\nOutput: {json.dumps(ex['output'], indent=2)}\n")
        if instr := step_cfg.get("instructions"):
            parts.append(f"INSTRUCTIONS: {instr}\n")
        if ofmt := step_cfg.get("output_format"):
            parts.append(f"OUTPUT FORMAT: {ofmt}\n")
        if prompt := step_cfg.get("prompt"):
            parts.append("CURRENT TASK:\n" + prompt.format(**state))
        return "\n".join(parts).strip()
