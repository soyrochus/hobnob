from __future__ import annotations

import json
import re
from typing import Any, Dict


class JsonParser:
    def parse(self, text: str) -> Dict[str, Any]:
        """
        Returns a dict if it finds a JSON object; otherwise raises ValueError.
        """
        match = re.search(r"\{.*\}", text, re.DOTALL)
        target = match.group(0) if match else text
        return json.loads(target)
