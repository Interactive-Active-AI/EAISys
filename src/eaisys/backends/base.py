"""Protocol and response structures for pluggable Frontier VLM/LLM backends."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Protocol


@dataclass
class ModelResponse:
    text: str
    model: str
    reasoning_effort: str = ""
    tokens_used: int = 0
    elapsed_s: float = 0.0
    raw_meta: dict[str, Any] = field(default_factory=dict)

    def parse_json(self) -> dict[str, Any]:
        """Extract and parse a JSON object from raw or markdown-fenced model output."""
        s = self.text.strip()
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return json.loads(m.group(1))
        first = s.find("{")
        last = s.rfind("}")
        if first != -1 and last != -1 and last > first:
            return json.loads(s[first : last + 1])
        return json.loads(s)

    def extract_python_code(self) -> str:
        """Extract Python source code from markdown fences (or raw text if unfenced)."""
        s = self.text.strip()
        m = re.search(r"```python\s*(.*?)```", s, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        m2 = re.search(r"```\s*(.*?)```", s, flags=re.DOTALL)
        if m2:
            return m2.group(1).strip()
        return s


class FrontierBackend(Protocol):
    model_name: str

    def generate(
        self,
        prompt: str,
        *,
        images: list[str | Path] | None = None,
        json_schema: dict[str, Any] | None = None,
        timeout_s: float = 240.0,
    ) -> ModelResponse: ...
