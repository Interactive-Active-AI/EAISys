"""Native Anthropic Claude multimodal VLM/LLM API backend."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import time
from typing import Any

from eaisys.backends.base import ModelResponse


class AnthropicBackend:
    """Calls Anthropic Messages API with multimodal image blocks & JSON guidance."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        api_key: str | None = None,
        max_tokens: int = 4096,
        reasoning_effort: str = "",
    ) -> None:
        self.model_name = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY") or None
        self.max_tokens = int(max_tokens)
        self.reasoning_effort = reasoning_effort
        self.fast_mode = False

    def generate(
        self,
        prompt: str,
        *,
        images: list[str | Path] | None = None,
        json_schema: dict[str, Any] | None = None,
        timeout_s: float = 240.0,
    ) -> ModelResponse:
        from anthropic import Anthropic

        t0 = time.monotonic()
        client = Anthropic(api_key=self.api_key, timeout=timeout_s)
        blocks: list[dict[str, Any]] = []
        for img in images or ():
            p = Path(img)
            b64 = base64.b64encode(p.read_bytes()).decode("ascii")
            mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
            blocks.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": mime, "data": b64},
                }
            )

        full_prompt = prompt
        if json_schema is not None:
            full_prompt += (
                "\n\nReturn ONLY a valid JSON object strictly conforming to this schema:\n"
                + json.dumps(json_schema, indent=2)
            )
        blocks.append({"type": "text", "text": full_prompt})

        resp = client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": blocks}],
        )
        elapsed = time.monotonic() - t0
        parts = [getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text"]
        text = "\n".join(parts).strip()
        in_tok = int(getattr(resp.usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(resp.usage, "output_tokens", 0) or 0)
        return ModelResponse(
            text=text,
            model=self.model_name,
            reasoning_effort=self.reasoning_effort,
            tokens_used=in_tok + out_tok,
            elapsed_s=round(elapsed, 3),
            raw_meta={
                "provider": "anthropic",
                "input_tokens": in_tok,
                "output_tokens": out_tok,
            },
        )
