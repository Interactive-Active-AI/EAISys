"""OpenAI-compatible multimodal VLM/LLM API backend (OpenAI, OpenRouter, vLLM, SGLang, Gemini-compat)."""

from __future__ import annotations

import base64
import os
from pathlib import Path
import time
from typing import Any

from eaisys.backends.base import ModelResponse


class OpenAICompatibleBackend:
    """Calls any OpenAI-compatible multimodal endpoint with structured JSON & retry support."""

    def __init__(
        self,
        model: str = "gpt-5.4",
        base_url: str | None = None,
        api_key: str | None = None,
        reasoning_effort: str | None = "high",
        fast_mode: bool = False,
        max_retries: int = 2,
    ) -> None:
        self.model_name = model
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL") or None
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or None
        self.reasoning_effort = reasoning_effort or ""
        self.fast_mode = bool(fast_mode)
        self.max_retries = max(0, int(max_retries))

    def generate(
        self,
        prompt: str,
        *,
        images: list[str | Path] | None = None,
        json_schema: dict[str, Any] | None = None,
        timeout_s: float = 240.0,
    ) -> ModelResponse:
        from openai import OpenAI

        t0 = time.monotonic()
        client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=timeout_s)
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in images or ():
            p = Path(img)
            b64 = base64.b64encode(p.read_bytes()).decode("ascii")
            mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                }
            )

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
        }
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.fast_mode:
            kwargs["service_tier"] = "priority"
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "eaisys_step", "schema": json_schema, "strict": True},
            }

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = client.chat.completions.create(**kwargs)
                elapsed = time.monotonic() - t0
                text = resp.choices[0].message.content or ""
                usage = resp.usage
                toks = int(usage.total_tokens) if usage and usage.total_tokens else 0
                raw_meta: dict[str, Any] = {
                    "provider": "openai_compatible",
                    "service_tier": "priority" if self.fast_mode else "default",
                }
                if usage:
                    raw_meta["input_tokens"] = int(getattr(usage, "prompt_tokens", 0) or 0)
                    raw_meta["output_tokens"] = int(getattr(usage, "completion_tokens", 0) or 0)
                return ModelResponse(
                    text=text,
                    model=self.model_name,
                    reasoning_effort=self.reasoning_effort,
                    tokens_used=toks,
                    elapsed_s=round(elapsed, 3),
                    raw_meta=raw_meta,
                )
            except Exception as exc:
                last_exc = exc
                # Some OpenAI-compatible servers (vLLM / older gateways) reject strict json_schema or service_tier.
                msg = str(exc).lower()
                if "service_tier" in msg and "service_tier" in kwargs:
                    kwargs.pop("service_tier", None)
                    continue
                if "reasoning_effort" in msg and "reasoning_effort" in kwargs:
                    kwargs.pop("reasoning_effort", None)
                    continue
                if "response_format" in msg and "response_format" in kwargs:
                    kwargs["response_format"] = {"type": "json_object"}
                    continue
                if attempt < self.max_retries:
                    time.sleep(min(4.0, 0.8 * (2**attempt)))
                    continue
                raise

        assert last_exc is not None
        raise last_exc
