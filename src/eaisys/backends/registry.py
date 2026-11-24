"""Provider factory for pluggable frontier & local open-weights VLM/LLM backends."""

from __future__ import annotations

import os

from eaisys.backends.anthropic_api import AnthropicBackend
from eaisys.backends.base import FrontierBackend
from eaisys.backends.codex_cli import CodexCLIBackend
from eaisys.backends.openai_api import OpenAICompatibleBackend


SUPPORTED_PROVIDERS = ("openai", "openrouter", "vllm", "anthropic", "codex")


def create_backend(
    provider: str = "openai",
    *,
    model: str | None = None,
    reasoning_effort: str = "high",
    fast_mode: bool = False,
    base_url: str | None = None,
    api_key: str | None = None,
) -> FrontierBackend:
    """Instantiate a configured model backend from provider name & environment."""
    p = provider.strip().lower()
    if p == "openai":
        return OpenAICompatibleBackend(
            model=model or os.environ.get("EAISYS_MODEL", "gpt-5.4"),
            base_url=base_url,
            api_key=api_key,
            reasoning_effort=reasoning_effort,
            fast_mode=fast_mode,
        )
    if p == "openrouter":
        return OpenAICompatibleBackend(
            model=model or os.environ.get("EAISYS_MODEL", "openai/gpt-5.4"),
            base_url=base_url or os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
            reasoning_effort=reasoning_effort,
            fast_mode=fast_mode,
        )
    if p == "vllm":
        return OpenAICompatibleBackend(
            model=model or os.environ.get("EAISYS_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"),
            base_url=base_url or os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
            api_key=api_key or os.environ.get("VLLM_API_KEY", "EMPTY"),
            reasoning_effort="",
            fast_mode=False,
        )
    if p == "anthropic":
        return AnthropicBackend(
            model=model or os.environ.get("EAISYS_MODEL", "claude-sonnet-4-5"),
            api_key=api_key,
            reasoning_effort=reasoning_effort,
        )
    if p == "codex":
        return CodexCLIBackend(
            model=model or os.environ.get("EAISYS_MODEL", "gpt-6.2-sol"),
            reasoning_effort=reasoning_effort or "max",
            fast_mode=fast_mode,
        )
    raise ValueError(f"Unsupported provider '{provider}'. Choose from: {', '.join(SUPPORTED_PROVIDERS)}")
