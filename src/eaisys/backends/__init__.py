"""Pluggable frontier & local VLM/LLM backends for URDF synthesis and closed-loop control."""

from eaisys.backends.anthropic_api import AnthropicBackend
from eaisys.backends.base import FrontierBackend, ModelResponse
from eaisys.backends.codex_cli import CodexCLIBackend, resolve_codex_binary
from eaisys.backends.openai_api import OpenAICompatibleBackend
from eaisys.backends.registry import SUPPORTED_PROVIDERS, create_backend

__all__ = [
    "AnthropicBackend",
    "CodexCLIBackend",
    "FrontierBackend",
    "ModelResponse",
    "OpenAICompatibleBackend",
    "SUPPORTED_PROVIDERS",
    "create_backend",
    "resolve_codex_binary",
]
