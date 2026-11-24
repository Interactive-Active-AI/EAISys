"""Generic privacy & secret sanitizer for recorded agent prompts, outputs, and telemetry."""

from __future__ import annotations

from functools import lru_cache
import getpass
import os
import re
import socket
from typing import Any


_PUBLIC_ALLOWLIST = {"github.com", "localhost", "127.0.0.1", "0.0.0.0", "git", "root", "user", "runner"}


@lru_cache(maxsize=1)
def _compiled_rules() -> list[tuple[re.Pattern[str], str]]:
    rules: list[tuple[re.Pattern[str], str]] = [
        # Generic OS user & temporary directories
        (re.compile(r"/[Uu]sers/[^/\s\"']+"), "<USER_HOME>"),
        (re.compile(r"/home/[^/\s\"']+"), "<USER_HOME>"),
        (re.compile(r"[A-Za-z]:\\[Uu]sers\\[^\\\s\"']+"), "<USER_HOME>"),
        (re.compile(r"/private/var/folders/[^\s\"']+"), "<TMP_DIR>"),
        (re.compile(r"/var/folders/[^\s\"']+"), "<TMP_DIR>"),
        (re.compile(r"/dev/shm/[^\s\"']*"), "<SHM_TMP>"),
        # Common API keys & personal access tokens
        (re.compile(r"\bsk-(?:proj-|ant-|or-v1-)?[A-Za-z0-9_-]{16,}\b"), "<REDACTED_KEY>"),
        (re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{16,}\b"), "<REDACTED_TOKEN>"),
        (re.compile(r"\bhf_[A-Za-z0-9]{16,}\b"), "<REDACTED_TOKEN>"),
        (re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"), "<REDACTED_KEY>"),
        (re.compile(r"(?i)(Bearer\s+)[A-Za-z0-9._-]{20,}"), r"\1<REDACTED_TOKEN>"),
    ]

    dyn_tokens: set[str] = set()
    try:
        user = getpass.getuser().strip()
        if len(user) >= 3 and user.lower() not in _PUBLIC_ALLOWLIST:
            dyn_tokens.add(user)
    except Exception:
        pass

    try:
        host = socket.gethostname().strip()
        for candidate in (host, host.split(".")[0]):
            if len(candidate) >= 3 and candidate.lower() not in _PUBLIC_ALLOWLIST:
                dyn_tokens.add(candidate)
    except Exception:
        pass

    extra = os.environ.get("EAISYS_REDACT_PATTERNS", "").strip()
    if extra:
        for item in extra.split(","):
            tok = item.strip()
            if len(tok) >= 3:
                dyn_tokens.add(tok)

    for tok in sorted(dyn_tokens, key=len, reverse=True):
        rules.append((re.compile(rf"\b{re.escape(tok)}\b", re.IGNORECASE), "<REDACTED_HOST>"))

    return rules


def sanitize_text(text: str) -> str:
    """Redact local filesystem paths, credentials, and host identifiers from text."""
    out = str(text)
    for pat, repl in _compiled_rules():
        out = pat.sub(repl, out)
    return out


def sanitize_payload(obj: Any) -> Any:
    """Recursively sanitize all strings and dictionary keys in a JSON-serializable payload."""
    if isinstance(obj, str):
        return sanitize_text(obj)
    if isinstance(obj, list):
        return [sanitize_payload(x) for x in obj]
    if isinstance(obj, tuple):
        return [sanitize_payload(x) for x in obj]
    if isinstance(obj, dict):
        return {sanitize_text(str(k)): sanitize_payload(v) for k, v in obj.items()}
    return obj
