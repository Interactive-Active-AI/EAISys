"""Optional Codex CLI frontier model backend with multimodal image & schema support."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any

from eaisys.backends.base import ModelResponse


def resolve_codex_binary(explicit_path: str | None = None) -> str:
    """Resolve the `codex` CLI binary from an explicit argument, `EAISYS_CODEX_BIN`, or `$PATH`."""
    if explicit_path and Path(explicit_path).is_file():
        return explicit_path
    env_bin = os.environ.get("EAISYS_CODEX_BIN", "").strip()
    if env_bin and Path(env_bin).is_file():
        return env_bin
    on_path = shutil.which("codex")
    if on_path:
        return on_path
    raise FileNotFoundError(
        "Codex CLI binary not found on $PATH. Install the `codex` CLI or set `EAISYS_CODEX_BIN`, "
        "or switch to an API provider (`--provider openai` / `openrouter` / `anthropic`)."
    )


class CodexCLIBackend:
    """Optional frontier VLM/LLM adapter invoking `codex exec` with image & JSON schema support."""

    def __init__(
        self,
        model: str = "gpt-6.2-sol",
        reasoning_effort: str = "max",
        fast_mode: bool = True,
        codex_bin: str | None = None,
        ephemeral: bool = True,
    ) -> None:
        self.model_name = model
        self.reasoning_effort = reasoning_effort
        self.fast_mode = bool(fast_mode)
        self.codex_bin = resolve_codex_binary(codex_bin)
        self.ephemeral = ephemeral

    def generate(
        self,
        prompt: str,
        *,
        images: list[str | Path] | None = None,
        json_schema: dict[str, Any] | None = None,
        timeout_s: float = 480.0,
        max_retries: int = 3,
    ) -> ModelResponse:
        t0 = time.monotonic()
        tier = "priority" if self.fast_mode else "default"
        last_err: Exception | None = None

        for attempt in range(1, max(1, int(max_retries)) + 1):
            with tempfile.TemporaryDirectory(prefix="eaisys_codex_") as tmpdir:
                td = Path(tmpdir)
                out_file = td / "last_message.txt"
                cmd = [
                    self.codex_bin,
                    "exec",
                    "-C",
                    str(td),
                    "--skip-git-repo-check",
                    "--ignore-user-config",
                    "--ignore-rules",
                    "-m",
                    self.model_name,
                    "-c",
                    f'model_reasoning_effort="{self.reasoning_effort}"',
                    "-c",
                    f'service_tier="{tier}"',
                    "--json",
                    "-o",
                    str(out_file),
                ]
                if self.ephemeral:
                    cmd.append("--ephemeral")

                if json_schema is not None:
                    schema_file = td / "schema.json"
                    schema_file.write_text(json.dumps(json_schema), encoding="utf-8")
                    cmd.extend(["--output-schema", str(schema_file)])

                for img in images or ():
                    cmd.extend(["-i", str(Path(img).resolve())])

                guarded_prompt = (
                    "IMPORTANT: Do NOT run any shell commands. "
                    "Reason directly over the prompt and any attached images, then write the final answer directly.\n\n"
                    + prompt
                )
                cmd.append("-")

                try:
                    proc = subprocess.run(
                        cmd,
                        input=guarded_prompt,
                        capture_output=True,
                        text=True,
                        timeout=timeout_s,
                    )
                except subprocess.TimeoutExpired as exc:
                    last_err = exc
                    if attempt < max_retries:
                        time.sleep(3.0)
                        continue
                    raise

                elapsed = time.monotonic() - t0

                if proc.returncode != 0 and not out_file.is_file():
                    last_err = RuntimeError(
                        f"codex exec failed (rc={proc.returncode}, attempt={attempt}/{max_retries}):\n"
                        f"STDOUT:\n{proc.stdout[-1200:]}\nSTDERR:\n{proc.stderr[-1200:]}"
                    )
                    if attempt < max_retries:
                        time.sleep(3.0)
                        continue
                    raise last_err

                text = out_file.read_text(encoding="utf-8").strip() if out_file.is_file() else ""
                input_tok = 0
                output_tok = 0
                reasoning_tok = 0
                for line in proc.stdout.splitlines():
                    line = line.strip()
                    if not line.startswith("{"):
                        continue
                    try:
                        ev = json.loads(line)
                    except Exception:
                        continue
                    if ev.get("type") == "item.completed":
                        item = ev.get("item", {})
                        if item.get("type") == "agent_message" and not text:
                            text = str(item.get("text", "")).strip()
                    elif ev.get("type") == "turn.completed":
                        u = ev.get("usage", {})
                        input_tok += int(u.get("input_tokens", 0))
                        output_tok += int(u.get("output_tokens", 0))
                        reasoning_tok += int(u.get("reasoning_output_tokens", 0))

                if not text and attempt < max_retries:
                    last_err = RuntimeError(f"codex exec returned empty text on attempt {attempt}/{max_retries}")
                    time.sleep(2.0)
                    continue

                return ModelResponse(
                    text=text,
                    model=self.model_name,
                    reasoning_effort=self.reasoning_effort,
                    tokens_used=input_tok + output_tok,
                    elapsed_s=round(elapsed, 3),
                    raw_meta={
                        "provider": "codex_cli",
                        "service_tier": tier,
                        "attempts": attempt,
                        "input_tokens": input_tok,
                        "output_tokens": output_tok,
                        "reasoning_output_tokens": reasoning_tok,
                    },
                )

        if last_err is not None:
            raise last_err
        raise RuntimeError("codex exec failed without response")
