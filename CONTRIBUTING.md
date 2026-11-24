<p align="right">
  <b>English</b> | <a href="./CONTRIBUTING.zh-CN.md">简体中文</a>
</p>

# Contributing to EAISys

Thanks for wanting to improve **EAISys**! We welcome new URDF robots, simulator integrations, zero-cheat plugins, manipulation playbook lessons, and post-training recipes.

## Ground Rules (Keep It Clean & Honest)

1. **Keep Robot Details in URDF, Not Core Code**: Add new robots via `robots/<id>/robot.urdf` + `eaisys synthesize`, rather than hard-coding robot-specific math inside `src/eaisys/`.
2. **Zero Simulator Cheats**: Online VLM observations and prompts must only use raw camera frames and hardware joint/gripper encoders—never object coordinates, table heights, rulers, or holding flags.
3. **Every Robot Must Pass Isolated Verification**: Any controller in `harnesses/embodiments/` must pass `uv run eaisys verify` (AST security check + isolated 6D translation, rotation, limit, and safety verification).
4. **Plugins Stay Independently Switchable**: Every new plugin in `src/eaisys/harness/plugins.py` needs a boolean toggle in `HarnessPluginConfig`.
5. **No Private Paths or Keys**: Never commit local machine paths, hostnames, tokens, or internal cluster names. `uv run pytest -v` runs an automated privacy audit to double-check for you.

## Quick Local Check Before Opening a PR

```bash
uv sync --extra dev
uv run pytest -v
uv run eaisys verify
```

To preview website changes locally:

```bash
uv run eaisys serve-site --host 127.0.0.1 --port 8765
```
