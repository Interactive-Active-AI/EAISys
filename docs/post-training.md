<p align="right">
  <b>English</b> | <a href="./post-training.zh-CN.md">简体中文</a>
</p>

# Browser Teleop, Gym Interface & Fine-Tuning Export

Frontier VLMs let you get working 3D trajectories on day one—and EAISys makes it effortless to turn those trajectories (or human browser demos) into training data for smaller, faster open-weights models.

## 1. Export Verified 3D Runs for SFT or GRPO

Every solved episode under `artifacts/live_runs/` stores the raw URDF, multi-camera images, zero-cheat prompts, structured skill actions, and final task success. Export them in one command:

```bash
# LLaMA-Factory ShareGPT format (ready for Qwen-VL / LLaVA LoRA SFT)
uv run eaisys export-sft --format llamafactory --out artifacts/posttrain/sft.jsonl

# OpenAI fine-tuning JSONL format
uv run eaisys export-sft --format openai --out artifacts/posttrain/openai.jsonl

# GRPO / RLVR episode bundles with binary success rewards
uv run eaisys export-sft --format grpo --out artifacts/posttrain/grpo.jsonl
```

Each exported sample automatically includes a compact **URDF Kinematic Card** (robot name, DoF, joint types, and joint list) extracted straight from the URDF so fine-tuned models learn to condition on body morphology.

## 2. GUMI-URDF: Drive Any Robot in Your Browser

Want to test an AI-generated URDF controller yourself or collect clean demos without real hardware? Launch the built-in web teleop server:

```bash
# Live 3D MuJoCo control in your browser
uv run --extra sim eaisys gumi --robot ur5e_mujoco --env Stack --robosuite-robot UR5e --port 8766

# Lightweight offline preview (no MuJoCo needed)
uv run eaisys gumi --robot ur5e --port 8766
```

Open `http://127.0.0.1:8766/`, control the arm with `W/A/S/D/Q/E`, `G/R` (grasp/release), and `1/2/3` (precision gears), and click **Save Demo JSON** to save recordings under `artifacts/gumi_demos/`.

## 3. Standard Gymnasium Wrapper

Plug any EAISys robot environment directly into RL loops or benchmarks using `EAISysGymEnv`:

```python
from eaisys import EAISysGymEnv, load_interpreter_module, make_standard_pick_place_env

interp = load_interpreter_module("harnesses/embodiments/ur5e.py", "robots/ur5e/robot.urdf")
gym_env = EAISysGymEnv(make_standard_pick_place_env(interp))
obs, info = gym_env.reset()
obs, reward, terminated, truncated, info = gym_env.step("MV_FWD")
```
