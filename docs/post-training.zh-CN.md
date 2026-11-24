<p align="right">
  <a href="./post-training.md">English</a> | <b>简体中文</b>
</p>

# 网页遥操、Gym 接口与微调数据导出

用前沿多模态大模型可以在第一天直接跑出能用的 3D 轨迹；之后你还可以把这些轨迹（或者自己用浏览器遥控采的示教）一键转成小模型微调数据集，让更轻、更快的开源模型也能学会跨身体操作。

## 1. 一键导出 SFT / GRPO 训练数据

`artifacts/live_runs/` 里每一条通关的 3D 零作弊记录都完整保存了裸 URDF、多视角原图、零作弊提示词、结构化技能输出和最终成败奖励，直接导出即可开训：

```bash
# LLaMA-Factory ShareGPT 多模态格式（可直接用于 Qwen-VL / LLaVA LoRA 微调）
uv run eaisys export-sft --format llamafactory --out artifacts/posttrain/sft.jsonl

# OpenAI 官方微调 JSONL 格式
uv run eaisys export-sft --format openai --out artifacts/posttrain/openai.jsonl

# GRPO / RLVR 强化学习整条轨迹包（含 0/1 物理成败奖励）
uv run eaisys export-sft --format grpo --out artifacts/posttrain/grpo.jsonl
```

导出器默认会从 URDF 自动抽取一张紧凑的 **URDF 运动学名片**（机器人名、自由度、各关节类型与名称）附在每条样本开头，让微调后的小模型也能看懂机身结构。

## 2. GUMI-URDF：打开浏览器直接遥控任意机器人

想亲自试开一下 AI 刚合成的 URDF 控制器，或者免真机采集高质量示教？启动内置网页遥操服务即可：

```bash
# 连接 3D MuJoCo 实时网页遥操
uv run --extra sim eaisys gumi --robot ur5e_mujoco --env Stack --robosuite-robot UR5e --port 8766

# 轻量离线模式（不装 MuJoCo 也能玩）
uv run eaisys gumi --robot ur5e --port 8766
```

浏览器打开 `http://127.0.0.1:8766/`，用键盘 `W/A/S/D/Q/E` 移动、`G/R` 抓放、`1/2/3` 换挡，点击 **Save Demo JSON** 即可把示教存到 `artifacts/gumi_demos/`。

## 3. 标准 Gymnasium 强化学习封装

用 `EAISysGymEnv` 可以把任意 EAISys 机器人环境直接接入标准 RL 训练或评测循环：

```python
from eaisys import EAISysGymEnv, load_interpreter_module, make_standard_pick_place_env

interp = load_interpreter_module("harnesses/embodiments/ur5e.py", "robots/ur5e/robot.urdf")
gym_env = EAISysGymEnv(make_standard_pick_place_env(interp))
obs, info = gym_env.reset()
obs, reward, terminated, truncated, info = gym_env.step("MV_FWD")
```
