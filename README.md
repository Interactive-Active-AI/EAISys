<p align="right">
  <b>English</b> | <a href="./README.zh-CN.md">简体中文</a>
</p>

<div align="center">

<h1>EAISys</h1>
<h3>One Brain for Any Robot Body.</h3>
<p><i>Give it a robot's URDF — it works right away. Give it nothing — it plays with its own joints for a few seconds and figures it out.</i></p>

<p>
  <a href="https://interactive-active-ai.github.io/EAISys/"><img alt="Interactive Demo" src="https://img.shields.io/badge/Live_Showcase-open-3E6280?style=flat&labelColor=F0F4F8" height="20"></a>&nbsp;
  <a href="docs/architecture.md"><img alt="Docs" src="https://img.shields.io/badge/Docs-Read-6A5484?style=flat&labelColor=F4F1F8" height="20"></a>&nbsp;
  <img alt="No Simulator Cheats" src="https://img.shields.io/badge/Control-Zero_Cheats_(RGB_+_Encoders)-98505E?style=flat&labelColor=FBF2F3" height="20">&nbsp;
  <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-526B85?style=flat&labelColor=F0F4F8" height="20">&nbsp;
  <a href="LICENSE"><img alt="Apache-2.0" src="https://img.shields.io/badge/License-Apache_2.0-7D6854?style=flat&labelColor=F7F3EE" height="20"></a>
</p>

</div>

---

## Why We Built This

Most robot AI systems break the moment you swap the hardware:
- **End-to-end VLAs** bake one specific robot's body into their neural weights—change the arm and you have to collect new demos and retrain.
- **Hand-coded robot agents** still ask engineers to manually write inverse kinematics, joint limits, and frame conversions for every new machine—ignoring that **URDF is already the most code-native robot manual ever invented**.

**EAISys** separates *how to do a task* from *what the robot body looks like*, and proves both ways a robot agent should adapt to a brand-new body:

1. **Have a URDF? Zero-shot manipulation right away.**
   - Hand EAISys any raw `.urdf` file (no `sidecar.yaml`, no human demos). An LLM reads the URDF, writes a Python controller for that robot, and tests it in an isolated sandbox (checking code safety, 6-axis motion accuracy `< 4 mm`, wrist rotation `< 5°`, joint limits, and workspace bounds).
   - Once verified, any vision-language model can immediately control single arms, tool-swapped arms, or even **two completely different unseen robots working together** (`Kinova3` + `UR5e`).
2. **No URDF at all? Self-learns its own body in seconds.**
   - Facing an unknown robot with **zero URDF**, no kinematic tree, and no 3D pose sensor, EAISys lets the agent explore itself first: it wiggles its fingers once to spot its own hand in the cameras, gently nudges each joint to learn how its hand responds, and adapts online while completing the 3D task.

---

## The Universal Skill Library

Instead of re-teaching every robot from scratch, every robot in EAISys shares one clean **4-part skill library**:

| Layer | Where It Lives | What It Gives Every Robot |
| --- | --- | --- |
| **25 Universal Motion Skills** | [`src/eaisys/harness/action_space.py`](src/eaisys/harness/action_space.py) | One simple vocabulary across all robots: move in 6 directions (`MV_*`), turn wrist in 6 directions (`ROT_*`), switch between 3 precision gears (`4 cm / 2 cm / 1 cm`), grip/flow (`GRASP`, `RELEASE`, `HOLD`, `DONE`), and roll a mobile base (`BASE_*`). |
| **Shared 244-Line Kinematic Engine** | [`harnesses/common/urdf_interpreter_base.py`](harnesses/common/urdf_interpreter_base.py) | After seeing the first few robots, the LLM distilled shared FK/IK, smooth motion, joint limits, and safety clamping into one reusable base class—now reused directly by **12 of 15** verified robots. |
| **Manipulation Habits & 114 Skill Traces** | [`configs/playbooks/v1.md`](configs/playbooks/v1.md) & [`artifacts/semantic_priors/`](artifacts/semantic_priors/) | Transferable zero-cheat manipulation habits (multi-view alignment, door handles, bin walls, ring-on-peg insertion, contact wiping, dual-arm lifting) plus 114 cross-robot skill traces retargeted from 8 public datasets. |
| **Blind Self-Discovery Skill** | [`src/eaisys/probing/self_probe.py`](src/eaisys/probing/self_probe.py) | Built-in "learn my own body without a URDF" skill: finger-wiggle hand localization, joint probing, visual-motor Jacobian estimation, and live Broyden adaptation. |

---

## Real 3D Results (12 / 12 Solved — Zero Simulator Cheats)

Every 3D MuJoCo run below operates **strictly blind to simulator state**: the VLM sees **only** raw multi-camera RGB images and raw hardware joint/gripper encoders (no object coordinates, no table height, no metric rulers, no holding flags—enforced at runtime by `assert_zero_privilege_prompt()`). Full videos, turn-by-turn thoughts, prompts, and raw URDF/code traces are archived in [`artifacts/live_runs/`](artifacts/live_runs/) and browsable in the [Interactive Observatory](website/index.html).

<!-- LIVE_RESULTS_TABLE_START -->
### With URDF $\rightarrow$ Zero-Shot Manipulation (11 / 11)

| Task | Unseen Robot (Raw URDF) | What Makes It Hard | Model | Turns | Final Physics Result | Status |
| --- | --- | --- | --- | ---: | --- | :---: |
| **`TwoArmLift`** | `Kinova3` (7-DoF) + `UR5e` (6-DoF) | Two different unseen arms lifting one heavy pot together | `gpt-6.2-sol` (`max`, fast) | 15 | Pot lifted `+10.96 cm` | PASS |
| **`NutAssemblyRound`** | `UR5e` (`ur5e_mujoco`) | Precision ring-on-peg insertion | `gpt-6.2-sol` (`max`, fast) | 24 | Concentric error `4.5 mm` | PASS |
| **`Wipe`** | `UR5e + WipingGripper` (`ur5e_wipe_mujoco`) | Gripper swapped for a fixed sponge; continuous table contact | `gpt-6.2-sol` (`max`, fast) | 18 | `40 / 40` marks cleaned (`100%`) | PASS |
| **`Door`** | `UR5e` (`ur5e_mujoco`) | Wrist rotation + pulling along a hinged arc | `gpt-6.1-sol` (`max`) | 17 | Door opened `22.94°` | PASS |
| **`Door`** | `Kinova3` (`kinova3_mujoco`) | Wrist rotation + pulling along a hinged arc | `gpt-6.1-sol` (`max`) | 18 | Door opened `17.48°` | PASS |
| **`PickPlaceMilk`** | `KUKA IIWA` (`iiwa_mujoco`) | Carrying a carton over wooden bin walls | `gpt-6.1-sol` (`max`) | 12 | Slot error `35.6 mm` | PASS |
| **`PickPlaceMilk`** | `Kinova3` (`kinova3_mujoco`) | Carrying a carton over wooden bin walls | `gpt-6.1-sol` (`max`) | 17 | Slot error `40.7 mm` | PASS |
| **`PickPlaceMilk`** | `UR5e` (`ur5e_mujoco`) | Carrying a carton over wooden bin walls | `gpt-6.1-sol` (`max`) | 20 | Slot error `66.5 mm` | PASS |
| **`Stack`** | `UR5e` (`ur5e_mujoco`) | 3D red-on-green block stacking | `gpt-6.1-sol` (`max`) | 11 | Error `4.4 mm`, $\Delta z = 44.8\text{ mm}$ | PASS |
| **`Stack`** | `Kinova3` (`kinova3_mujoco`) | 3D red-on-green block stacking | `gpt-6.1-sol` (`max`) | 16 | Error `8.6 mm`, $\Delta z = 44.8\text{ mm}$ | PASS |
| **`Stack`** | `KUKA IIWA` (`iiwa_mujoco`) | 3D red-on-green block stacking | `gpt-6.1-sol` (`max`) | 20 | Error `8.0 mm`, $\Delta z = 44.8\text{ mm}$ | PASS |

### Without URDF $\rightarrow$ Self-Probe Few-Shot $\rightarrow$ Manipulation (1 / 1)

| Task | Unknown Robot (Zero URDF) | How It Learned Its Body | Model | Turns | Final Physics Result | Status |
| --- | --- | --- | --- | ---: | --- | :---: |
| **`Lift` (No URDF)** | `Panda` (7-DoF, `urdf_provided = false`) | Finger-wiggle hand spot + 7 joint nudges + live Broyden updates | `gpt-6.2-sol` (`max`, fast) | 12 | Cube lifted to `z = 0.8632 m` | PASS |

*Offline Robot Library: **15 / 15** AI-generated URDF controllers (6 seen + 9 unseen raw-URDF robots, including 3-finger `Jaco`, `Sawyer`, `ALOHA`, `AgiBot G1`, `Unitree G1`, `UR5e`, `Kinova3`, `KUKA IIWA`, and `UR5e + WipingGripper`) pass isolated safety & 6D verification (`uv run eaisys verify`).*
<!-- LIVE_RESULTS_TABLE_END -->

---

## Quickstart

### 1. Install & Verify All 15 Robot Controllers

```bash
git clone https://github.com/Interactive-Active-AI/EAISys.git
cd EAISys
uv sync --extra dev

# Run AST security audit + isolated 6D motion checks on all 15 robots
uv run eaisys verify

# List all registered robots
uv run eaisys list-harnesses
```

### 2. Add Any Brand-New Robot from Its URDF

EAISys works with any multimodal backend: `openai` (default), `openrouter`, local `vllm`, `anthropic`, or optional `codex`.

```bash
export OPENAI_API_KEY="your-api-key"

uv run eaisys synthesize robots/ur5e/robot.urdf \
  --provider openai --model gpt-5.4 --effort high

# Or point to a local open-weights server (vLLM / SGLang):
uv run eaisys synthesize robots/ur5e/robot.urdf \
  --provider vllm --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen2.5-VL-72B-Instruct
```

### 3. Run Real 3D Tasks (With or Without URDF)

```bash
uv sync --extra sim --extra llm

# Single arm from raw URDF (Stack, Door, PickPlaceMilk, PickPlaceCan, NutAssemblyRound, Wipe)
uv run --extra sim eaisys play-mujoco \
  --env NutAssemblyRound --robosuite-robot UR5e --robot-id ur5e_mujoco \
  --run-id my_ur5e_nut --provider openai --model gpt-5.4

# Two different unseen robots working together (Kinova3 + UR5e)
uv run --extra sim eaisys play-bimanual-mujoco \
  --env TwoArmLift --left-robot Kinova3 --right-robot UR5e \
  --run-id my_bimanual_lift --provider openai --model gpt-5.4

# Zero URDF: self-probe joints for a few seconds, then solve the 3D task
uv run --extra sim eaisys play-urdf-free \
  --robosuite-robot Panda --env Lift \
  --run-id my_urdf_free_lift --provider openai --model gpt-5.4
```

### 4. Browser Teleop, Fine-Tuning Export & Interactive Website

```bash
# Drive any URDF robot right in your browser (no real hardware required)
uv run --extra sim eaisys gumi --robot ur5e_mujoco --env Stack --robosuite-robot UR5e --port 8766

# Export verified 3D episodes into LLaMA-Factory SFT or GRPO training files
uv run eaisys export-sft --format llamafactory --out artifacts/posttrain/sft.jsonl
uv run eaisys export-sft --format grpo --out artifacts/posttrain/grpo.jsonl

# Preview the interactive bilingual showcase locally
uv run eaisys serve-site --host 127.0.0.1 --port 8765
```

---

## How Does This Compare?

| Approach | How It Controls Hardware | Adding a Brand-New Robot | Two Different Arms Together |
| --- | --- | --- | --- |
| **End-to-End VLAs** (π0.5, OpenVLA, GR00T) | Body geometry locked inside neural weights | Collect new teleop demos and fine-tune | Fixed to the training robot |
| **Hand-Coded Harnesses** (Show-Harness) | Engineers hand-write a backend per robot | Manual IK, joint limits, and frame setup | Hand-coded dual-arm sessions |
| **Single-Body Agents** (RobotUse, CaP-X, Skill2Real) | Pre-built single-robot backend | No raw-URDF controller synthesis | Mostly single-arm setups |
| **EAISys v0.3 (Ours)** | **Reads raw `.urdf` + shared 244-line skill engine** | **Zero-shot with URDF · few-shot self-probe without URDF** | **Native heterogeneous bimanual (`Kinova3` + `UR5e`)** |

---

## Documentation & Repo Layout

- [How It Works (Architecture)](docs/architecture.md)
- [URDF Contract & 6D Verification](docs/urdf-contract.md)
- [No-URDF Active Self-Probing](docs/urdf-free-probing.md)
- [Toggleable Plugins & Playbook Memory](docs/plugins-and-playbook.md)
- [Browser Teleop (GUMI), Gym Wrapper & SFT/GRPO Export](docs/post-training.md)
- [Contributing Guide](CONTRIBUTING.md)

```text
EAISys/
├── harnesses/
│   ├── common/urdf_interpreter_base.py   # Shared 244-line kinematic engine (reused by 12 robots)
│   └── embodiments/*.py                  # 15 verified AI-written robot controllers
├── robots/<robot_id>/robot.urdf          # Raw URDF manuals (seen + unseen robots)
├── configs/playbooks/v1.md               # Reusable zero-cheat manipulation habits
├── src/eaisys/
│   ├── urdf/                             # URDF/MJCF parser, FK, Jacobian & damped IK
│   ├── backends/                         # OpenAI, OpenRouter, vLLM, Anthropic & Codex CLI
│   ├── harness/                          # 25 skills, AST+6D sandbox verifier, plugins, MuJoCo & Gym
│   ├── probing/                          # No-URDF self-probing (finger wiggle + Jacobian + Broyden)
│   ├── memory/                           # Playbook loader & zero-cheat Memory Governor
│   ├── synthesis/                        # LLM URDF controller writer & dataset retargeting
│   ├── posttrain/                        # LLaMA-Factory / OpenAI / GRPO exporter
│   ├── gumi.py                           # Browser teleop server
│   └── site.py                           # Interactive showcase builder & local server
├── artifacts/                            # Sanitized 3D traces & 114 cross-robot skill priors
└── website/                              # Zero-dependency bilingual interactive showcase
```

---

## License & Citation

Released under the [Apache License 2.0](LICENSE). Citation metadata is provided in [`CITATION.cff`](CITATION.cff).
