<p align="right">
  <b>English</b> | <a href="./urdf-contract.zh-CN.md">简体中文</a>
</p>

# URDF Contract & 6D Verification

In EAISys, every supported robot has just two files:
1. Its raw robot description: `robots/<robot_id>/robot.urdf` (standard URDF or MuJoCo MJCF `.xml`).
2. Its AI-written Python controller: `harnesses/embodiments/<robot_id>.py`.

## What Every Robot Controller Must Do

Each module under `harnesses/embodiments/<robot_id>.py` exports `build_interpreter(urdf_path)` implementing 5 clean methods:

| Method | Plain-English Job |
| --- | --- |
| `reset()` | Put the robot into a safe, bent-elbow ready posture well inside joint limits with grippers open. |
| `arm_joint_names(arm)` | Return the ordered list of motor joints for `"main"` (or `"left"` / `"right"` on dual-arm robots). |
| `tcp_link_name(arm)` | Tell EAISys which link in the URDF is the fingertip / tool tip. |
| `fk_tcp(state, arm)` | Compute where the fingertip currently sits in 3D space (including torso or mobile base if present). |
| `apply_unit(unit, state, arm)` | Turn one universal skill command (`MV_FWD`, `ROT_YAW_CW`, `GRASP`, `BASE_FWD`, etc.) into safe joint targets without exceeding joint limits or the safety box. |

## The 25 Universal Skill Commands

Every robot speaks the exact same 25-word skill vocabulary ([`action_space.py`](../src/eaisys/harness/action_space.py)):

- **Move in 3D (6)**: `MV_FWD`, `MV_BACK`, `MV_LEFT`, `MV_RIGHT`, `MV_UP`, `MV_DOWN`
- **Turn Wrist (6)**: `ROT_ROLL_CW/CCW`, `ROT_PITCH_CW/CCW`, `ROT_YAW_CW/CCW` (`15°` per step)
- **Precision Gears (3)**: `STEP_COARSE` (`4 cm`), `STEP_FINE` (`2 cm`), `STEP_ULTRA_FINE` (`1 cm`)
- **Grip & Flow (4)**: `GRASP`, `RELEASE`, `HOLD`, `DONE`
- **Mobile Base (6)**: `BASE_FWD`, `BASE_BACK`, `BASE_LEFT`, `BASE_RIGHT`, `BASE_YAW_LEFT`, `BASE_YAW_RIGHT` (enabled automatically for wheeled robots)

## Adding Your Own Robot in 2 Commands

```bash
# 1. Drop your raw URDF at robots/my_arm/robot.urdf (no extra YAML needed)
export OPENAI_API_KEY="your-key"
uv run eaisys synthesize robots/my_arm/robot.urdf --provider openai --model gpt-5.4

# 2. Independently verify safety (AST) and 6D motion accuracy in an isolated subprocess
uv run eaisys verify --robot my_arm
```
