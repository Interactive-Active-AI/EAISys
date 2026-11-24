<p align="right">
  <a href="./urdf-contract.md">English</a> | <b>简体中文</b>
</p>

# URDF 说明书契约与 6D 安全验算

在 EAISys 中，接入一台机器人只需要两份文件：
1. 原始机器人描述文件：`robots/<robot_id>/robot.urdf`（支持标准 URDF 与 MuJoCo MJCF `.xml`）。
2. 大模型自动编写的 Python 控制器：`harnesses/embodiments/<robot_id>.py`。

## 每份机器人控制器必须实现哪 5 件事？

`harnesses/embodiments/<robot_id>.py` 统一导出 `build_interpreter(urdf_path)`，实现以下 5 个清晰方法：

| 方法 | 作用（大白话） |
| --- | --- |
| `reset()` | 把机械臂放到一个屈肘、远离奇异点、安全在限位内的初始待命姿势，并打开夹爪。 |
| `arm_joint_names(arm)` | 返回对应手臂（单臂 `"main"`，或双臂 `"left"` / `"right"`）按顺序排列的电机关节名。 |
| `tcp_link_name(arm)` | 标明 URDF 里哪个连杆是机器人指尖 / 工具末端。 |
| `fk_tcp(state, arm)` | 计算当前指尖在 3D 空间里的位姿（如有躯干或移动底盘也一并算进去）。 |
| `apply_unit(unit, state, arm)` | 把一条通用技能指令（如 `MV_FWD`、`ROT_YAW_CW`、`GRASP`、`BASE_FWD`）翻译成安全的目标关节角，同时死守关节限位与工作空间边界。 |

## 所有机器人共用的 25 个基础动作技能

每台机器人听懂的都是同一套 25 词通用动作语言（[`action_space.py`](../src/eaisys/harness/action_space.py)）：

- **空间 6 向平移（6）**：`MV_FWD`、`MV_BACK`、`MV_LEFT`、`MV_RIGHT`、`MV_UP`、`MV_DOWN`
- **手腕 3 轴旋转（6）**：`ROT_ROLL_CW/CCW`、`ROT_PITCH_CW/CCW`、`ROT_YAW_CW/CCW`（每次 `15°`）
- **3 档步长变速（3）**：`STEP_COARSE`（`4 cm`）、`STEP_FINE`（`2 cm`）、`STEP_ULTRA_FINE`（`1 cm`）
- **抓放与流程（4）**：`GRASP`、`RELEASE`、`HOLD`、`DONE`
- **移动底盘行走（6）**：`BASE_FWD`、`BASE_BACK`、`BASE_LEFT`、`BASE_RIGHT`、`BASE_YAW_LEFT`、`BASE_YAW_RIGHT`（轮式底盘自动开启）

## 两行命令接入你自己的新机器人

```bash
# 1. 把你的裸 URDF 放到 robots/my_arm/robot.urdf（不用手写任何配置文件）
export OPENAI_API_KEY="your-key"
uv run eaisys synthesize robots/my_arm/robot.urdf --provider openai --model gpt-5.4

# 2. 在隔离沙盒里独立跑完 AST 安全审计与 6D 物理验算
uv run eaisys verify --robot my_arm
```
