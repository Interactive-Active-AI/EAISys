<p align="right">
  <a href="./README.md">English</a> | <b>简体中文</b>
</p>

<div align="center">

<h1>EAISys</h1>
<h3>一套大脑，开箱控制任何机器人。</h3>
<p><i>给一份 URDF 说明书，新机械臂直接零样本干活；什么都不给，让它自己动几秒摸清身体，照样完成 3D 任务。</i></p>

<p>
  <a href="https://interactive-active-ai.github.io/EAISys/"><img alt="交互式主页" src="https://img.shields.io/badge/Live_Showcase-open-3E6280?style=flat&labelColor=F0F4F8" height="20"></a>&nbsp;
  <a href="docs/architecture.md"><img alt="文档" src="https://img.shields.io/badge/Docs-Read-6A5484?style=flat&labelColor=F4F1F8" height="20"></a>&nbsp;
  <img alt="零作弊控制" src="https://img.shields.io/badge/Control-Zero_Cheats_(RGB_+_Encoders)-98505E?style=flat&labelColor=FBF2F3" height="20">&nbsp;
  <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-526B85?style=flat&labelColor=F0F4F8" height="20">&nbsp;
  <a href="LICENSE"><img alt="Apache-2.0" src="https://img.shields.io/badge/License-Apache_2.0-7D6854?style=flat&labelColor=F7F3EE" height="20"></a>
</p>

</div>

---

## 为什么要做这个项目？

现有的机器人 AI 一换机身就容易“失灵”：
- **端到端 VLA** 把特定机器人的身体结构锁死在神经网络权重里——换一条机械臂，就得重新采数据、重新微调。
- **传统手写具身框架** 仍然需要工程师为每台新机器人手写逆运动学、限位和坐标系转换——完全忽略了 **URDF 本身就是最贴近代码的机器人说明书**。

**EAISys** 把“怎么做任务”和“身体长什么样”彻底解耦，并端到端跑通了跨身体泛化的**两种核心场景**：

1. **有 URDF？读完即用，新机器人零样本直接干活。**
   - 只要丢给 EAISys 一份裸 `.urdf` 文件（不用配 `sidecar.yaml`，不用给人工示教），大模型就会自己读懂说明书、写出专属 Python 控制器，并在隔离沙盒里跑完安全审查与 6D 物理验算（6 向平移误差 `< 4 mm`、手腕旋转 `< 5°`、关节限位与安全围栏全过）。
   - 验过之后，任何多模态大模型立刻就能直接操控单臂、换工具手臂，乃至**两台完全不同的未见机器人异构双臂协同**（`Kinova3` + `UR5e`）。
2. **没 URDF？自己玩几秒关节，照样能完成任务。**
   - 面对一台**完全没有 URDF**、没有运动学树、没有 3D 位姿传感器的陌生机械臂，Agent 会先自己摸索几秒：静止抖一下手指从相机里找到自己的手，再轻推每个关节估出“怎么动关节手会往哪走”，并在真实执行 3D 任务时边做边校准。

---

## 通用技能库：所有机器人共用的动作与经验

不用每换一台机器人就从零教起，EAISys 为所有机器人沉淀了一套 **4 层通用技能库**：

| 技能层 | 代码位置 | 作用（说人话） |
| --- | --- | --- |
| **25 个基础动作技能** | [`src/eaisys/harness/action_space.py`](src/eaisys/harness/action_space.py) | 所有机器人通用的动作词汇：6 向平移（`MV_*`）、6 向转腕（`ROT_*`）、3 档变速（`4 cm / 2 cm / 1 cm`）、抓放控制（`GRASP` / `RELEASE` / `HOLD` / `DONE`）以及轮式底盘行走（`BASE_*`）。 |
| **244 行跨机身共享引擎** | [`harnesses/common/urdf_interpreter_base.py`](harnesses/common/urdf_interpreter_base.py) | 大模型看懂前几台机器人后，自动把共用的正逆运动学、平滑插值、限位保护抽成了一份 244 行通用基类，目前 **15 款机器人里已有 12 款直接复用**。 |
| **操作经验库 & 114 条跨机轨迹** | [`configs/playbooks/v1.md`](configs/playbooks/v1.md) & [`artifacts/semantic_priors/`](artifacts/semantic_priors/) | 6 类零作弊可迁移习惯（多视角核对、拉门把手、翻越分拣墙、精密套环、触地擦除、双臂同步），以及从 8 套开源数据集中统一翻译出的 114 条跨机身技能示教。 |
| **无 URDF 自主试探技能** | [`src/eaisys/probing/self_probe.py`](src/eaisys/probing/self_probe.py) | 内置的“没说明书也能摸清自己身体”技能：抖指找手 → 逐关节轻推 → 估出视觉运动映射 → 边抓边做在线 Broyden 修正。 |

---

## 真实 3D 物理实测（12 / 12 全通关 · 100% 零作弊）

以下全部 3D MuJoCo 实录均**严格禁止偷看仿真内部状态**：模型只能看到原始多视角画面和机器人自己的关节/夹爪编码器读数（零物体坐标、零桌面高度、零辅助标尺、零抓持作弊，由 `assert_zero_privilege_prompt()` 在运行时强制拦截）。所有完整视频、每一步思维链、原始 Prompt 与 URDF/代码记录都在 [`artifacts/live_runs/`](artifacts/live_runs/)，也可以在 [交互式观测台](website/index.html) 直接点开看。

<!-- LIVE_RESULTS_TABLE_START -->
### 有 URDF $\rightarrow$ 新本体零样本操作（11 / 11）

| 任务 | 未见过新机器人（裸 URDF） | 任务难点 | 模型 | 步数 | 最终物理结果 | 状态 |
| --- | --- | --- | --- | ---: | --- | :---: |
| **`TwoArmLift`** | `Kinova3` (7 轴) + `UR5e` (6 轴) | 两台完全不同的未见机械臂同步抬起 1.2kg 重锅 | `gpt-6.2-sol` (`max`, fast) | 15 | 锅底离桌 `+10.96 cm` | PASS |
| **`NutAssemblyRound`** | `UR5e` (`ur5e_mujoco`) | 抓取圆环并精准套进圆柱销 | `gpt-6.2-sol` (`max`, fast) | 24 | 同心误差 `4.5 mm` | PASS |
| **`Wipe`** | `UR5e + 海绵垫` (`ur5e_wipe_mujoco`) | 末端换装固定海绵垫，触地连续擦除表面污渍 | `gpt-6.2-sol` (`max`, fast) | 18 | `40 / 40` 污渍全擦净（`100%`） | PASS |
| **`Door`** | `UR5e` (`ur5e_mujoco`) | 旋转手腕勾住门把并沿铰链弧线拉开 | `gpt-6.1-sol` (`max`) | 17 | 开门角度 `22.94°` | PASS |
| **`Door`** | `Kinova3` (`kinova3_mujoco`) | 旋转手腕勾住门把并沿铰链弧线拉开 | `gpt-6.1-sol` (`max`) | 18 | 开门角度 `17.48°` | PASS |
| **`PickPlaceMilk`** | `KUKA IIWA` (`iiwa_mujoco`) | 抓起牛奶盒翻越木箱围挡放进目标格 | `gpt-6.1-sol` (`max`) | 12 | 落点误差 `35.6 mm` | PASS |
| **`PickPlaceMilk`** | `Kinova3` (`kinova3_mujoco`) | 抓起牛奶盒翻越木箱围挡放进目标格 | `gpt-6.1-sol` (`max`) | 17 | 落点误差 `40.7 mm` | PASS |
| **`PickPlaceMilk`** | `UR5e` (`ur5e_mujoco`) | 抓起牛奶盒翻越木箱围挡放进目标格 | `gpt-6.1-sol` (`max`) | 20 | 落点误差 `66.5 mm` | PASS |
| **`Stack`** | `UR5e` (`ur5e_mujoco`) | 3D 红方块精准叠放到绿方块上方 | `gpt-6.1-sol` (`max`) | 11 | 偏差 `4.4 mm`, $\Delta z = 44.8\text{ mm}$ | PASS |
| **`Stack`** | `Kinova3` (`kinova3_mujoco`) | 3D 红方块精准叠放到绿方块上方 | `gpt-6.1-sol` (`max`) | 16 | 偏差 `8.6 mm`, $\Delta z = 44.8\text{ mm}$ | PASS |
| **`Stack`** | `KUKA IIWA` (`iiwa_mujoco`) | 3D 红方块精准叠放到绿方块上方 | `gpt-6.1-sol` (`max`) | 20 | 偏差 `8.0 mm`, $\Delta z = 44.8\text{ mm}$ | PASS |

### 无 URDF $\rightarrow$ 少样本自主试探 $\rightarrow$ 完成操作（1 / 1）

| 任务 | 未知机器人（零 URDF） | 如何摸清自己身体 | 模型 | 步数 | 最终物理结果 | 状态 |
| --- | --- | --- | --- | ---: | --- | :---: |
| **`Lift`（无 URDF）** | `Panda`（7 轴，`urdf_provided = false`） | 抖指找到手在哪 + 7 关节试探 + 在线 Broyden 校准 | `gpt-6.2-sol` (`max`, fast) | 12 | 红块稳举至 `z = 0.8632 m` | PASS |

*离线机器人控制器库：共 **15 / 15** 款由 AI 编写的 URDF 控制器（6 款已见 + 9 款零 sidecar 未见，含三指手 `Jaco`、`Sawyer`、`ALOHA`、`AgiBot G1`、`Unitree G1`、`UR5e`、`Kinova3`、`KUKA IIWA`、`UR5e + 海绵垫`）全部通过子进程安全与 6D 验算（`uv run eaisys verify`）。*
<!-- LIVE_RESULTS_TABLE_END -->

---

## 快速上手

### 1. 安装并验算全部 15 款机器人控制器

```bash
git clone https://github.com/Interactive-Active-AI/EAISys.git
cd EAISys
uv sync --extra dev

# 对全部 15 款 AI 编写的控制器做代码安全审查 + 隔离 6D 运动验算
uv run eaisys verify

# 查看已支持的机器人列表
uv run eaisys list-harnesses
```

### 2. 接入任意一台新机器人（只需一份 URDF）

不挑模型，直接支持 `openai`（默认）、`openrouter`、本地 `vllm`、`anthropic` 及可选 `codex` CLI：

```bash
export OPENAI_API_KEY="your-api-key"

uv run eaisys synthesize robots/ur5e/robot.urdf \
  --provider openai --model gpt-5.4 --effort high

# 或者接入你自己的本地开源模型（vLLM / SGLang）：
uv run eaisys synthesize robots/ur5e/robot.urdf \
  --provider vllm --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen2.5-VL-72B-Instruct
```

### 3. 跑真实 3D 任务（有 URDF / 无 URDF 都支持）

```bash
uv sync --extra sim --extra llm

# 单臂裸 URDF（支持 Stack / Door / PickPlaceMilk / PickPlaceCan / NutAssemblyRound / Wipe）
uv run --extra sim eaisys play-mujoco \
  --env NutAssemblyRound --robosuite-robot UR5e --robot-id ur5e_mujoco \
  --run-id my_ur5e_nut --provider openai --model gpt-5.4

# 两台不同未见机器人协同抬锅（Kinova3 + UR5e）
uv run --extra sim eaisys play-bimanual-mujoco \
  --env TwoArmLift --left-robot Kinova3 --right-robot UR5e \
  --run-id my_bimanual_lift --provider openai --model gpt-5.4

# 无 URDF：先让它自己试探关节几秒，再闭环完成 3D 抓取
uv run --extra sim eaisys play-urdf-free \
  --robosuite-robot Panda --env Lift \
  --run-id my_urdf_free_lift --provider openai --model gpt-5.4
```

### 4. 网页遥操、导出微调数据 & 本地预览网站

```bash
# 免真机：打开浏览器直接遥控任意 URDF 机器人
uv run --extra sim eaisys gumi --robot ur5e_mujoco --env Stack --robosuite-robot UR5e --port 8766

# 把 3D 零作弊轨迹一键导出为 LLaMA-Factory SFT 或 GRPO 训练数据
uv run eaisys export-sft --format llamafactory --out artifacts/posttrain/sft.jsonl
uv run eaisys export-sft --format grpo --out artifacts/posttrain/grpo.jsonl

# 本地启动中英双语交互式主页
uv run eaisys serve-site --host 127.0.0.1 --port 8765
```

---

## 和其他开源方案有什么区别？

| 路线 | 怎么控制机器人身体 | 换一台全新机器人怎么办 | 两台不同机械臂协同 |
| --- | --- | --- | --- |
| **端到端 VLA**（π0.5 / OpenVLA / GR00T） | 机器人形态锁死在权重里 | 必须重采示教再微调 | 限于训练时的固定本体 |
| **手写具身框架**（Show-Harness） | 工程师逐台手写后端控制器 | 人工排查 IK、限位与坐标系 | 需手写同构双臂适配 |
| **单身体智能体**（RobotUse / CaP-X / Skill2Real） | 预设固定单臂后端 | 不支持读裸 URDF 自动生成控制器 | 以单臂工作站为主 |
| **EAISys v0.3（本项目）** | **读裸 `.urdf` 自动写控制器 + 244 行通用引擎** | **有 URDF 零样本即用；没 URDF 自己试探几秒即用** | **原生支持异构双臂（`Kinova3` + `UR5e`）** |

---

## 延伸阅读

- [系统是怎么运转的（整体架构）](docs/architecture.zh-CN.md)
- [URDF 说明书契约与 6D 安全验算](docs/urdf-contract.zh-CN.md)
- [没有 URDF 时，它是怎么自己摸清身体的？](docs/urdf-free-probing.zh-CN.md)
- [可开关插件与经验 Playbook](docs/plugins-and-playbook.zh-CN.md)
- [网页遥操（GUMI）、Gym 接口与微调数据导出](docs/post-training.zh-CN.md)
- [参与贡献指南](CONTRIBUTING.zh-CN.md)

---

## 协议与引用

本项目采用 [Apache License 2.0](LICENSE) 开源协议，引用格式见 [`CITATION.cff`](CITATION.cff)。
