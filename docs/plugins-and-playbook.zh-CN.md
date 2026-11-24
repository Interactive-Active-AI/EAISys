<p align="right">
  <a href="./plugins-and-playbook.md">English</a> | <b>简体中文</b>
</p>

# 可开关插件与操作习惯 Playbook

为了让控制提示词既干净又方便做消融实验，EAISys 把每一类上下文都做成了独立可开关的插件；并且所有拼装好的提示词在发给模型前，都必须经过严格的零作弊防火墙。

## 8 个可独立开关的插件（`HarnessPluginConfig`）

关闭任意插件后，对应段落会自动完全移除，零副作用：

| 插件 | CLI / 配置开关 | 作用（说人话） |
| --- | --- | --- |
| `MultiViewGuidancePlugin` | `multi_view_guidance` | 告诉模型每个相机画面里的上下左右分别对应机械臂的前后左右。 |
| `ProprioceptionPlugin` | `proprioception_feedback` | 提供机器人自己的硬件读数：当前步长档位、关节角、指尖位置与夹爪编码器。 |
| `AdaptiveStepPlugin` | `adaptive_step_hint` | 提示模型空旷区大步快走、接近目标切小步精调。 |
| `SubtaskPlanningPlugin` | `subtask_planning` | 把上一轮模型自己定下的子目标带到下一轮，防止做着做着忘事。 |
| `ActionHistoryPlugin` | `action_history_window` | 展示最近刚执行过的几条技能指令，方便发现原地打转或卡住。 |
| `PlaybookMemoryPlugin` | `--playbook v1` / `--no-playbook` | 从 `configs/playbooks/v1.md` 按任务自动抽取对应操作习惯（如擦除触地判断、双臂下探等）。 |
| `CrossEmbodimentPriorsPlugin` | `--no-priors` | 从 8 套开源数据集翻译来的 114 条跨机身轨迹里检索相似技能参考。 |
| `VisualDiffPlugin` | `--visual-diff` | 把上一轮画面（`t-1`）和当前画面（`t`）并排送给模型，一眼看清刚才动了多少。 |

## 持久化经验本与零作弊守门员

- **人类可读的操作习惯本（`configs/playbooks/v1.md`）**：按 6 大场景分类管理——`[universal]`、`[articulated]`、`[sorting]`、`[insertion]`、`[wiping]`、`[bimanual]`。
- **自动审核经验写入**：通过 `PlaybookStore.append_governed_lesson()` 新增经验时，如果内容里夹带了仿真器内部变量名、抓持作弊标志或硬编码的 `cm`/`mm` 数值，守门员会自动拒收。
- **运行时提示词防火墙**：每轮调用模型前，`assert_zero_privilege_prompt()` 都会全量扫描提示词，一旦发现任何越权信息立刻抛出 `ZeroPrivilegeViolationError`。
