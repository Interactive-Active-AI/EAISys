<p align="right">
  <b>English</b> | <a href="./plugins-and-playbook.zh-CN.md">简体中文</a>
</p>

# Toggleable Plugins & Manipulation Playbook

EAISys keeps its online control prompt clean, modular, and easy to ablate: every piece of context is an independent plugin you can switch on or off with a single flag, and every prompt passes a strict zero-cheat firewall before reaching the model.

## The 8 Toggleable Plugins (`HarnessPluginConfig`)

Turn any plugin off and its section disappears cleanly with zero side effects:

| Plugin | CLI / Config Flag | What It Adds (Plain English) |
| --- | --- | --- |
| `MultiViewGuidancePlugin` | `multi_view_guidance` | Explains which image direction corresponds to forward/left/up in each camera view. |
| `ProprioceptionPlugin` | `proprioception_feedback` | Shares the robot's real hardware readings: current step size, joint angles, fingertip position in `base_link`, and jaw encoder values. |
| `AdaptiveStepPlugin` | `adaptive_step_hint` | Reminds the model to use big steps in open air and tiny steps near contact. |
| `SubtaskPlanningPlugin` | `subtask_planning` | Carries the model's own previous step goal into the next turn so it doesn't lose track. |
| `ActionHistoryPlugin` | `action_history_window` | Shows the last few executed commands so the model notices if it is repeating or stuck. |
| `PlaybookMemoryPlugin` | `--playbook v1` / `--no-playbook` | Adds relevant manipulation habits from `configs/playbooks/v1.md` (e.g., wiping contact checks or dual-arm grasping). |
| `CrossEmbodimentPriorsPlugin` | `--no-priors` | Retrieves matching universal skill sequences translated from 8 open robot datasets. |
| `VisualDiffPlugin` | `--visual-diff` | Attaches the previous frame (`t-1`) right beside the current frame (`t`) so motion is effortless to see. |

## Playbook Memory & Zero-Cheat Guard

- **Human-Readable Playbook (`configs/playbooks/v1.md`)**: Organized into 6 clear manipulation families—`[universal]`, `[articulated]`, `[sorting]`, `[insertion]`, `[wiping]`, and `[bimanual]`.
- **Automatic Memory Governor**: Adding a new lesson via `PlaybookStore.append_governed_lesson()` automatically rejects any rule that sneaks in simulator variable names, holding cheats, or hardcoded `cm`/`mm` hints.
- **Runtime Prompt Firewall**: Before any VLM call goes out, `assert_zero_privilege_prompt()` scans the full prompt and raises `ZeroPrivilegeViolationError` immediately if any forbidden simulator state ever appears.
