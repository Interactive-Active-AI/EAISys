<p align="right">
  <b>English</b> | <a href="./urdf-free-probing.zh-CN.md">简体中文</a>
</p>

# No-URDF Active Self-Probing

What if you plug in a brand-new robot that has **no URDF file at all**, no kinematic tree, and no 3D fingertip pose sensor?

Instead of giving up, EAISys lets the agent **play with its own body for a few seconds** to figure out how its joints move its hand—then solves real 3D manipulation tasks using the exact same universal skill commands.

## How It Works in 4 Steps (`src/eaisys/probing/self_probe.py`)

1. **Wiggle Fingers Once to Spot the Hand**
   - Keeping all arm joints completely still, the robot closes and opens its gripper once (`GRASP` → `RELEASE`).
   - Comparing pixel differences $|I_{\text{open}} - I_{\text{closed}}|$ across orthogonal views (`frontview` + `sideview`) immediately pinpoints where the fingertips sit on screen—zero simulator pose shortcuts.
2. **Gently Nudge Each Joint**
   - For each joint $j \in \{0, \dots, D-1\}$, the robot nudges that joint slightly, tracks where the fingertips shifted in pixels, and builds a visual-motor sensitivity matrix:
     $$\hat{J}_{\text{vis}}[:, j] = \frac{[\Delta u_{\text{side}},\; \Delta u_{\text{front}},\; -\Delta v_{\text{front}}]^T}{\Delta q_j}$$
3. **Quick Sanity Check & LLM Diagnosis**
   - The system checks that moving along `+X`, `+Y`, and `+Z` actually shifts the hand in the expected camera directions, while an LLM reviews the probing montage (`self_probe_montage.jpg`) and summarizes each joint's role.
4. **Adapt Live While Doing the Task**
   - During the real 3D task, the exact same universal commands (`MV_FWD`, `MV_DOWN`, `GRASP`, `MV_UP`, etc.) solve joint increments via damped least squares with gentle home-posture regularization.
   - After every move, a patch tracker measures where the fingertips actually landed and applies a guarded rank-1 Broyden update so the robot stays accurate as its posture changes:
     $$\hat{J}_{k+1} = \hat{J}_k + \alpha \frac{(\Delta p_{\text{vis}} - \hat{J}_k \Delta q)\,\Delta q^T}{\|\Delta q\|^2 + \lambda}$$

## Try It Yourself

```bash
uv run --extra sim eaisys play-urdf-free \
  --robosuite-robot Panda \
  --env Lift \
  --run-id my_urdf_free_panda_lift \
  --provider openai --model gpt-5.4
```
