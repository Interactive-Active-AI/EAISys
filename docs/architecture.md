<p align="right">
  <b>English</b> | <a href="./architecture.zh-CN.md">简体中文</a>
</p>

# How EAISys Works (Architecture)

Most robot systems either hard-code one robot's math by hand or lock a single robot's body into end-to-end neural weights. **EAISys** splits the problem cleanly into two loops: **learn the body once**, then **control it with universal skills**.

```text
               Loop A · Onboarding a New Robot (Once per Robot)
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Raw .urdf manual ──▶ Frontier LLM ──▶ Python controller (g_E)     │
  │                             ▲                     │                 │
  │                      auto self-fix        Code safety audit (AST)   │
  │                        feedback         + Isolated 6D physics check │
  │                             └─────────────────────┘                 │
  └─────────────────────────────────────────────────────────────────────┘
                                    │
                       Verified Controller (g_E)
                                    ▼
               Loop B · Solving 3D Tasks (Eyes & Joints Only)
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Raw Camera Images + Joint Encoders ──▶ Plugins + Habit Playbook    │
  │                                                  │                  │
  │                                          Any Vision-Language Model  │
  │                                                  │                  │
  │                                      Universal Skill Commands (A)   │
  │                                                  │                  │
  │                    Controller g_E (FK/IK + Safety Box) ──▶ Robot    │
  └─────────────────────────────────────────────────────────────────────┘
```

## Five Simple Ideas Behind the System

1. **URDF Is Already a Code-Native Robot Manual**
   - Every robot comes with a `.urdf` (or MuJoCo `.xml`) file describing its links, joints, axes, and limits.
   - Instead of engineers writing boilerplate controllers by hand for each new arm, an LLM reads the raw URDF, writes the Python controller, and reuses the shared 244-line kinematic engine ([`urdf_interpreter_base.py`](../harnesses/common/urdf_interpreter_base.py)).
2. **Never Trust Unchecked LLM Code (Isolated 6D Sandbox)**
   - Before a generated controller ever touches a robot, `audit_interpreter_ast()` checks its syntax tree so it cannot import filesystem, network, or shell modules.
   - `verify_interpreter_isolated()` then runs it in a clean child process to verify forward kinematics (`< 2 mm`), all 6 move directions (`< 4 mm`), all 6 wrist rotations (`< 5°`), joint limits, mobile-base motion, and workspace safety bounds.
3. **Zero Simulator Cheats (`assert_zero_privilege_prompt`)**
   - During live control, the AI only sees raw camera frames and the robot's own hardware encoders (joint angles, gripper jaw reading, and its own fingertip position).
   - No object coordinates, no table height, no drawn rulers, and no "holding object" flags—checked automatically on every single turn.
4. **Two Different Unseen Robots Working Together**
   - Because each robot's body is translated independently by its own URDF controller, EAISys can pair two completely different robots—such as a 7-DoF `Kinova3` and a 6-DoF `UR5e`—under one synchronized dual-arm loop (`left_chunk` + `right_chunk`).
5. **No URDF? Explore Yourself First**
   - If no URDF exists at all, Loop A is replaced by [`self_probe.py`](../src/eaisys/probing/self_probe.py): the robot wiggles its fingers to spot its hand, nudges each joint safely to estimate how its hand moves, and runs the exact same Loop B universal skills. See [No-URDF Active Self-Probing](urdf-free-probing.md).
