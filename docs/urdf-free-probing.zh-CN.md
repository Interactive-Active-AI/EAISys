<p align="right">
  <a href="./urdf-free-probing.md">English</a> | <b>简体中文</b>
</p>

# 没有 URDF 时，它是怎么自己摸清身体的？

如果面前是一台**完全没有 URDF 说明书**、没有运动学树、也没有 3D 位姿传感器的陌生机械臂，怎么办？

EAISys 不会直接报错退出，而是让智能体**先自己活动几秒关节**，用原始相机看懂“动哪个关节会让手往哪边走”，然后再用同一套通用技能完成真实的 3D 操作任务。

## 自主试探 4 步流程（`src/eaisys/probing/self_probe.py`）

1. **静止抖一下手指，找到自己的手在哪**
   - 保持整条手臂静止，只把夹爪闭合再张开一次（`GRASP` → `RELEASE`）。
   - 用正交双视角（`frontview` + `sideview`）的纯像素差 $|I_{\text{open}} - I_{\text{closed}}|$ 直接锁定指尖像素位置——完全不需要仿真器告诉它手在哪。
2. **安全轻推每一个未知关节**
   - 对第 $j \in \{0, \dots, D-1\}$ 个关节分别施加极小幅度的安全试探，追踪指尖在画面里的像素偏移，拼出视觉运动灵敏度矩阵：
     $$\hat{J}_{\text{vis}}[:, j] = \frac{[\Delta u_{\text{side}},\; \Delta u_{\text{front}},\; -\Delta v_{\text{front}}]^T}{\Delta q_j}$$
3. **三轴自检与大模型形态诊断**
   - 自动检验向前/向左/向上试探方向是否与画面一致，同时让大模型查看试探拼图（`self_probe_montage.jpg`）总结各关节的分工。
4. **干活的同时用 Broyden 在线微调**
   - 正式执行 3D 任务时，依然使用同一套通用指令（`MV_FWD`、`MV_DOWN`、`GRASP`、`MV_UP` 等），通过阻尼最小二乘加姿态保持解出关节增量。
   - 每执行完一步，用局部图像模板追踪指尖真实落点，并用带门控的秩-1 Broyden 公式在线更新映射矩阵，确保手臂换了姿势依然走得准：
     $$\hat{J}_{k+1} = \hat{J}_k + \alpha \frac{(\Delta p_{\text{vis}} - \hat{J}_k \Delta q)\,\Delta q^T}{\|\Delta q\|^2 + \lambda}$$

## 一行命令亲自试跑

```bash
uv run --extra sim eaisys play-urdf-free \
  --robosuite-robot Panda \
  --env Lift \
  --run-id my_urdf_free_panda_lift \
  --provider openai --model gpt-5.4
```
