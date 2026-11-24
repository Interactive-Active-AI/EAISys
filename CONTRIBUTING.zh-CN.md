<p align="right">
  <a href="./CONTRIBUTING.md">English</a> | <b>简体中文</b>
</p>

# 参与贡献指南

欢迎为 **EAISys** 贡献新的机器人 URDF、仿真器适配、零作弊插件、操作习惯 Playbook 或微调配方！

## 几条共同约定（保持项目干净、真实）

1. **机身细节留在 URDF，不写进核心引擎**：新增机器人请走 `robots/<id>/robot.urdf` + `eaisys synthesize`，不要把某台特定机器人的运动学硬编码进 `src/eaisys/`。
2. **严守零作弊底线**：在线 VLM 只能拿到原始相机画面和硬件关节/夹爪编码器——绝不偷偷传入物体坐标、桌面高度、辅助尺或抓持状态。
3. **所有控制器必须过隔离验算**：`harnesses/embodiments/` 下的每份控制器都要通过 `uv run eaisys verify`（AST 安全审查 + 独立子进程 6D 平移、旋转、限位与围栏检查）。
4. **插件保持独立可开关**：在 `src/eaisys/harness/plugins.py` 新增插件时，请在 `HarnessPluginConfig` 提供对应的布尔开关。
5. **不提交隐私路径或密钥**：不要提交本机绝对路径、主机名、Token 或内部集群标识。提 PR 前跑一遍 `uv run pytest -v` 会自动帮你做完隐私和零作弊静态检查。

## 提 PR 前本地快速自检

```bash
uv sync --extra dev
uv run pytest -v
uv run eaisys verify
```

本地预览网站改动：

```bash
uv run eaisys serve-site --host 127.0.0.1 --port 8765
```
