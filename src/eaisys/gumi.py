"""GUMI-URDF: Zero-dependency browser manipulation interface for human teleop & demonstration collection.

Inspired by Show-Harness's GUMI, generalized to ANY LLM-synthesized URDF interpreter.
Plays a robot in 3D MuJoCo (or offline tabletop) from the browser using the exact 25-unit
semantic action space and exports training-ready zero-privilege episode traces.
"""

from __future__ import annotations

import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
from pathlib import Path
import time
from typing import Any
from PIL import Image

from eaisys.harness.interpreter import load_interpreter_module
from eaisys.harness.sim_env import make_standard_pick_place_env
from eaisys.urdf import default_harnesses_root, default_robots_root


_GUMI_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>EAISys · GUMI-URDF Browser Teleop</title>
<style>
  body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:#FBF9F7; color:#353D43; }
  header { padding:14px 24px; background:#FFFFFF; border-bottom:1px solid #E6E2E3; display:flex; justify-content:space-between; align-items:center; }
  main { max-width:1200px; margin:20px auto; padding:0 20px; display:grid; grid-template-columns:1.6fr 1fr; gap:20px; }
  .card { background:#FFFFFF; border:1px solid #E6E2E3; border-radius:14px; padding:18px; }
  #frame { width:100%; border-radius:10px; border:1px solid #DFE6EE; background:#F0F4F8; display:block; }
  .grid { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin-top:10px; }
  button { padding:10px 8px; border-radius:9px; border:1px solid #DFE6EE; background:#F0F4F8; color:#3E6280; font-weight:600; cursor:pointer; }
  button:hover { background:#E2ECF5; }
  button.rose { background:#FBF2F3; color:#965C66; border-color:#E7C7CD; }
  button.purple { background:#F4F1F8; color:#6A5484; border-color:#D8CFE6; }
  pre { background:#F7F8FA; border:1px solid #E6E2E3; border-radius:8px; padding:10px; font-size:12px; overflow:auto; max-height:180px; }
</style>
</head>
<body>
<header>
  <div><strong>EAISys · GUMI-URDF</strong> <span style="color:#616B78;font-size:13px">Hardware-Free Browser Manipulation Interface</span></div>
  <div id="status" style="font-size:13px;color:#3E6280;font-weight:600">Connecting...</div>
</header>
<main>
  <div class="card">
    <img id="frame" alt="Live camera observation" />
    <div style="margin-top:10px;font-size:12.5px;color:#616B78">
      Keys: <b>W/S</b> (FWD/BACK) · <b>A/D</b> (LEFT/RIGHT) · <b>Q/E</b> (UP/DOWN) · <b>G/R</b> (GRASP/RELEASE) · <b>1/2/3</b> (Coarse/Fine/Ultra) · <b>X</b> (Reset)
    </div>
  </div>
  <div class="card">
    <h3 style="margin:0 0 8px">Semantic Action Pad</h3>
    <div class="grid">
      <button onclick="act('MV_UP')">MV_UP (Q)</button>
      <button onclick="act('MV_FWD')">MV_FWD (W)</button>
      <button onclick="act('MV_DOWN')">MV_DOWN (E)</button>
      <button onclick="act('MV_LEFT')">MV_LEFT (A)</button>
      <button onclick="act('HOLD')">HOLD</button>
      <button onclick="act('MV_RIGHT')">MV_RIGHT (D)</button>
      <button class="purple" onclick="act('STEP_COARSE')">COARSE (1)</button>
      <button onclick="act('MV_BACK')">MV_BACK (S)</button>
      <button class="purple" onclick="act('STEP_FINE')">FINE (2)</button>
    </div>
    <div class="grid" style="margin-top:8px">
      <button class="rose" onclick="act('GRASP')">GRASP (G)</button>
      <button class="rose" onclick="act('RELEASE')">RELEASE (R)</button>
      <button class="purple" onclick="act('STEP_ULTRA_FINE')">ULTRA (3)</button>
      <button onclick="act('ROT_YAW_CCW')">YAW CCW</button>
      <button onclick="act('ROT_YAW_CW')">YAW CW</button>
      <button onclick="act('DONE')">DONE</button>
    </div>
    <div style="display:flex;gap:8px;margin-top:12px">
      <button style="flex:1" onclick="post('/api/reset')">Reset Env (X)</button>
      <button class="purple" style="flex:1" onclick="post('/api/save')">Save Demo JSON</button>
    </div>
    <h4 style="margin:14px 0 6px;font-size:13px">Proprioception &amp; Recorded Units</h4>
    <pre id="telemetry"></pre>
  </div>
</main>
<script>
async function refresh(data) {
  if (!data) { data = await (await fetch('/api/state')).json(); }
  document.getElementById('frame').src = 'data:image/jpeg;base64,' + data.image_b64;
  document.getElementById('status').textContent = data.robot_id + ' · ' + data.env_name + ' · steps=' + data.step_count + (data.saved ? ' · SAVED ✓' : '');
  document.getElementById('telemetry').textContent = JSON.stringify({
    step_m: data.step_m, tcp_base_xyz: data.tcp_base_xyz, gripper_cmd: data.gripper_cmd,
    recent_units: data.recent_units.slice(-16), saved_path: data.saved_path || null
  }, null, 2);
}
async function act(unit) {
  const res = await fetch('/api/step', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({unit}) });
  refresh(await res.json());
}
async function post(url) {
  const res = await fetch(url, { method:'POST' });
  refresh(await res.json());
}
window.addEventListener('keydown', (e) => {
  const map = { w:'MV_FWD', s:'MV_BACK', a:'MV_LEFT', d:'MV_RIGHT', q:'MV_UP', e:'MV_DOWN', g:'GRASP', r:'RELEASE', '1':'STEP_COARSE', '2':'STEP_FINE', '3':'STEP_ULTRA_FINE' };
  const k = e.key.toLowerCase();
  if (map[k]) act(map[k]);
  else if (k === 'x') post('/api/reset');
});
refresh();
</script>
</body>
</html>
"""


def serve_gumi(
    *,
    robot_id: str = "ur5e_mujoco",
    robosuite_env: str | None = None,
    robosuite_robot: str | None = None,
    task_instruction: str = "Teleoperate the robot to complete the manipulation task.",
    host: str = "127.0.0.1",
    port: int = 8766,
    save_dir: str | Path = "artifacts/gumi_demos",
    seed: int = 11,
) -> None:
    """Launch the local GUMI-URDF browser teleoperation server."""
    rroot = default_robots_root()
    hroot = default_harnesses_root()
    urdf = rroot / robot_id / "robot.urdf"
    mod = hroot / "embodiments" / f"{robot_id}.py"
    interp = load_interpreter_module(mod, urdf, harnesses_root=hroot)

    is_mujoco = bool(robosuite_env and robosuite_robot)
    if is_mujoco:
        from eaisys.harness.mujoco_env import MuJoCoPhysicalHarnessEnv

        env: Any = MuJoCoPhysicalHarnessEnv(
            robosuite_env_name=str(robosuite_env),
            robosuite_robot=str(robosuite_robot),
            interpreter=interp,
            task_instruction=task_instruction,
            seed=seed,
        )
        env_label = f"MuJoCo:{robosuite_env}"
    else:
        env = make_standard_pick_place_env(interp, seed=seed)
        env.task_instruction = task_instruction
        env_label = "Tabletop2D"

    recorded_steps: list[dict[str, Any]] = []
    out_dir = Path(save_dir)

    def _state_payload(saved_path: str | None = None) -> dict[str, Any]:
        buf = io.BytesIO()
        if is_mujoco:
            Image.fromarray(env.compose_frame_rgb()).save(buf, format="JPEG", quality=85)
            tcp = [round(float(x), 4) for x in env.proprio_tcp_in_base()]
            g_cmd, _ = env.raw_gripper_proprio()
        else:
            import tempfile

            with tempfile.TemporaryDirectory() as td:
                p = env.render_multiview_png(Path(td) / "obs.png")
                Image.open(p).convert("RGB").save(buf, format="JPEG", quality=85)
            arm = interp.arm_names[0]
            tcp = [round(float(x), 4) for x in interp.fk_tcp(env.state, arm=arm).p]
            g_cmd = "OPEN" if env.state.gripper_open.get(arm, 1.0) > 0.5 else "CLOSED"
        return {
            "robot_id": interp.robot_id,
            "env_name": env_label,
            "step_count": env.step_count,
            "step_m": float(env.state.step_m),
            "tcp_base_xyz": tcp,
            "gripper_cmd": g_cmd,
            "recent_units": list(env.recent_units),
            "image_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
            "saved": saved_path is not None,
            "saved_path": saved_path,
        }

    class _GumiHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def _send_json(self, obj: dict[str, Any], code: int = 200) -> None:
            raw = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                body = _GUMI_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/state":
                self._send_json(_state_payload())
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            if self.path == "/api/step":
                data = json.loads(raw.decode("utf-8"))
                unit = str(data.get("unit", "HOLD"))
                if is_mujoco:
                    res = env.step_unit(unit)
                else:
                    res = env.step_unit(unit, arm=interp.arm_names[0]).to_dict()
                recorded_steps.append({"unit": unit, "result": res})
                self._send_json(_state_payload())
            elif self.path == "/api/reset":
                recorded_steps.clear()
                if is_mujoco:
                    env.reset()
                else:
                    env.state = interp.reset()
                    env.step_count = 0
                    env.recent_units.clear()
                self._send_json(_state_payload())
            elif self.path == "/api/save":
                out_dir.mkdir(parents=True, exist_ok=True)
                fp = out_dir / f"gumi_{robot_id}_{int(time.time())}.json"
                fp.write_text(
                    json.dumps(
                        {
                            "robot_id": interp.robot_id,
                            "environment": env_label,
                            "task_instruction": task_instruction,
                            "zero_privilege": True,
                            "semantic_units": list(env.recent_units),
                            "steps": recorded_steps,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                self._send_json(_state_payload(saved_path=str(fp)))
            else:
                self.send_error(404)

    server = HTTPServer((host, port), _GumiHandler)
    print(f"GUMI-URDF Teleop live at http://{host}:{port}/  (robot={robot_id}, env={env_label})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if is_mujoco:
            env.close()
