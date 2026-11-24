"""End-to-end verification, security, zero-privilege, and privacy audit suite for EAISys v0.3."""

from __future__ import annotations

import json
from pathlib import Path
import re
import numpy as np
import pytest

from eaisys.backends.base import ModelResponse
from eaisys.datasets.schema import ArmTrajectoryStream, RawEpisode
from eaisys.harness import (
    ALL_SEMANTIC_UNITS,
    ArmProprioPacket,
    EAISysGymEnv,
    HarnessPluginConfig,
    PluginObservationContext,
    PluginPipeline,
    VLMHarnessRunner,
    ZeroPrivilegeViolationError,
    assert_zero_privilege_prompt,
    audit_interpreter_ast,
    load_interpreter_module,
    make_standard_pick_place_env,
    verify_interpreter,
    verify_interpreter_isolated,
)
from eaisys.memory import PlaybookStore
from eaisys.posttrain import build_urdf_kinematic_card, export_posttraining_dataset
from eaisys.sanitize import sanitize_text
from eaisys.synthesis import build_semantic_prior_bank
from eaisys.types import axis_angle_to_matrix, matrix_to_rotvec
from eaisys.urdf import (
    default_harnesses_root,
    default_robots_root,
    forward_kinematics_tree,
    parse_robot_file,
)


ROBOTS_DIR = default_robots_root()
HARNESSES_DIR = default_harnesses_root()
REPO_ROOT = Path(__file__).resolve().parents[1]


def _all_registered_robots() -> list[str]:
    return [
        sub.name
        for sub in sorted(ROBOTS_DIR.iterdir())
        if sub.is_dir()
        and (sub / "robot.urdf").is_file()
        and (HARNESSES_DIR / "embodiments" / f"{sub.name}.py").is_file()
    ]


@pytest.mark.parametrize("robot_id", _all_registered_robots())
def test_all_llm_synthesized_interpreters_pass_ast_and_6d_verification(robot_id: str) -> None:
    urdf = ROBOTS_DIR / robot_id / "robot.urdf"
    mod = HARNESSES_DIR / "embodiments" / f"{robot_id}.py"
    assert mod.is_file(), f"Missing LLM-synthesized interpreter: {mod}"
    assert audit_interpreter_ast(mod.read_text(encoding="utf-8")) == []
    interp = load_interpreter_module(mod, urdf, harnesses_root=HARNESSES_DIR, enforce_ast_audit=True)
    rep = verify_interpreter(interp, urdf, check_rotations=True, check_mobile=True)
    assert rep.passed, f"Verification failed on {robot_id}: {rep.failures}"


def test_isolated_subprocess_verifier_and_ast_security_gate(tmp_path: Path) -> None:
    # 1. Legitimate interpreter passes isolated child-process verification
    rep = verify_interpreter_isolated(
        HARNESSES_DIR / "embodiments" / "ur5e.py",
        ROBOTS_DIR / "ur5e" / "robot.urdf",
        harnesses_root=HARNESSES_DIR,
    )
    assert rep.passed is True and rep.isolated is True

    # 2. Malicious interpreter is blocked by static AST gate before execution
    evil_code = "import os\nimport subprocess\ndef build_interpreter(p):\n    os.system('echo pwn')\n"
    violations = audit_interpreter_ast(evil_code)
    assert len(violations) >= 2
    evil_py = tmp_path / "evil.py"
    evil_py.write_text(evil_code, encoding="utf-8")
    evil_rep = verify_interpreter_isolated(evil_py, ROBOTS_DIR / "ur5e" / "robot.urdf", harnesses_root=HARNESSES_DIR)
    assert evil_rep.passed is False

    # 3. Dunder frame escape is blocked
    escape_code = "def f(x):\n    return x.__func__.__globals__['__builtins__']\n"
    assert len(audit_interpreter_ast(escape_code)) >= 2


def test_common_library_is_reused_by_unseen_raw_urdf_interpreters() -> None:
    common_file = HARNESSES_DIR / "common" / "urdf_interpreter_base.py"
    assert common_file.is_file()
    assert audit_interpreter_ast(common_file.read_text(encoding="utf-8")) == []
    for unseen in ("ur5e", "kinova_gen3", "kuka_iiwa", "ur5e_mujoco", "kinova3_mujoco", "iiwa_mujoco"):
        code = (HARNESSES_DIR / "embodiments" / f"{unseen}.py").read_text(encoding="utf-8")
        assert "urdf_interpreter_base" in code, f"{unseen}.py did not reuse Cross-URDF Common Library"
        assert not (ROBOTS_DIR / unseen / "sidecar.yaml").exists()


def test_ablation_plugins_playbook_governor_and_zero_privilege_guard(tmp_path: Path) -> None:
    ctx = PluginObservationContext(
        embodiment_summary="unseen 3D robot `ur5e_mujoco`",
        task_instruction="Grasp the round ring nut and insert it over the peg.",
        turn_idx=1,
        max_turns=16,
        allowed_units=list(ALL_SEMANTIC_UNITS[:19]),
        camera_guides=[("CAM 1 (OVERHEAD)", "down = +X, right = +Y")],
        arms=[
            ArmProprioPacket(
                arm_label="main",
                robot_id="ur5e_mujoco",
                step_m=0.02,
                joint_q_rad=[0.0] * 6,
                tcp_base_xyz_m=[0.35, 0.0, 0.25],
                gripper_cmd="OPEN",
                gripper_encoder_qpos=[0.04, -0.04],
            )
        ],
    )
    p_on = PluginPipeline(cfg=HarnessPluginConfig(playbook_memory=True)).build_prompt(ctx)
    p_off = PluginPipeline(cfg=HarnessPluginConfig(playbook_memory=False)).build_prompt(ctx)
    assert "## Operational Playbook" in p_on
    assert "## Operational Playbook" not in p_off

    # Zero-privilege guard catches coordinate / cm leaks
    with pytest.raises(ZeroPrivilegeViolationError):
        assert_zero_privilege_prompt("Move cubeA_pos to goal")
    with pytest.raises(ZeroPrivilegeViolationError):
        assert_zero_privilege_prompt("Lower gripper by 3.5 cm to touch table")

    # Playbook Memory Governor rejects privileged lessons, accepts visual ones
    pb = PlaybookStore.load("v1")
    out_md = tmp_path / "v2.md"
    assert pb.append_governed_lesson("insertion", "Move 2.5 cm left of peg", out_md) is False
    assert pb.append_governed_lesson("insertion", "Check concentric bore alignment in overhead view before release.", out_md) is True
    assert "concentric bore alignment" in out_md.read_text(encoding="utf-8")


def test_strict_zero_privilege_and_sanitization_audit_across_all_live_runs() -> None:
    live_dir = REPO_ROOT / "artifacts" / "live_runs"
    web_dir = REPO_ROOT / "website"

    for jf in sorted(live_dir.glob("*.json")):
        raw = jf.read_text(encoding="utf-8")
        assert sanitize_text(raw) == raw, f"Unsanitized host/path token found in {jf.name}"

    mj_files = sorted(live_dir.glob("unseen_mujoco_*.json"))
    assert len(mj_files) >= 8, "Expected at least 8 verified 3D MuJoCo runs"

    for mf in mj_files:
        mdata = json.loads(mf.read_text(encoding="utf-8"))
        assert mdata.get("zero_privilege") is True, mf.name
        assert mdata.get("success") is True, mf.name
        assert (web_dir / mdata["video_path"]).is_file(), f"Missing MP4 for {mf.name}"
        assert (web_dir / mdata["video_path"]).stat().st_size > 80_000
        s1_turns = mdata["stage1_urdf_and_synthesis"]["synthesis_report"].get("turns", [])
        assert len(s1_turns) >= 1 and all(bool(st.get("prompt")) for st in s1_turns), f"Missing Stage-1 trace in {mf.name}"
        for t in mdata["stage2_vlm_closed_loop_turns"]:
            assert t.get("zero_privilege_verified") is True
            p = t.get("prompt", "")
            assert "STRICT ZERO-PRIVILEGE MODE" in p
            assert_zero_privilege_prompt(p)
            assert (web_dir / t["observation_image"]).is_file()


def test_repository_privacy_hygiene_audit() -> None:
    """Ensure zero personal usernames, cluster paths, or machine identifiers leak into public files."""
    forbidden_re = re.compile(
        r"(selensu|selen-suyue|yue\s+su|SELENSU-MC|m5air|m3pro|daxiao|ma-user|"
        r"apdcephfs|cephfs|modelarts|ChatGPT\.app|/data/home/|/Users/[A-Za-z0-9_.-]+|"
        r"aiwt_pick_and_place|libero-world-benchmark|162355200)",
        re.IGNORECASE,
    )
    scan_roots = [
        REPO_ROOT / "src",
        REPO_ROOT / "configs",
        REPO_ROOT / "docs",
        REPO_ROOT / "harnesses",
        REPO_ROOT / "robots",
        REPO_ROOT / "artifacts",
    ]
    scan_files = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "README.zh-CN.md",
        REPO_ROOT / "LICENSE",
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "CITATION.cff",
        REPO_ROOT / "CONTRIBUTING.md",
        REPO_ROOT / "website" / "index.html",
        REPO_ROOT / "website" / "styles.css",
        REPO_ROOT / "website" / "app.js",
    ]
    for root in scan_roots:
        if root.is_dir():
            for p in root.rglob("*"):
                if p.is_file() and p.suffix in (".py", ".yaml", ".yml", ".md", ".json", ".urdf", ".xml"):
                    scan_files.append(p)

    leaks: list[str] = []
    for f in scan_files:
        if not f.is_file():
            continue
        txt = f.read_text(encoding="utf-8", errors="ignore")
        hits = sorted(set(m.group(0) for m in forbidden_re.finditer(txt)))
        if hits:
            leaks.append(f"{f.relative_to(REPO_ROOT)}: {hits}")
    assert not leaks, "Privacy/machine leak detected:\n" + "\n".join(leaks)


def test_closed_loop_vlm_runner_gym_wrapper_and_sft_export(tmp_path: Path) -> None:
    ur5e = load_interpreter_module(
        HARNESSES_DIR / "embodiments" / "ur5e.py",
        ROBOTS_DIR / "ur5e" / "robot.urdf",
        harnesses_root=HARNESSES_DIR,
    )
    env = make_standard_pick_place_env(ur5e, seed=11)

    # 1. Gym wrapper sanity
    gym_env = EAISysGymEnv(make_standard_pick_place_env(ur5e, seed=11))
    obs, info = gym_env.reset()
    assert "tcp_base_xyz" in obs and info["zero_privilege"] is True
    obs2, reward, term, trunc, step_info = gym_env.step("MV_FWD")
    assert step_info["step"]["accepted"] is True and reward in (0.0, 1.0)

    # 2. VLMHarnessRunner closed-loop test
    class _ScriptedPolicyBackend:
        model_name = "test-vlm"

        def generate(self, prompt: str, *, images=None, json_schema=None, timeout_s=60.0):
            assert images and Path(images[0]).is_file()
            assert_zero_privilege_prompt(prompt)
            arm = ur5e.arm_names[0]
            tcp = ur5e.fk_tcp(env.state, arm=arm).p
            blk = env.objects["red_block"]
            plt = env.objects["blue_plate"]

            if blk.held_by_arm is None:
                diff = blk.xyz - tcp
                if np.linalg.norm(diff) <= 0.03:
                    chunk = ["GRASP", "MV_UP", "MV_UP"]
                else:
                    from eaisys.harness import delta_xyz_to_semantic_units

                    chunk = delta_xyz_to_semantic_units(diff, step_m=env.state.step_m, max_units_per_axis=2)[:5]
            else:
                goal = plt.xyz + np.array([0.0, 0.0, 0.03])
                diff = goal - tcp
                if np.linalg.norm(diff[:2]) <= 0.03:
                    chunk = ["RELEASE", "DONE"]
                else:
                    from eaisys.harness import delta_xyz_to_semantic_units

                    chunk = delta_xyz_to_semantic_units(diff, step_m=env.state.step_m, max_units_per_axis=2)[:5]

            payload = {
                "thought": "Step toward target in visual+proprioceptive loop.",
                "current_subtask": "transport" if blk.held_by_arm else "approach_grasp",
                "action_chunk": chunk or ["HOLD"],
            }
            return ModelResponse(text=json.dumps(payload), model="test-vlm", tokens_used=100)

    runner = VLMHarnessRunner(backend=_ScriptedPolicyBackend(), max_turns=12)
    rep = runner.run_episode(env, frame_save_dir=tmp_path / "frames")
    assert rep.success is True
    assert (tmp_path / "frames" / "final_success.png").is_file()

    # 3. URDF-conditioned SFT & GRPO export
    card = build_urdf_kinematic_card((ROBOTS_DIR / "ur5e" / "robot.urdf").read_text(encoding="utf-8"))
    assert "actuated_dof=" in card
    for fmt in ("llamafactory", "openai", "grpo"):
        out_jsonl = tmp_path / f"export_{fmt}.jsonl"
        meta = export_posttraining_dataset(REPO_ROOT / "artifacts" / "live_runs", out_jsonl, fmt=fmt)
        assert meta["episodes_exported"] >= 8
        assert meta["samples_exported"] >= 8
        first_line = json.loads(out_jsonl.read_text(encoding="utf-8").splitlines()[0])
        assert isinstance(first_line, dict)


def test_substrate_math_mjcf_retarget_and_site_snapshot(tmp_path: Path) -> None:
    u = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
    R_pi = axis_angle_to_matrix(u, np.pi)
    rv = matrix_to_rotvec(R_pi)
    np.testing.assert_allclose(
        axis_angle_to_matrix(rv / np.linalg.norm(rv), np.linalg.norm(rv)), R_pi, atol=1e-7
    )

    mjcf = tmp_path / "piv.xml"
    mjcf.write_text(
        """<mujoco model="piv_test">
          <worldbody>
            <body name="b1" pos="1 0 0" euler="0 0 90">
              <joint name="j1" type="hinge" pos="0.2 0 0" axis="0 0 1" range="-90 90"/>
              <site name="tip" pos="1 0 0"/>
            </body>
          </worldbody>
        </mujoco>""",
        encoding="utf-8",
    )
    tree = parse_robot_file(mjcf)
    assert np.isclose(tree.joints["j1"].upper, np.pi / 2.0, atol=1e-5)
    poses_zero = forward_kinematics_tree(tree, {"j1": 0.0})
    np.testing.assert_allclose(poses_zero["site::tip"].p, [1.0, 1.0, 0.0], atol=1e-5)

    T = 40
    pts = np.column_stack([np.linspace(0.30, 0.42, T), np.zeros(T), np.full(T, 0.22), np.zeros((T, 3)), np.ones(T)])
    grip = np.ones(T)
    grip[15:30] = 0.0
    ep = RawEpisode(
        dataset_name="unit_test",
        embodiment_id="franka_panda",
        episode_id="ep_0",
        task_instruction="pick red block",
        fps=20.0,
        arms={"main": ArmTrajectoryStream("main", eef_pose7_xyzw=pts, gripper_open=grip)},
    )
    bank = build_semantic_prior_bank([ep])
    assert "GRASP" in bank.unit_frequencies
    assert len(ALL_SEMANTIC_UNITS) == 25

    from eaisys.site import build_site_snapshot

    snap = build_site_snapshot(website_dir=tmp_path / "web")
    assert snap["stats"]["total_robots"] >= 12
    assert snap["stats"]["live_vlm_success_rate"] in (0.0, 1.0)


def test_urdf_free_finger_wiggle_ncc_and_broyden_adaptation() -> None:
    from eaisys.probing import SelfProbedEmbodimentModel, locate_tcp_by_finger_wiggle, track_tcp_ncc

    # 1. Synthetic finger-wiggle visual TCP localization
    img_open = np.full((120, 120, 3), 230, dtype=np.uint8)
    img_closed = img_open.copy()
    img_open[46:54, 62:68] = 40
    img_open[46:54, 72:78] = 40
    img_closed[46:54, 67:73] = 40
    (u, v), _ = locate_tcp_by_finger_wiggle(img_open, img_closed)
    assert abs(u - 70.0) < 3.0 and abs(v - 50.0) < 3.0

    # 2. NCC template tracking across shifted frame
    g0 = np.random.default_rng(0).integers(60, 180, size=(120, 120), dtype=np.uint8).astype(np.float32)
    g0[40:61, 50:71] = 20.0
    g0[48:53, 58:63] = 240.0
    tpl = g0[40:61, 50:71].copy()
    g1 = np.roll(np.roll(g0, 7, axis=0), -5, axis=1)
    (nu, nv), conf = track_tcp_ncc(g1, tpl, (60.0, 50.0), search_radius=16, stride=2)
    assert conf > 0.95 and abs(nu - 55.0) <= 1.0 and abs(nv - 57.0) <= 1.0

    # 3. Broyden online visual-motor Jacobian reduces prediction error
    J_true = np.array([[100.0, 20.0, 0.0], [0.0, 120.0, -15.0], [10.0, 0.0, 110.0]])
    J_init = J_true * 0.7
    mdl = SelfProbedEmbodimentModel(dof=3, J_vis=J_init.copy(), px_per_meter=200.0)
    dq = np.array([0.08, -0.05, 0.06])
    dp_true = J_true @ dq
    err_before = float(np.linalg.norm(dp_true - mdl.J_vis @ dq))
    mdl.broyden_update(dq, dp_true)
    err_after = float(np.linalg.norm(dp_true - mdl.J_vis @ dq))
    assert err_after < err_before

