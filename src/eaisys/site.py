"""Static observatory snapshot builder and local HTTP server for EAISys v0.3."""

from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from typing import Any

from eaisys import __version__
from eaisys.harness import (
    ALL_SEMANTIC_UNITS,
    HarnessPluginConfig,
    audit_interpreter_ast,
    load_interpreter_module,
    verify_interpreter,
)
from eaisys.memory import PlaybookStore
from eaisys.synthesis import SemanticPriorBank
from eaisys.urdf import default_harnesses_root, default_robots_root


def default_website_dir() -> Path:
    pkg_web = Path(__file__).resolve().parent / "_bundled" / "website"
    if pkg_web.is_dir():
        return pkg_web
    return Path(__file__).resolve().parents[2] / "website"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


_TASK_FAMILY_ORDER: dict[str, tuple[int, str, str, str]] = {
    "TwoArmLift": (0, "bimanual", "Heterogeneous Bimanual Lift", "异构双臂协同抬举"),
    "Lift_URDF_Free": (1, "urdf_free", "URDF-Free Few-Shot Self-Probing (Lift)", "无 URDF 自主试探操作 (Lift)"),
    "Stack_URDF_Free": (1, "urdf_free", "URDF-Free Few-Shot Self-Probing (Stack)", "无 URDF 自主试探操作 (Stack)"),
    "NutAssemblyRound": (2, "insertion", "Precision Ring-on-Peg Assembly", "精密圆环套柱装配"),
    "Wipe": (3, "contact", "Non-Prehensile Surface Wiping", "非抓持连续接触擦拭"),
    "Door": (4, "articulated", "Articulated Door Opening", "铰链门拉手开门"),
    "PickPlaceCan": (5, "sorting", "Walled-Bin Can Sorting", "深框易拉罐越障入格"),
    "PickPlaceMilk": (6, "sorting", "Walled-Bin Carton Sorting", "深框牛奶盒越障入格"),
    "Stack": (7, "stacking", "3D Precision Block Stacking", "三维高精度叠块"),
}


def _classify_demo(d: dict[str, Any]) -> dict[str, Any]:
    tname = str(d.get("task_name") or "Stack")
    order, fam_key, en_label, zh_label = _TASK_FAMILY_ORDER.get(
        tname, (99, "manipulation", tname, tname)
    )
    return {
        "order": order,
        "family_key": fam_key,
        "task_name": tname,
        "label_en": en_label,
        "label_zh": zh_label,
    }


def build_site_snapshot(
    website_dir: str | Path | None = None,
    robots_dir: str | Path | None = None,
    harnesses_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Assemble `website/data/snapshot.{json,js}` dynamically from verified URDF harnesses & 3D MuJoCo traces."""
    wdir = Path(website_dir) if website_dir else default_website_dir()
    rdir = Path(robots_dir) if robots_dir else default_robots_root()
    hdir = Path(harnesses_dir) if harnesses_dir else default_harnesses_root()
    root = _repo_root()
    live_dir = root / "artifacts" / "live_runs"
    priors_file = root / "artifacts" / "semantic_priors" / "semantic_prior_bank.json"

    seen_ids = {"franka_panda", "panda_omron", "aloha", "dual_franka", "arx_bimanual", "arx_mobile"}
    common_file = hdir / "common" / "urdf_interpreter_base.py"
    common_code = common_file.read_text(encoding="utf-8") if common_file.is_file() else ""
    common_meta_file = live_dir / "synth_common_library.json"
    common_meta = (
        json.loads(common_meta_file.read_text(encoding="utf-8"))
        if common_meta_file.is_file()
        else {"model": "frontier-llm", "lines_of_code": len(common_code.splitlines())}
    )

    # Auto-discover all verified 3D MuJoCo traces (single-arm & heterogeneous bimanual)
    mujoco_unseen_demos: list[dict[str, Any]] = []
    mujoco_synth_by_robot: dict[str, dict[str, Any]] = {}
    total_synth_tokens = int(common_meta.get("tokens_used", 0))

    if live_dir.is_dir():
        for mf in sorted(live_dir.glob("unseen_mujoco_*.json")):
            m_data = json.loads(mf.read_text(encoding="utf-8"))
            if not m_data.get("zero_privilege", False):
                continue
            if not m_data.get("success", False):
                continue
            vpath = m_data.get("video_path", "")
            if vpath and not (wdir / vpath).is_file():
                continue
            meta = _classify_demo(m_data)
            m_data["task_family_key"] = meta["family_key"]
            m_data["task_label_en"] = meta["label_en"]
            m_data["task_label_zh"] = meta["label_zh"]
            m_data["_sort_order"] = (meta["order"], str(m_data.get("run_id", "")))
            mujoco_unseen_demos.append(m_data)

            rid_mj = m_data.get("robot_id")
            srep_mj = m_data.get("stage1_urdf_and_synthesis", {}).get("synthesis_report", {})
            if rid_mj and srep_mj and "+" not in str(rid_mj) and str(rid_mj) not in mujoco_synth_by_robot:
                mujoco_synth_by_robot[str(rid_mj)] = srep_mj
                total_synth_tokens += int(srep_mj.get("total_tokens", 0))

    mujoco_unseen_demos.sort(key=lambda x: x.pop("_sort_order"))

    robots_payload: list[dict[str, Any]] = []
    for sub in sorted(rdir.iterdir()):
        urdf = sub / "robot.urdf"
        mod = hdir / "embodiments" / f"{sub.name}.py"
        if not sub.is_dir() or not urdf.is_file() or not mod.is_file():
            continue
        mod_code = mod.read_text(encoding="utf-8")
        urdf_xml = urdf.read_text(encoding="utf-8")
        ast_clean = len(audit_interpreter_ast(mod_code)) == 0
        interp = load_interpreter_module(mod, urdf, harnesses_root=hdir)
        ver = verify_interpreter(interp, urdf, check_rotations=True)

        synth_file = live_dir / f"synth_{sub.name}.json"
        if synth_file.is_file():
            synth_meta = json.loads(synth_file.read_text(encoding="utf-8"))
            total_synth_tokens += int(synth_meta.get("total_tokens", 0))
        else:
            synth_meta = mujoco_synth_by_robot.get(sub.name, {})

        is_bimanual = len(interp.arm_names) >= 2
        morphology = (
            "mobile_bimanual"
            if interp.supports_mobile_base and is_bimanual
            else "mobile_single"
            if interp.supports_mobile_base
            else "bimanual"
            if is_bimanual
            else "single_arm"
        )
        chains_info = {}
        for aname in interp.arm_names:
            jnames = interp.arm_joint_names(aname)
            dir_errs = ver.checks.get(f"{aname}_directional_errors_m", {})
            rot_errs = ver.checks.get(f"{aname}_rotation_errors_deg", {})
            valid_rot = [float(v) for v in rot_errs.values() if float(v) >= 0.0]
            max_dir_err_mm = max((float(v) * 1000.0 for v in dir_errs.values()), default=0.0)
            max_rot_err_deg = max(valid_rot, default=0.0)
            chains_info[aname] = {
                "dof": len(jnames),
                "tcp_link": interp.tcp_link_name(aname),
                "joints": list(jnames),
                "reset_tcp_xyz": ver.checks.get(f"{aname}_reset_fk", {}).get("true_tcp_xyz", []),
                "max_directional_err_mm": round(max_dir_err_mm, 3),
                "max_rotation_err_deg": round(max_rot_err_deg, 3),
            }

        robots_payload.append(
            {
                "robot_id": interp.robot_id,
                "split": "seen" if sub.name in seen_ids else "unseen",
                "morphology": morphology,
                "sidecar_used": (sub / "sidecar.yaml").is_file(),
                "used_common_library": "urdf_interpreter_base" in mod_code,
                "ast_audit_passed": ast_clean,
                "verified": ver.passed,
                "num_arms": len(interp.arm_names),
                "dof_per_arm": max(c["dof"] for c in chains_info.values()),
                "is_mobile": interp.supports_mobile_base,
                "is_bimanual": is_bimanual,
                "max_gripper_width_m": round(float(interp.max_gripper_width_m), 4),
                "chains": chains_info,
                "lines_of_code": len(mod_code.splitlines()),
                "urdf_preview": urdf_xml,
                "code_preview": mod_code,
                "synthesis": {
                    "model": synth_meta.get("model", "frontier-llm"),
                    "attempts": synth_meta.get("attempts", 1),
                    "tokens_used": synth_meta.get("total_tokens", 0),
                    "elapsed_s": synth_meta.get("total_elapsed_s", 0.0),
                    "turns": synth_meta.get("turns", []),
                },
            }
        )

    priors_payload: dict[str, Any] = {}
    if priors_file.is_file():
        priors_payload = SemanticPriorBank.load_json(priors_file).to_dict()

    playbook = PlaybookStore.load("v1")
    playbook_payload = {
        "version": playbook.version,
        "sections": [
            {"tag": s.tag, "title": s.title, "rules": list(s.rules)}
            for s in playbook.sections
        ],
    }

    task_families = sorted({d["task_family_key"] for d in mujoco_unseen_demos})
    distinct_tasks = sorted({str(d.get("task_name", "")) for d in mujoco_unseen_demos})
    models_used = sorted({str(d.get("model", "")) for d in mujoco_unseen_demos if d.get("model")})
    common_reusers = sum(1 for r in robots_payload if r["used_common_library"])

    self_probe_file = Path(__file__).resolve().parent / "probing" / "self_probe.py"
    self_probe_code = self_probe_file.read_text(encoding="utf-8") if self_probe_file.is_file() else ""

    universal_skills_payload: dict[str, Any] = {
        "common_base_path": "harnesses/common/urdf_interpreter_base.py",
        "common_base_loc": len(common_code.splitlines()),
        "common_base_reusers": [r["robot_id"] for r in robots_payload if r["used_common_library"]],
        "self_probe_path": "src/eaisys/probing/self_probe.py",
        "self_probe_loc": len(self_probe_code.splitlines()),
        "action_space_path": "src/eaisys/harness/action_space.py",
        "total_units": len(ALL_SEMANTIC_UNITS),
        "playbook_version": playbook.version,
        "playbook_rules_count": sum(len(s.rules) for s in playbook.sections),
        "retargeted_episodes": priors_payload.get("total_episodes", 0),
        "datasets_covered": len(priors_payload.get("dataset_counts", {})),
    }

    snapshot: dict[str, Any] = {
        "version": __version__,
        "repository_url": "https://github.com/Interactive-Active-AI/EAISys",
        "models_verified": models_used,
        "plugins": HarnessPluginConfig().enabled_plugin_names(),
        "playbook": playbook_payload,
        "universal_skills": universal_skills_payload,
        "stats": {
            "total_robots": len(robots_payload),
            "seen_robots": sum(1 for r in robots_payload if r["split"] == "seen"),
            "unseen_sidecarless_robots": sum(
                1 for r in robots_payload if r["split"] == "unseen" and not r["sidecar_used"]
            ),
            "common_lib_reusers": common_reusers,
            "semantic_action_units": len(ALL_SEMANTIC_UNITS),
            "common_lib_loc": len(common_code.splitlines()),
            "mujoco_unseen_videos": len(mujoco_unseen_demos),
            "bimanual_unseen_videos": sum(1 for d in mujoco_unseen_demos if d.get("is_bimanual")),
            "urdf_free_videos": sum(1 for d in mujoco_unseen_demos if d.get("urdf_free")),
            "task_families_count": len(task_families),
            "distinct_tasks_count": len(distinct_tasks),
            "live_vlm_rollouts": len(mujoco_unseen_demos),
            "live_vlm_success_rate": (
                round(
                    sum(1 for x in mujoco_unseen_demos if x.get("success"))
                    / max(1, len(mujoco_unseen_demos)),
                    4,
                )
                if mujoco_unseen_demos
                else 0.0
            ),
            "retargeted_episodes": priors_payload.get("total_episodes", 0),
            "datasets_covered": len(priors_payload.get("dataset_counts", {})),
            "total_synthesis_tokens": total_synth_tokens,
        },
        "common_library": {
            **common_meta,
            "code": common_code,
        },
        "mujoco_unseen_demos": mujoco_unseen_demos,
        "robots": robots_payload,
        "priors": priors_payload,
    }

    data_dir = wdir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(snapshot, indent=2, ensure_ascii=False)
    (data_dir / "snapshot.json").write_text(json_text, encoding="utf-8")
    (data_dir / "snapshot.js").write_text(
        f"window.__EAISYS_SNAPSHOT__ = {json_text};\n", encoding="utf-8"
    )
    return snapshot


def serve_site(
    host: str = "127.0.0.1",
    port: int = 8765,
    website_dir: str | Path | None = None,
    rebuild: bool = True,
) -> None:
    wdir = Path(website_dir) if website_dir else default_website_dir()
    if rebuild:
        build_site_snapshot(website_dir=wdir)

    class _SafeStaticHandler(SimpleHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            try:
                super().log_message(format, *args)
            except OSError:
                pass

    handler = partial(_SafeStaticHandler, directory=str(wdir))
    server = ThreadingHTTPServer((host, port), handler)
    print(f"EAISys Observatory live at http://{host}:{port}/  (serving {wdir})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
