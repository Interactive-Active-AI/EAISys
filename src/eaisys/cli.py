"""Command-line interface for EAISys v0.3 (URDF-Centric Embodied AI Harness)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import yaml

from eaisys.backends import SUPPORTED_PROVIDERS, create_backend
from eaisys.datasets import load_aloha_hdf5_episodes, load_episodes_by_adapter, load_robomind_episode
from eaisys.harness import (
    HarnessPluginConfig,
    VLMHarnessRunner,
    load_interpreter_module,
    make_standard_pick_place_env,
    verify_interpreter,
    verify_interpreter_isolated,
)
from eaisys.posttrain import export_posttraining_dataset
from eaisys.synthesis import SemanticPriorBank, URDFHarnessSynthesizer, build_semantic_prior_bank
from eaisys.urdf import default_harnesses_root, default_robots_root


def _default_datasets_config() -> Path:
    pkg_cfg = Path(__file__).resolve().parent / "_bundled" / "configs" / "datasets.yaml"
    if pkg_cfg.is_file():
        return pkg_cfg
    return Path(__file__).resolve().parents[2] / "configs" / "datasets.yaml"


def _make_backend(args: argparse.Namespace):
    return create_backend(
        provider=getattr(args, "provider", "openai"),
        model=getattr(args, "model", None),
        reasoning_effort=getattr(args, "effort", "high"),
        fast_mode=bool(getattr(args, "fast", False)),
        base_url=getattr(args, "base_url", None),
    )


def _make_plugin_cfg(args: argparse.Namespace) -> HarnessPluginConfig:
    return HarnessPluginConfig(
        playbook_memory=not bool(getattr(args, "no_playbook", False)),
        playbook_version=str(getattr(args, "playbook", "v1")),
        cross_embodiment_priors=not bool(getattr(args, "no_priors", False)),
        visual_diff_frame=bool(getattr(args, "visual_diff", False)),
    )


def cmd_list_harnesses(_args: argparse.Namespace) -> int:
    rroot = default_robots_root()
    hroot = default_harnesses_root()
    rows = []
    for d in sorted(rroot.iterdir()):
        urdf = d / "robot.urdf"
        mod = hroot / "embodiments" / f"{d.name}.py"
        if not d.is_dir() or not urdf.is_file():
            continue
        status = "missing"
        arms: list[str] = []
        mobile = False
        if mod.is_file():
            interp = load_interpreter_module(mod, urdf, harnesses_root=hroot)
            ver = verify_interpreter(interp, urdf)
            status = "verified" if ver.passed else "failed"
            arms = list(interp.arm_names)
            mobile = interp.supports_mobile_base
        rows.append(
            {
                "robot_id": d.name,
                "interpreter_module": f"harnesses/embodiments/{d.name}.py" if mod.is_file() else None,
                "status": status,
                "arms": arms,
                "supports_mobile_base": mobile,
                "has_sidecar": (d / "sidecar.yaml").is_file(),
            }
        )
    print(json.dumps(rows, indent=2))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    rroot = default_robots_root()
    hroot = default_harnesses_root()
    targets = [args.robot] if args.robot else [d.name for d in sorted(rroot.iterdir()) if (d / "robot.urdf").is_file()]
    all_ok = True
    reports = []
    for rid in targets:
        urdf = rroot / rid / "robot.urdf"
        mod = hroot / "embodiments" / f"{rid}.py"
        if args.in_process:
            interp = load_interpreter_module(mod, urdf, harnesses_root=hroot)
            rep = verify_interpreter(interp, urdf)
        else:
            rep = verify_interpreter_isolated(mod, urdf, harnesses_root=hroot)
        all_ok = all_ok and rep.passed
        reports.append(rep.to_dict())
    print(json.dumps(reports if len(reports) > 1 else reports[0], indent=2))
    return 0 if all_ok else 1


def cmd_synthesize(args: argparse.Namespace) -> int:
    backend = _make_backend(args)
    synth = URDFHarnessSynthesizer(backend=backend, max_attempts=args.max_attempts)

    demos = []
    if args.calib_robomind_hdf5:
        demos.append(load_robomind_episode(args.calib_robomind_hdf5))
    if args.calib_aloha_hdf5:
        demos.extend(load_aloha_hdf5_episodes(args.calib_aloha_hdf5, max_episodes=args.k_shot))

    hroot = default_harnesses_root()
    common_p = Path(args.common_lib) if args.common_lib else (hroot / "common" / "urdf_interpreter_base.py")
    out_p = Path(args.out) if args.out else (hroot / "embodiments" / f"{Path(args.urdf).parent.name}.py")

    rep = synth.synthesize_embodiment_interpreter(
        urdf_path=args.urdf,
        output_py_path=out_p,
        sidecar_path=args.sidecar,
        sample_episodes=demos or None,
        common_library_path=common_p if common_p.is_file() and not args.no_common else None,
        record_full_trace=bool(args.record_trace),
    )
    print(json.dumps(rep.to_dict(), indent=2, ensure_ascii=False))
    return 0 if rep.passed else 1


def cmd_abstract_common(args: argparse.Namespace) -> int:
    backend = _make_backend(args)
    synth = URDFHarnessSynthesizer(backend=backend)
    hroot = default_harnesses_root()
    seeds = [hroot / "embodiments" / f"{r}.py" for r in args.seeds]
    out_p = Path(args.out) if args.out else (hroot / "common" / "urdf_interpreter_base.py")
    res = synth.synthesize_common_library(seed_interpreter_paths=seeds, output_common_path=out_p)
    print(json.dumps(res, indent=2))
    return 0


def cmd_play(args: argparse.Namespace) -> int:
    backend = _make_backend(args)
    rroot = default_robots_root()
    hroot = default_harnesses_root()
    urdf = rroot / args.robot / "robot.urdf"
    mod = hroot / "embodiments" / f"{args.robot}.py"
    interp = load_interpreter_module(mod, urdf, harnesses_root=hroot)
    env = make_standard_pick_place_env(interp, seed=args.seed)
    if args.task:
        env.task_instruction = args.task

    demos = None
    if args.priors and Path(args.priors).is_file():
        bank = SemanticPriorBank.load_json(args.priors)
        demos = bank.retrieve(env.task_instruction, top_k=2)

    runner = VLMHarnessRunner(backend=backend, plugin_cfg=_make_plugin_cfg(args), max_turns=args.max_turns)
    rep = runner.run_episode(env, in_context_demos=demos, frame_save_dir=args.save_frames)
    if args.save_report:
        Path(args.save_report).write_text(json.dumps(rep.to_dict(), indent=2), encoding="utf-8")
    print(json.dumps(rep.to_dict(), indent=2, ensure_ascii=False))
    return 0 if rep.success else 1


def cmd_play_mujoco(args: argparse.Namespace) -> int:
    from eaisys.harness.mujoco_env import run_unseen_mujoco_pipeline
    from eaisys.site import default_website_dir

    backend = _make_backend(args)
    rroot = default_robots_root()
    hroot = default_harnesses_root()
    wdir = default_website_dir()
    repo_root = Path(__file__).resolve().parents[2]
    env_kwargs: dict[str, object] = {}
    if args.env == "Door" and args.no_latch:
        env_kwargs["use_latch"] = False
    if args.env == "Wipe" and args.wipe_markers is not None:
        from robosuite.environments.manipulation.wipe import DEFAULT_WIPE_CONFIG

        wcfg = dict(DEFAULT_WIPE_CONFIG)
        wcfg["num_markers"] = int(args.wipe_markers)
        env_kwargs["task_config"] = wcfg

    res = run_unseen_mujoco_pipeline(
        run_id=args.run_id,
        robosuite_robot=args.robosuite_robot,
        robot_id=args.robot_id,
        robosuite_env_name=args.env,
        task_instruction=args.task,
        backend=backend,
        robots_root=rroot,
        harnesses_root=hroot,
        media_out_dir=wdir / "media",
        trace_json_out=repo_root / "artifacts" / "live_runs" / f"unseen_mujoco_{args.run_id}.json",
        priors_path=repo_root / "artifacts" / "semantic_priors" / "semantic_prior_bank.json",
        seed=args.seed,
        max_turns=args.max_turns,
        force_resynthesize=not args.reuse_interpreter,
        env_kwargs=env_kwargs,
        plugin_cfg=_make_plugin_cfg(args),
    )
    print(
        json.dumps(
            {
                "run_id": res["run_id"],
                "success": res["success"],
                "video_path": res["video_path"],
                "total_vlm_turns": res["total_vlm_turns"],
                "final_metrics": res["final_metrics"],
            },
            indent=2,
        )
    )
    return 0 if res["success"] else 1


def cmd_play_bimanual_mujoco(args: argparse.Namespace) -> int:
    from eaisys.harness.mujoco_env import run_bimanual_mujoco_pipeline
    from eaisys.site import default_website_dir

    backend = _make_backend(args)
    rroot = default_robots_root()
    hroot = default_harnesses_root()
    wdir = default_website_dir()
    repo_root = Path(__file__).resolve().parents[2]
    res = run_bimanual_mujoco_pipeline(
        run_id=args.run_id,
        right_robosuite_robot=args.right_robot,
        right_robot_id=args.right_robot_id,
        left_robosuite_robot=args.left_robot,
        left_robot_id=args.left_robot_id,
        robosuite_env_name=args.env,
        task_instruction=args.task,
        backend=backend,
        robots_root=rroot,
        harnesses_root=hroot,
        media_out_dir=wdir / "media",
        trace_json_out=repo_root / "artifacts" / "live_runs" / f"unseen_mujoco_{args.run_id}.json",
        priors_path=repo_root / "artifacts" / "semantic_priors" / "semantic_prior_bank.json",
        seed=args.seed,
        max_turns=args.max_turns,
        force_resynthesize=not args.reuse_interpreters,
        plugin_cfg=_make_plugin_cfg(args),
    )
    print(
        json.dumps(
            {
                "run_id": res["run_id"],
                "success": res["success"],
                "video_path": res["video_path"],
                "total_vlm_turns": res["total_vlm_turns"],
                "final_metrics": res["final_metrics"],
            },
            indent=2,
        )
    )
    return 0 if res["success"] else 1


def cmd_play_urdf_free(args: argparse.Namespace) -> int:
    from eaisys.probing import run_urdf_free_self_probe_pipeline
    from eaisys.site import default_website_dir

    backend = _make_backend(args)
    wdir = default_website_dir()
    repo_root = Path(__file__).resolve().parents[2]
    res = run_urdf_free_self_probe_pipeline(
        run_id=args.run_id,
        robosuite_robot=args.robosuite_robot,
        robosuite_env_name=args.env,
        task_instruction=args.task,
        backend=backend,
        media_out_dir=wdir / "media",
        trace_json_out=repo_root / "artifacts" / "live_runs" / f"unseen_mujoco_{args.run_id}.json",
        priors_path=repo_root / "artifacts" / "semantic_priors" / "semantic_prior_bank.json",
        seed=args.seed,
        max_turns=args.max_turns,
        plugin_cfg=_make_plugin_cfg(args),
    )
    print(
        json.dumps(
            {
                "run_id": res["run_id"],
                "urdf_free": True,
                "success": res["success"],
                "video_path": res["video_path"],
                "total_vlm_turns": res["total_vlm_turns"],
                "final_metrics": res["final_metrics"],
            },
            indent=2,
        )
    )
    return 0 if res["success"] else 1


def cmd_retarget(args: argparse.Namespace) -> int:
    raw_root = (args.data_root or os.environ.get("EAISYS_DATA_ROOT") or "").strip()
    if not raw_root:
        print("Error: specify --data-root or set EAISYS_DATA_ROOT.", file=sys.stderr)
        return 2
    data_root = Path(raw_root)
    if not data_root.is_dir():
        print(f"Error: dataset directory '{data_root}' does not exist.", file=sys.stderr)
        return 2

    cfg_path = Path(args.config) if args.config else _default_datasets_config()
    if not cfg_path.is_file():
        cfg_path = _default_datasets_config()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    all_eps = []
    for ds_name, spec in cfg.get("datasets", {}).items():
        full_p = data_root / spec["relative_path"]
        if not full_p.exists():
            continue
        all_eps.extend(
            load_episodes_by_adapter(
                adapter=spec["adapter"],
                path=full_p,
                robot_id=spec["robot_id"],
                dataset_name=ds_name,
                max_episodes=args.max_per_dataset,
            )
        )
    if not all_eps:
        print(f"Error: zero episodes found under '{data_root}'.", file=sys.stderr)
        return 2

    bank = build_semantic_prior_bank(all_eps, max_exemplars=args.max_exemplars)
    bank.save_json(args.out)
    print(
        json.dumps(
            {
                "saved_to": str(args.out),
                "total_episodes": bank.total_episodes,
                "dataset_counts": bank.dataset_counts,
                "embodiment_counts": bank.embodiment_counts,
            },
            indent=2,
        )
    )
    return 0


def cmd_export_sft(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    live_dir = Path(args.live_runs_dir) if args.live_runs_dir else (repo_root / "artifacts" / "live_runs")
    res = export_posttraining_dataset(
        live_runs_dir=live_dir,
        output_jsonl=args.out,
        fmt=args.format,
        include_urdf_card=not args.no_urdf_card,
        only_successful=not args.include_failed,
    )
    print(json.dumps(res, indent=2))
    return 0


def cmd_gumi(args: argparse.Namespace) -> int:
    from eaisys.gumi import serve_gumi

    serve_gumi(
        robot_id=args.robot,
        robosuite_env=args.env,
        robosuite_robot=args.robosuite_robot,
        task_instruction=args.task,
        host=args.host,
        port=args.port,
        save_dir=args.save_dir,
        seed=args.seed,
    )
    return 0


def cmd_build_site(args: argparse.Namespace) -> int:
    from eaisys.site import build_site_snapshot, default_website_dir

    wdir = Path(args.website_dir) if args.website_dir else default_website_dir()
    snap = build_site_snapshot(website_dir=wdir)
    print(json.dumps({"website_dir": str(wdir), "stats": snap["stats"]}, indent=2))
    return 0


def cmd_serve_site(args: argparse.Namespace) -> int:
    from eaisys.site import serve_site

    serve_site(
        host=args.host,
        port=args.port,
        website_dir=args.website_dir,
        rebuild=not args.no_rebuild,
    )
    return 0


def _add_llm_flags(sp: argparse.ArgumentParser) -> None:
    sp.add_argument(
        "--provider",
        choices=list(SUPPORTED_PROVIDERS),
        default="openai",
        help="Model provider: openai (default), openrouter, vllm, anthropic, or codex",
    )
    sp.add_argument("--model", default=None, help="Model slug (provider default if omitted)")
    sp.add_argument("--effort", default="high", help="Reasoning effort (e.g., high, xhigh, max)")
    sp.add_argument("--fast", action="store_true", help="Enable priority/fast service tier")
    sp.add_argument("--base-url", default=None, help="Optional OpenAI-compatible API base URL")


def _add_plugin_flags(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--playbook", default="v1", help="Playbook policy version under configs/playbooks/ (default: v1)")
    sp.add_argument("--no-playbook", action="store_true", help="Ablation toggle: disable PlaybookMemoryPlugin")
    sp.add_argument("--no-priors", action="store_true", help="Ablation toggle: disable CrossEmbodimentPriorsPlugin")
    sp.add_argument("--visual-diff", action="store_true", help="Attach previous turn frame alongside current frame")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="eaisys", description="EAISys: URDF-Centric Embodied AI Harness CLI")
    sub = p.add_subparsers(dest="command", required=True)

    sp_ls = sub.add_parser("list-harnesses", help="List all URDF embodiments and LLM-synthesized interpreters")
    sp_ls.set_defaults(func=cmd_list_harnesses)

    sp_ver = sub.add_parser("verify", help="Verify LLM-synthesized interpreter(s) against ground-truth URDF FK/limits/SO(3)")
    sp_ver.add_argument("--robot", default=None, help="Robot ID (verifies all if omitted)")
    sp_ver.add_argument("--in-process", action="store_true", help="Run in current process instead of isolated subprocess")
    sp_ver.set_defaults(func=cmd_verify)

    sp_syn = sub.add_parser("synthesize", help="Use Frontier LLM to synthesize & verify a URDF interpreter")
    sp_syn.add_argument("urdf", help="Path to target robot .urdf")
    sp_syn.add_argument("--sidecar", default=None, help="Optional YAML sidecar")
    sp_syn.add_argument("--out", default=None, help="Output Python interpreter path")
    sp_syn.add_argument("--common-lib", default=None, help="Cross-URDF common library path")
    sp_syn.add_argument("--no-common", action="store_true", help="Synthesize from scratch without common library")
    sp_syn.add_argument("--calib-robomind-hdf5", default=None)
    sp_syn.add_argument("--calib-aloha-hdf5", default=None)
    sp_syn.add_argument("--k-shot", type=int, default=3)
    sp_syn.add_argument("--max-attempts", type=int, default=3)
    sp_syn.add_argument("--record-trace", action="store_true")
    _add_llm_flags(sp_syn)
    sp_syn.set_defaults(func=cmd_synthesize)

    sp_abs = sub.add_parser("abstract-common", help="Use Frontier LLM to abstract Cross-URDF Common Library from seed interpreters")
    sp_abs.add_argument("--seeds", nargs="+", default=["franka_panda", "aloha", "panda_omron"])
    sp_abs.add_argument("--out", default=None)
    _add_llm_flags(sp_abs)
    sp_abs.set_defaults(func=cmd_abstract_common)

    sp_play = sub.add_parser("play", help="Run Frontier VLM closed-loop manipulation in lightweight tabletop env")
    sp_play.add_argument("--robot", required=True, help="Robot ID (e.g., franka_panda, ur5e, kinova_gen3)")
    sp_play.add_argument("--task", default=None, help="Optional custom task instruction")
    sp_play.add_argument("--seed", type=int, default=7)
    sp_play.add_argument("--max-turns", type=int, default=16)
    sp_play.add_argument("--priors", default="artifacts/semantic_priors/semantic_prior_bank.json")
    sp_play.add_argument("--save-frames", default=None)
    sp_play.add_argument("--save-report", default=None)
    _add_llm_flags(sp_play)
    _add_plugin_flags(sp_play)
    sp_play.set_defaults(func=cmd_play)

    sp_mj = sub.add_parser(
        "play-mujoco",
        help="End-to-end Unseen Embodiment 3D MuJoCo pipeline: URDF -> LLM Interpreter -> VLM Closed-Loop MP4 + Trace",
    )
    sp_mj.add_argument("--robosuite-robot", required=True, help="Robosuite robot name (e.g., UR5e, Kinova3, IIWA, Sawyer)")
    sp_mj.add_argument("--robot-id", required=True, help="Target robot ID (e.g., ur5e_mujoco, sawyer_mujoco)")
    sp_mj.add_argument("--env", default="Stack", help="Robosuite 3D task (Stack, Door, PickPlaceMilk, PickPlaceCan, NutAssemblyRound, Wipe)")
    sp_mj.add_argument(
        "--task",
        default="Pick up the small red cube from the table and stack it stably on top of the green cube, then lift the gripper clear and emit DONE.",
    )
    sp_mj.add_argument("--run-id", required=True, help="Run folder name under website/media/")
    sp_mj.add_argument("--seed", type=int, default=11)
    sp_mj.add_argument("--max-turns", type=int, default=24)
    sp_mj.add_argument("--no-latch", action="store_true", help="Disable door latch when --env Door")
    sp_mj.add_argument("--wipe-markers", type=int, default=None, help="Optional marker count when --env Wipe")
    sp_mj.add_argument("--reuse-interpreter", action="store_true", help="Skip re-synthesizing interpreter if already present")
    _add_llm_flags(sp_mj)
    _add_plugin_flags(sp_mj)
    sp_mj.set_defaults(func=cmd_play_mujoco)

    sp_bm = sub.add_parser(
        "play-bimanual-mujoco",
        help="Heterogeneous Unseen-URDF Bimanual 3D MuJoCo pipeline (`left_chunk` + `right_chunk`)",
    )
    sp_bm.add_argument("--left-robot", default="Kinova3", help="Robosuite robot for left (+Y) arm (default: Kinova3)")
    sp_bm.add_argument("--left-robot-id", default="kinova3_mujoco", help="URDF robot_id for left arm")
    sp_bm.add_argument("--right-robot", default="UR5e", help="Robosuite robot for right (-Y) arm (default: UR5e)")
    sp_bm.add_argument("--right-robot-id", default="ur5e_mujoco", help="URDF robot_id for right arm")
    sp_bm.add_argument("--env", default="TwoArmLift", help="Robosuite bimanual task (default: TwoArmLift)")
    sp_bm.add_argument(
        "--task",
        default="Coordinate both unseen arms to grasp the two opposite green handles of the pot simultaneously and lift the pot cleanly above the table, then emit DONE.",
    )
    sp_bm.add_argument("--run-id", required=True, help="Run folder name under website/media/")
    sp_bm.add_argument("--seed", type=int, default=11)
    sp_bm.add_argument("--max-turns", type=int, default=24)
    sp_bm.add_argument("--reuse-interpreters", action="store_true", help="Reuse verified LLM interpreters if present")
    _add_llm_flags(sp_bm)
    _add_plugin_flags(sp_bm)
    sp_bm.set_defaults(func=cmd_play_bimanual_mujoco)

    sp_uf = sub.add_parser(
        "play-urdf-free",
        help="Pillar 2: ZERO-URDF Active Self-Probing (few-shot embodiment discovery) + Zero-Privilege 3D Manipulation",
    )
    sp_uf.add_argument("--robosuite-robot", default="Panda", help="Unknown robosuite robot (NO URDF provided)")
    sp_uf.add_argument("--env", default="Lift", help="Robosuite 3D task (Lift, Stack)")
    sp_uf.add_argument(
        "--task",
        default="Grasp the red cube from the table and lift it cleanly into the air, then emit DONE.",
    )
    sp_uf.add_argument("--run-id", required=True, help="Run folder name under website/media/")
    sp_uf.add_argument("--seed", type=int, default=11)
    sp_uf.add_argument("--max-turns", type=int, default=18)
    _add_llm_flags(sp_uf)
    _add_plugin_flags(sp_uf)
    sp_uf.set_defaults(func=cmd_play_urdf_free)

    sp_ret = sub.add_parser("retarget", help="Retarget multi-embodiment datasets into Semantic Action Space priors")
    sp_ret.add_argument("--config", default=None)
    sp_ret.add_argument("--data-root", default=None)
    sp_ret.add_argument("--out", default="artifacts/semantic_priors/semantic_prior_bank.json")
    sp_ret.add_argument("--max-per-dataset", type=int, default=15)
    sp_ret.add_argument("--max-exemplars", type=int, default=32)
    sp_ret.set_defaults(func=cmd_retarget)

    sp_sft = sub.add_parser("export-sft", help="Export URDF-conditioned SFT / LoRA / GRPO post-training JSONL")
    sp_sft.add_argument("--live-runs-dir", default=None)
    sp_sft.add_argument("--out", default="artifacts/posttrain/urdf_rua_sft.jsonl")
    sp_sft.add_argument("--format", choices=["llamafactory", "openai", "grpo"], default="llamafactory")
    sp_sft.add_argument("--no-urdf-card", action="store_true")
    sp_sft.add_argument("--include-failed", action="store_true")
    sp_sft.set_defaults(func=cmd_export_sft)

    sp_gumi = sub.add_parser("gumi", help="Launch GUMI-URDF browser manipulation interface for human teleop")
    sp_gumi.add_argument("--robot", default="ur5e_mujoco", help="Synthesized robot_id")
    sp_gumi.add_argument("--env", default=None, help="Optional robosuite 3D env (e.g., Stack, Door); defaults to lightweight 2D if omitted")
    sp_gumi.add_argument("--robosuite-robot", default=None, help="Optional robosuite robot name (e.g., UR5e)")
    sp_gumi.add_argument("--task", default="Teleoperate the robot to complete the manipulation task.")
    sp_gumi.add_argument("--host", default="127.0.0.1")
    sp_gumi.add_argument("--port", type=int, default=8766)
    sp_gumi.add_argument("--save-dir", default="artifacts/gumi_demos")
    sp_gumi.add_argument("--seed", type=int, default=11)
    sp_gumi.set_defaults(func=cmd_gumi)

    sp_bsite = sub.add_parser("build-site", help="Build website/data/snapshot.{json,js} from live LLM artifacts")
    sp_bsite.add_argument("--website-dir", default=None)
    sp_bsite.set_defaults(func=cmd_build_site)

    sp_ssite = sub.add_parser("serve-site", help="Build snapshot and serve EAISys observatory website locally")
    sp_ssite.add_argument("--host", default="127.0.0.1")
    sp_ssite.add_argument("--port", type=int, default=8765)
    sp_ssite.add_argument("--no-rebuild", action="store_true")
    sp_ssite.add_argument("--website-dir", default=None)
    sp_ssite.set_defaults(func=cmd_serve_site)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
