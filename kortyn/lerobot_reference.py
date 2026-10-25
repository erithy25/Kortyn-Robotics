"""Reference measurement with LeRobot's own evaluation script on the migrated checkpoint.

Runs the official ``lerobot-eval`` (synchronous rollouts, vectorized envs, same seeds as Kortyn) with the
sampler of a Brain profile and stores the result next to the Kortyn results. Together with the
checkpoint's published numbers and Kortyn's ``referenz_sync`` run this separates

* checkpoint migration + sampler choice   (published vs. lerobot-eval),
* Kortyn's Brain/Agent pipeline            (lerobot-eval vs. referenz_sync),
* real-time latency and the network        (referenz_sync vs. the network scenarios).

PushT (phase 1, Diffusion Policy), Meta-World (phase 3a, SmolVLA: env task, camera key and flow-matching
steps from the experiment config; one env at a time) and LIBERO (phase 3: suite, task ids, camera size and
episodes per task from the experiment config; ``--episodes`` counts episodes per task, as lerobot-eval does):

    python -m kortyn.lerobot_reference --profile ddim10
    python -m kortyn.lerobot_reference --scheduler DDPM --num-inference-steps 100 --name ddpm100
    python -m kortyn.lerobot_reference --config kortyn/experiments/phase3a/smolvla_coffee_push.yaml \
        --profile smolvla --episodes 50
    python -m kortyn.lerobot_reference --config kortyn/experiments/phase3/libero_pilot.yaml \
        --profile smolvla_libero --threads 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from kortyn.config import DEFAULT_BASE_CONFIG, ConfigError, LiberoEnvConfig, load_config

LOG = logging.getLogger("kortyn.lerobot_reference")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run LeRobot's official eval on the migrated checkpoint.")
    parser.add_argument("--config", default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument("--profile", default=None, help="take scheduler/steps from this Brain profile")
    parser.add_argument("--scheduler", choices=["DDPM", "DDIM"], default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--name", default=None, help="result folder name (default: profile or sampler)")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="parallel envs (default: all episodes)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=None, help="torch/OpenMP threads of lerobot-eval")
    parser.add_argument(
        "--task-ids",
        default=None,
        help="LIBERO: evaluate only these tasks of the config (comma-separated, seeds as in the config)",
    )
    parser.add_argument(
        "--from-eval-info",
        default=None,
        help="summarize an eval_info.json from an earlier run of the identical command instead of re-running",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    try:
        config = load_config(args.config)
        profile = config.profile(args.profile) if args.profile else None
    except ConfigError as exc:
        LOG.error("%s", exc)
        return 2
    scheduler = args.scheduler or (profile.noise_scheduler_type if profile else "DDIM")
    steps = args.num_inference_steps or (profile.num_inference_steps if profile else 10)
    name = args.name or args.profile or f"{scheduler.lower()}{steps}"
    metaworld = config.env.metaworld if config.env.kind == "metaworld" else None
    libero = config.env.libero if config.env.kind == "libero" else None
    default_episodes = libero.episodes_per_task if libero is not None else config.experiment.episodes
    episodes = args.episodes or default_episodes
    # Meta-World and LIBERO render camera frames per env and step: evaluate one env at a time
    batch = args.batch_size or (1 if metaworld is not None or libero is not None else episodes)
    if libero is not None:
        if libero.init_state_offset != 0:
            LOG.error("lerobot-eval always starts at init state 0; libero.init_state_offset must be 0")
            return 2
        sampler = f"flow-{steps}"
        policy_args = [f"--policy.num_steps={steps}"]
        if profile is not None and profile.n_action_steps is not None:
            policy_args.append(f"--policy.n_action_steps={profile.n_action_steps}")
        evaluated = list(libero.task_ids)
        if args.task_ids is not None:
            evaluated = [int(part) for part in args.task_ids.split(",") if part.strip()]
            unknown = sorted(set(evaluated) - set(libero.task_ids))
            if not evaluated or unknown:
                LOG.error("--task-ids must be a non-empty subset of the config's tasks %s", libero.task_ids)
                return 2
        env_args = libero_env_args(libero, evaluated)
        # LeRobot's env names the cameras image/image2; the checkpoint was trained with them renamed
        rename = {
            "observation.images.image": libero.cameras.get("agentview_image", "observation.images.camera1"),
            "observation.images.image2": libero.cameras.get(
                "robot0_eye_in_hand_image", "observation.images.camera2"
            ),
        }
        env_args.append("--rename_map=" + json.dumps(rename))
        from kortyn.libero_env import prepare_libero

        prepare_libero(libero)
    elif metaworld is not None:
        # LeRobot's Meta-World env delivers the camera as observation.image; the policy's key comes from
        # Kortyn's features_map (pixels -> policy key), exactly as the agent maps it
        image_key = config.env.features_map.get("pixels", "observation.images.camera1")
        sampler = f"flow-{steps}"
        policy_args = [f"--policy.num_steps={steps}"]
        env_args = [
            "--env.type=metaworld",
            f"--env.task={metaworld.task}",
            "--rename_map=" + json.dumps({"observation.image": image_key}),
        ]
    else:
        sampler = f"{scheduler}-{steps}"
        policy_args = [f"--policy.noise_scheduler_type={scheduler}", f"--policy.num_inference_steps={steps}"]
        env_args = ["--env.type=pusht"]

    from kortyn.checkpoint import ensure_checkpoint

    checkpoint = ensure_checkpoint(config.brain.checkpoint).resolve()
    digest = hashlib.sha256()
    with (checkpoint / "model.safetensors").open("rb") as weights:
        for block in iter(lambda: weights.read(1 << 20), b""):
            digest.update(block)
    out_dir = Path(config.experiment.results_dir) / "lerobot_eval" / name
    work_dir = out_dir / "lerobot_output"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    command = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_eval",
        f"--policy.path={checkpoint}",
        f"--policy.device={args.device}",
        *policy_args,
        *env_args,
        f"--eval.n_episodes={episodes}",
        f"--eval.batch_size={batch}",
        "--eval.use_async_envs=false",
        f"--seed={config.experiment.seed_start}",
        f"--output_dir={work_dir}",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.from_eval_info:
        source = Path(args.from_eval_info)
        LOG.info("summarizing %s (produced by: %s)", source, " ".join(command))
        info = json.loads(source.read_text(encoding="utf-8"))
        provenance = (
            f"imported from an earlier lerobot-eval run with the same parameters ({source.name}); "
            "that run used a checkpoint copy whose weights must match checkpoint_sha256"
        )
    else:
        LOG.info("running: %s", " ".join(command))
        environment = dict(os.environ)
        if args.threads is not None:
            for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                environment[variable] = str(args.threads)
        with (out_dir / "lerobot_eval.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, check=False, env=environment
            )
        if result.returncode != 0:
            LOG.error(
                "lerobot-eval failed with code %d, see %s", result.returncode, out_dir / "lerobot_eval.log"
            )
            return 1
        info = json.loads((work_dir / "eval_info.json").read_text(encoding="utf-8"))
        provenance = "run by kortyn.lerobot_reference"
    if libero is not None:
        success_by_seed = libero_success_by_seed(info, libero, evaluated)
        successes = list(success_by_seed.values())
    else:
        successes = [bool(s) for s in info["per_task"][0]["metrics"]["successes"]]
        seeds = range(config.experiment.seed_start, config.experiment.seed_start + len(successes))
        success_by_seed = {str(seed): ok for seed, ok in zip(seeds, successes, strict=True)}
    summary = {
        "tool": "lerobot-eval (LeRobot's official evaluation, synchronous)",
        "command": " ".join(command[1:]),
        "provenance": provenance,
        "checkpoint": f"{config.brain.checkpoint.repo_id}@{config.brain.checkpoint.revision}"
        + (" (migrated)" if metaworld is None and libero is None else ""),
        "checkpoint_sha256": digest.hexdigest(),
        "sampler": sampler,
        "device": args.device,
        "episodes": len(successes),
        "successes": sum(successes),
        "success_rate": sum(successes) / len(successes) if successes else None,
        "avg_max_reward": info.get("overall", {}).get("avg_max_reward"),
        "success_by_seed": success_by_seed,
        "threads": args.threads,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    shutil.rmtree(work_dir / "videos", ignore_errors=True)  # large and not needed for the report
    LOG.info("%s: %d/%d successes", summary["sampler"], summary["successes"], summary["episodes"])
    return 0


def libero_env_args(libero: LiberoEnvConfig, task_ids: list[int]) -> list[str]:
    """lerobot-eval arguments for LIBERO episodes of an experiment config (LeRobot's env defaults)."""
    return [
        "--env.type=libero",
        f"--env.task={libero.suite}",
        "--env.task_ids=[" + ",".join(str(task_id) for task_id in task_ids) + "]",
        f"--env.observation_height={libero.observation_height}",
        f"--env.observation_width={libero.observation_width}",
        f"--env.control_mode={libero.control_mode}",
        "--env.init_states=true",
        f"--env.hard_reset={'true' if libero.hard_reset else 'false'}",
        "--env.max_parallel_tasks=1",
    ]


def libero_success_by_seed(
    info: dict[str, Any], libero: LiberoEnvConfig, task_ids: list[int] | None = None
) -> dict[str, bool]:
    """Per-episode success of a LIBERO lerobot-eval run, keyed by the Kortyn seed of the same episode.

    lerobot-eval runs episode ``j`` of every task from init state ``j`` with seed ``eval_seed + j``; Kortyn's
    run episode ``i`` plays task ``libero.task_ids[i // episodes_per_task]``, episode
    ``i % episodes_per_task``. ``task_ids``: the evaluated subset (default: all tasks of the config).
    """
    by_task: dict[int, list[bool]] = {}
    for entry in info.get("per_task", []):
        task_id = int(entry["task_id"])
        by_task[task_id] = [bool(s) for s in entry["metrics"]["successes"]]
    evaluated = set(libero.task_ids if task_ids is None else task_ids)
    result: dict[str, bool] = {}
    for position, task_id in enumerate(libero.task_ids):
        if task_id not in evaluated:
            continue
        successes = by_task.get(task_id)
        if successes is None or len(successes) != libero.episodes_per_task:
            found = None if successes is None else len(successes)
            raise ConfigError(
                f"lerobot-eval result has {found} episodes for task {task_id}, "
                f"expected {libero.episodes_per_task}"
            )
        for within, success in enumerate(successes):
            result[str(libero.seed_base + position * libero.episodes_per_task + within)] = success
    return result


if __name__ == "__main__":
    raise SystemExit(main())
