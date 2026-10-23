"""Reference measurement with LeRobot's own evaluation script on the migrated checkpoint.

Runs the official ``lerobot-eval`` (synchronous rollouts, vectorized envs, same seeds as Kortyn) with the
sampler of a Brain profile and stores the result next to the Kortyn results. Together with the
checkpoint's published numbers and Kortyn's ``referenz_sync`` run this separates

* checkpoint migration + sampler choice   (published vs. lerobot-eval),
* Kortyn's Brain/Agent pipeline            (lerobot-eval vs. referenz_sync),
* real-time latency and the network        (referenz_sync vs. the network scenarios).

    python -m kortyn.lerobot_reference --profile ddim10
    python -m kortyn.lerobot_reference --scheduler DDPM --num-inference-steps 100 --name ddpm100
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

from kortyn.config import DEFAULT_BASE_CONFIG, ConfigError, load_config

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
    episodes = args.episodes or config.experiment.episodes
    batch = args.batch_size or episodes

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
        f"--policy.noise_scheduler_type={scheduler}",
        f"--policy.num_inference_steps={steps}",
        "--env.type=pusht",
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
        with (out_dir / "lerobot_eval.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode != 0:
            LOG.error(
                "lerobot-eval failed with code %d, see %s", result.returncode, out_dir / "lerobot_eval.log"
            )
            return 1
        info = json.loads((work_dir / "eval_info.json").read_text(encoding="utf-8"))
        provenance = "run by kortyn.lerobot_reference"
    successes = [bool(s) for s in info["per_task"][0]["metrics"]["successes"]]
    seeds = range(config.experiment.seed_start, config.experiment.seed_start + len(successes))
    summary = {
        "tool": "lerobot-eval (LeRobot's official evaluation, synchronous)",
        "command": " ".join(command[1:]),
        "provenance": provenance,
        "checkpoint": f"{config.brain.checkpoint.repo_id}@{config.brain.checkpoint.revision} (migrated)",
        "checkpoint_sha256": digest.hexdigest(),
        "sampler": f"{scheduler}-{steps}",
        "device": args.device,
        "episodes": len(successes),
        "successes": sum(successes),
        "success_rate": sum(successes) / len(successes) if successes else None,
        "avg_max_reward": info.get("overall", {}).get("avg_max_reward"),
        "success_by_seed": {str(seed): ok for seed, ok in zip(seeds, successes, strict=True)},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    shutil.rmtree(work_dir / "videos", ignore_errors=True)  # large and not needed for the report
    LOG.info("%s: %d/%d successes", summary["sampler"], summary["successes"], summary["episodes"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
