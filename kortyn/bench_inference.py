"""Real inference time of the Brain's policy on this machine (phase 3a): the CPU numbers the virtual-time
experiments replace by published GPU times.

    python -m kortyn.bench_inference --config kortyn/experiments/phase3a/smolvla_push.yaml --threads 1,4

Observations come from the experiment's own environment (resets with the evaluation seeds), requests take
the Brain's path (``PolicyRunner.infer``: preprocessing, model, postprocessing). Run it on an otherwise idle
machine; the result goes to ``results/phase3a/cpu_inference_<profile>.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from kortyn.config import ConfigError, load_config

LOG = logging.getLogger("kortyn.bench_inference")


def processor_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def measure(
    config_path: Path, profile_name: str | None, threads: int, repeats: int, seed_start: int
) -> dict[str, Any]:
    """Median/p90 of ``repeats`` inferences with ``threads`` torch threads (one warm-up first)."""
    os.environ["OMP_NUM_THREADS"] = str(threads)
    from kortyn.checkpoint import ensure_checkpoint
    from kortyn.policy_runner import PolicyRunner
    from kortyn.sim_env import make_env

    config = load_config(config_path)
    profile_name = profile_name or config.experiment.suites[0].profile
    runner = PolicyRunner(
        ensure_checkpoint(config.brain.checkpoint),
        config.profile(profile_name),
        device="cpu",
        torch_threads=threads,
    )
    env = make_env(config.env)
    task = getattr(env, "task_description", None) or None
    keys = [f.key for f in runner.features]
    observations = []
    for index in range(repeats + 1):
        first = env.reset(seed=seed_start + index)
        observations.append({key: np.stack([first[key]] * runner.n_obs_steps) for key in keys})
    env.close()
    runner.infer(observations[0], seed=0, max_actions=runner.max_actions, task=task)  # warm-up
    timings: list[float] = []
    for index, observation in enumerate(observations[1:], start=1):
        started = time.perf_counter()
        runner.infer(observation, seed=index, max_actions=runner.max_actions, task=task)
        timings.append(1000.0 * (time.perf_counter() - started))
    return {
        "threads": threads,
        "repeats": repeats,
        "median_ms": statistics.median(timings),
        "p90_ms": float(np.percentile(timings, 90)),
        "min_ms": min(timings),
        "timings_ms": timings,
        "policy": runner.info.policy_type,
        "sampler": runner.info.sampler,
        "profile": profile_name,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure the real CPU inference time of the Brain's policy.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--threads", default="1,4")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--out", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout
    )
    import subprocess

    results: list[dict[str, Any]] = []
    try:
        thread_counts = [int(t) for t in args.threads.split(",") if t]
        if args.repeats < 1 or not thread_counts:
            raise ValueError("need --repeats >= 1 and at least one thread count")
        for count in thread_counts:
            # one process per thread count: torch fixes its thread pools at the first parallel region
            command = [
                sys.executable, "-c",
                "import json,sys; from pathlib import Path; from kortyn.bench_inference import measure; "
                f"print(json.dumps(measure(Path({args.config!r}), {args.profile!r}, {count}, {args.repeats}, "
                f"{args.seed_start})))",
            ]  # fmt: skip
            output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
            result = json.loads(output.strip().splitlines()[-1])
            LOG.info("%d threads: median %.0f ms, p90 %.0f ms", count, result["median_ms"], result["p90_ms"])
            results.append(result)
    except (ValueError, ConfigError, subprocess.CalledProcessError) as exc:
        LOG.error("%s", exc)
        return 1
    profile = results[0]["profile"]
    out = Path(args.out or f"results/phase3a/cpu_inference_{profile}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "config": args.config,
                "processor": processor_name(),
                "cpu_count": os.cpu_count(),
                "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    LOG.info("written %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
