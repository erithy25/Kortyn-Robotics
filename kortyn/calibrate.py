"""Choose the Brain's diffusion step count for the measurement machine.

The phase-1 question is how much the *network* hurts. That is only measurable if the Brain itself is fast
enough that the ideal network ("gut") does not starve. This command runs a few "gut" episodes for each
candidate DDIM step count and applies the gate

* no starvation in "gut", and
* p99 end-to-end latency <= ``max_ticks`` control ticks (default 5 of the 7 that the refill threshold
  allows, i.e. 2 ticks of headroom for the network),

then recommends the largest (= closest to the checkpoint) step count that passes. Results go to
``results/calibration/``; put the recommendation into ``profiles.kalibriert`` of ``base.yaml``.

    python -m kortyn.calibrate --steps 10 8 6 5 4 --episodes 3
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

from kortyn.config import DEFAULT_BASE_CONFIG, BrainProfile, ConfigError, load_config, load_scenario
from kortyn.metrics import EpisodeRecord, RequestRecord, read_jsonl, summarize_run
from kortyn.run_experiment import EPISODES_FILE, REQUESTS_FILE, RunnerError, RunPlan, SuiteRunner

LOG = logging.getLogger("kortyn.calibrate")
PROFILE = "kalibriert"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate the DDIM step count for real-time operation.")
    parser.add_argument("--config", default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument("--steps", type=int, nargs="+", default=[10, 8, 6, 5, 4])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-ticks", type=float, default=5.0, help="p99 E2E latency budget in ticks")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    config_path = Path(args.config).resolve()
    try:
        config = load_config(config_path)
        gut = load_scenario("gut")
    except ConfigError as exc:
        LOG.error("%s", exc)
        return 2
    root = Path(args.results_dir or config.experiment.results_dir) / "calibration"
    dt_ms = 1000.0 / config.env.fps
    seeds = list(range(config.experiment.seed_start, config.experiment.seed_start + args.episodes))
    rows = []
    for steps in sorted(set(args.steps), reverse=True):
        config.profiles[PROFILE] = BrainProfile(noise_scheduler_type="DDIM", num_inference_steps=steps)
        overrides = ["--scheduler", "DDIM", "--num-inference-steps", str(steps)]
        results_dir = root / f"ddim{steps}"
        try:
            SuiteRunner(
                config, config_path, PROFILE, [RunPlan("gut", gut)], seeds, results_dir, overrides
            ).run()
        except RunnerError as exc:
            LOG.error("calibration for DDIM-%d failed: %s", steps, exc)
            return 1
        run_dir = results_dir / PROFILE / "gut"
        episodes = [EpisodeRecord.from_json(r) for r in read_jsonl(run_dir / EPISODES_FILE)]
        requests = [RequestRecord.from_json(r) for r in read_jsonl(run_dir / REQUESTS_FILE)]
        summary = summarize_run(episodes, requests)
        p99_ticks = (summary.e2e_p99_ms or math.inf) / dt_ms
        passed = summary.starved_ticks_per_min == 0 and p99_ticks <= args.max_ticks
        rows.append(
            {
                "num_inference_steps": steps,
                "server_inference_p50_ms": summary.server_inference_p50_ms,
                "e2e_p50_ms": summary.e2e_p50_ms,
                "e2e_p99_ms": summary.e2e_p99_ms,
                "e2e_p99_ticks": p99_ticks,
                "starved_ticks_per_min": summary.starved_ticks_per_min,
                "passed": passed,
            }
        )
        LOG.info(
            "DDIM-%-3d inference p50 %6.0f ms | E2E p50 %6.0f ms p99 %6.0f ms (%.1f ticks) "
            "| starved/min %.1f -> %s",
            steps,
            summary.server_inference_p50_ms or math.nan,
            summary.e2e_p50_ms or math.nan,
            summary.e2e_p99_ms or math.nan,
            p99_ticks,
            summary.starved_ticks_per_min,
            "PASS" if passed else "fail",
        )
    passing = [row["num_inference_steps"] for row in rows if row["passed"]]
    recommendation = max(passing) if passing else None
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(
        json.dumps(
            {"max_ticks": args.max_ticks, "rows": rows, "recommended_steps": recommendation}, indent=2
        ),
        encoding="utf-8",
    )
    if recommendation is None:
        LOG.warning("no step count passed the gate; the Brain hardware is too slow for this control rate")
        return 1
    LOG.info(
        "recommended: profiles.kalibriert.num_inference_steps = %d (written to %s)",
        recommendation,
        root / "summary.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
