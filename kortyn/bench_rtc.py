"""Phase 4, step 2: extra compute per request with real-time chunking (RTC), on this machine's CPU.

Loads the LIBERO policy of the experiment twice in one process - once as in phase 3 (no RTC, the request
path under ``torch.inference_mode``) and once with LeRobot's RTC switched on - and times both on the same
real LIBERO observation, alternating plain and RTC requests after a warm-up (pre-registered in
reports/phase4_preregistration.md, section 4): the RTC request carries the rest of a previous plan (25
actions, the first ``delay`` of them frozen). The relative overhead ``r = median(RTC) / median(plain) - 1``
(with a bootstrap interval over the pairs) scales the GPU service time of the fleet model:
``s1 * (1 + max(r, 0))``.

    UV_PROJECT_ENVIRONMENT=.venv-libero uv run --no-group metaworld --group libero \\
        python -m kortyn.bench_rtc --config kortyn/experiments/phase3/libero_main.yaml \\
        --out results/phase4/rtc_overhead.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from kortyn.config import ConfigError, RtcSettings, load_config
from kortyn.typing_utils import FloatArray

LOG = logging.getLogger("kortyn.bench_rtc")

WARMUP = 5
PAIRS = 30
DELAY = 5  # frozen actions of the guide (250 ms at 20 Hz)
HORIZON = 25  # current action + 24 queued actions: the overlap at a plan every 25 ticks


def rtc_guide_weights(delay: int, horizon: int) -> FloatArray:
    """Frozen ``delay`` entries, then a decay over the rest of the overlap (the agent's ``exp`` schedule)."""
    from kortyn.freshness import overlap_weights

    if not 1 <= delay <= horizon:
        raise ConfigError(f"need 1 <= delay <= horizon, got {delay} and {horizon}")
    return overlap_weights(delay, horizon - delay, "exp")


def relative_overhead(
    plain_ms: Sequence[float], rtc_ms: Sequence[float], draws: int, seed: int
) -> tuple[float, tuple[float, float]]:
    """``median(rtc) / median(plain) - 1`` and its 95-% interval (bootstrap over the pairs)."""
    plain = np.asarray(plain_ms, dtype=float)
    rtc = np.asarray(rtc_ms, dtype=float)
    if plain.shape != rtc.shape or plain.size < 2 or np.any(plain <= 0) or np.any(rtc <= 0):
        raise ConfigError("need at least two positive timing pairs")
    point = float(np.median(rtc) / np.median(plain) - 1.0)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, plain.size, size=(draws, plain.size))
    ratios = np.median(rtc[picks], axis=1) / np.median(plain[picks], axis=1) - 1.0
    low, high = np.percentile(ratios, [2.5, 97.5])
    return point, (float(low), float(high))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase-4 RTC overhead per request on the CPU.")
    parser.add_argument("--config", required=True, help="LIBERO experiment (policy, cameras)")
    parser.add_argument("--profile", default="smolvla_libero")
    parser.add_argument("--out", default="results/phase4/rtc_overhead.json")
    parser.add_argument("--pairs", type=int, default=PAIRS)
    parser.add_argument("--seed", type=int, default=1000, help="episode whose first observation is used")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    from kortyn.libero_env import LiberoEnv
    from kortyn.policy_runner import GuideSpec, PolicyRunner
    from kortyn.run_virtual import _limit_threads
    from kortyn.sim_env import make_env

    try:
        if args.pairs < 2:
            raise ConfigError("--pairs must be >= 2")
        config = load_config(args.config)
        profile = config.profiles[args.profile]
        _limit_threads(1)
        checkpoint = Path(config.brain.checkpoint.local_dir)
        plain_runner = PolicyRunner(checkpoint, profile, device="cpu", torch_threads=1)
        rtc_profile = dataclasses.replace(profile, rtc=RtcSettings())
        rtc_runner = PolicyRunner(checkpoint, rtc_profile, device="cpu", torch_threads=1)
        env = make_env(config.env)
        if not isinstance(env, LiberoEnv):
            raise ConfigError("the RTC benchmark needs a LIBERO experiment (env.kind: libero)")
        lazy = env.reset(args.seed)
        observation: dict[str, Any] = {key: lazy[key][np.newaxis] for key in env.image_keys}
        state_keys = [f.key for f in plain_runner.features if f.kind != "VISUAL"]
        for key in state_keys:
            observation[key] = np.asarray(lazy[key], dtype=np.float32)[np.newaxis]
        task = env.task_description
        env.close()
        chunk = plain_runner.max_actions
        previous = plain_runner.infer(observation, seed=99, max_actions=chunk, task=task).actions
        guide = GuideSpec(
            offset=0,
            actions=previous[:HORIZON].astype(np.float32),
            weights=rtc_guide_weights(DELAY, HORIZON),
        )
        rtc_runner.check_guide(guide)
        for i in range(WARMUP):
            plain_runner.infer(observation, seed=i, max_actions=chunk, task=task)
            rtc_runner.infer(observation, seed=i, max_actions=chunk, task=task, guide=guide)
        plain_ms: list[float] = []
        rtc_ms: list[float] = []
        for i in range(args.pairs):
            seed = 1000 + i
            if i % 2 == 0:  # alternate the order within a pair: drift hits both paths alike
                plain_ms.append(plain_runner.infer(observation, seed, chunk, task=task).inference_ms)
                rtc_ms.append(rtc_runner.infer(observation, seed, chunk, guide=guide, task=task).inference_ms)
            else:
                rtc_ms.append(rtc_runner.infer(observation, seed, chunk, guide=guide, task=task).inference_ms)
                plain_ms.append(plain_runner.infer(observation, seed, chunk, task=task).inference_ms)
        overhead, interval = relative_overhead(plain_ms, rtc_ms, 10_000, 4242)
    except (ConfigError, KeyError, ValueError) as exc:
        LOG.error("%s", exc)
        return 2
    document = {
        "experiment": args.config,
        "profile": args.profile,
        "observation_seed": args.seed,
        "warmup": WARMUP,
        "pairs": args.pairs,
        "delay": DELAY,
        "horizon": HORIZON,
        "rtc": dataclasses.asdict(RtcSettings()),
        "plain_ms": plain_ms,
        "rtc_ms": rtc_ms,
        "plain_median_ms": statistics.median(plain_ms),
        "rtc_median_ms": statistics.median(rtc_ms),
        "relative_overhead": overhead,
        "relative_overhead_ci": list(interval),
        "applied_overhead": max(overhead, 0.0),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=1), encoding="utf-8")
    LOG.info(
        "plain %.0f ms, RTC %.0f ms (median): overhead %+.1f %% [%+.1f, %+.1f]; written %s",
        document["plain_median_ms"],
        document["rtc_median_ms"],
        100 * overhead,
        100 * interval[0],
        100 * interval[1],
        out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
