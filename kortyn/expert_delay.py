"""Delay tolerance of Meta-World tasks, measured with Meta-World's scripted expert policies (phase 3a).

How "dynamic" a task is does not depend on the learned policy: it is how fast the task state runs away
while the controller acts on outdated information. The scripted expert of every Meta-World task is a
closed-loop controller; here it acts on the observation from ``d`` control ticks ago (pure observation
delay). Quasi-static tasks tolerate the delay (the expert just takes longer); in dynamic tasks the object
moves on, the expert overshoots, and success collapses at small delays. The result is the basis for the
task selection of the SmolVLA experiments.

    python -m kortyn.expert_delay --tasks push-v3,coffee-push-v3 --delays 0,4,8,16,24,40 --seeds 30

Environments are built like LeRobot's ``MetaworldEnv`` (seeded object placement per episode); output:
``results/phase3a/expert_delay.json`` (success rate per task and delay).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

LOG = logging.getLogger("kortyn.expert_delay")

DEFAULT_TASKS = (
    "drawer-close-v3",
    "plate-slide-v3",
    "push-v3",
    "door-open-v3",
    "hammer-v3",
    "stick-push-v3",
    "pick-place-v3",
    "coffee-push-v3",
    "soccer-v3",
    "sweep-into-v3",
    "basketball-v3",
    "button-press-v3",
)
DEFAULT_DELAYS = (0, 4, 8, 16, 24, 40)  # control ticks at 80 Hz: 0, 50, 100, 200, 300, 500 ms
FPS = 80.0
MAX_STEPS = 500


def expert_success(task: str, delay_ticks: int, seeds: list[int]) -> float:
    """Success rate of the task's scripted expert acting on observations ``delay_ticks`` old."""
    from lerobot.envs.metaworld import TASK_POLICY_MAPPING, MetaworldEnv

    if delay_ticks < 0:
        raise ValueError(f"delay must be >= 0, got {delay_ticks}")
    wrapper = MetaworldEnv(task=task, obs_type="pixels_agent_pos")
    successes = 0
    try:
        for seed in seeds:
            wrapper.reset(seed=seed)  # LeRobot's reset: seeds the object placement
            env: Any = wrapper._env
            expert = TASK_POLICY_MAPPING[task]()
            observation = env._get_obs()
            past: deque[Any] = deque([observation] * (delay_ticks + 1), maxlen=delay_ticks + 1)
            for _ in range(MAX_STEPS):
                action = np.clip(expert.get_action(past[0]), -1.0, 1.0).astype(np.float32)
                observation, _reward, _done, truncated, info = env.step(action)
                past.append(observation)
                if info.get("success"):
                    successes += 1
                    break
                if truncated:
                    break
    finally:
        wrapper.close()
    return successes / len(seeds)


def measure(tasks: list[str], delays: list[int], seeds: list[int]) -> dict[str, Any]:
    results: dict[str, dict[str, float]] = {}
    for task in tasks:
        results[task] = {str(delay): expert_success(task, delay, seeds) for delay in delays}
        LOG.info(
            "%-18s %s",
            task,
            "  ".join(f"{d * 1000 / FPS:.0f} ms: {results[task][str(d)]:.0%}" for d in delays),
        )
    return {
        "description": "Success rate of Meta-World's scripted expert acting on delayed observations",
        "fps": FPS,
        "delays_ticks": delays,
        "seeds": [seeds[0], seeds[-1]],
        "success": results,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Delay tolerance of Meta-World tasks (scripted experts).")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--delays", default=",".join(str(d) for d in DEFAULT_DELAYS), help="ticks at 80 Hz")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--out", default="results/phase3a/expert_delay.json")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout
    )
    from kortyn.metaworld_env import configure_rendering

    configure_rendering()
    try:
        delays = [int(d) for d in args.delays.split(",") if d]
        tasks = [t for t in args.tasks.split(",") if t]
        if args.seeds < 1 or not tasks or not delays:
            raise ValueError("need at least one task, one delay and one seed")
        result = measure(tasks, delays, list(range(args.seed_start, args.seed_start + args.seeds)))
    except (ValueError, KeyError) as exc:
        LOG.error("%s", exc)
        return 2
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    LOG.info("written %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
