"""Is the PushT simulation bitwise reproducible? (diagnostic of phase 3a)

The simulation-time runs are deterministic in everything the virtual clock and the latency model decide
(request times, tick series, outcomes); repeated executions still differed in the last bits of the PushT
coverage. This tool isolates the environment: one fixed, contact-rich action sequence is executed on the
same seed in several situations, and the coverage traces are compared bit for bit:

* the same env instance again,
* the same instance after other episodes,
* a second env instance in the same process.

The actions come from the PushT policy (as in the sync reference: a plan every ``sync_actions_per_chunk``
steps, noise seeded per plan), which pushes the T with sustained contact on both of its parts. Without the
checkpoint, ``--source pusher`` uses a scripted pusher that circles the block (fewer contacts). Run the tool
twice and compare the printed checksums to see differences between processes as well.

    python -m kortyn.env_determinism --seed 1000 --steps 300 [--source pusher]
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from kortyn.config import DEFAULT_BASE_CONFIG, KortynConfig, load_config
from kortyn.sim_env import SimEnv, make_env
from kortyn.typing_utils import FloatArray

LOG = logging.getLogger("kortyn.env_determinism")

Trace = npt.NDArray[np.float64]


def pusher_actions(env: SimEnv, seed: int, steps: int) -> FloatArray:
    """Actions of a scripted pusher for ``seed``: it circles the T block at a shrinking radius."""
    env.reset(seed=seed)
    inner: Any = env._env.unwrapped  # gym_pusht's PushTEnv: block and agent are pymunk bodies
    actions = np.empty((steps, 2), dtype=np.float32)
    for k in range(steps):
        block = inner.block.position
        angle = 2.0 * math.pi * k / 60.0
        radius = max(10.0, 90.0 - 0.25 * k)
        target = np.array([block.x + radius * math.cos(angle), block.y + radius * math.sin(angle)])
        actions[k] = np.clip(target, 0.0, 512.0).astype(np.float32)
        env.step(actions[k])
    return actions


def policy_actions(config: KortynConfig, env: SimEnv, seed: int, steps: int, profile: str) -> FloatArray:
    """Actions of the PushT policy on ``seed`` as in the sync reference: a new plan every
    ``agent.sync_actions_per_chunk`` steps from the last ``n_obs_steps`` observations, seeded noise."""
    from kortyn.policy_runner import PolicyRunner

    checkpoint = Path(config.brain.checkpoint.local_dir)
    if not (checkpoint / "model.safetensors").is_file():
        raise SystemExit(
            f"no PushT checkpoint in {checkpoint}: run an experiment first or use --source pusher"
        )
    runner = PolicyRunner(checkpoint, config.profile(profile), "cpu", torch_threads=1)
    every = config.agent.sync_actions_per_chunk
    n_obs = runner.info.n_obs_steps
    observation = env.reset(seed=seed)
    history = [observation] * n_obs
    actions = np.empty((steps, env.action_dim), dtype=np.float32)
    plan: FloatArray | None = None
    for k in range(steps):
        if k % every == 0:
            stacked = {key: np.stack([h[key] for h in history[-n_obs:]]) for key in observation}
            plan = runner.infer(stacked, seed=seed * 1000 + k, max_actions=every).actions
        assert plan is not None
        actions[k] = plan[k % every]
        history.append(env.step(actions[k]).observation)
    return actions


def coverage_trace(env: SimEnv, seed: int, actions: FloatArray) -> Trace:
    env.reset(seed=seed)
    trace: Trace = np.empty(len(actions), dtype=np.float64)
    for k, action in enumerate(actions):
        result = env.step(action)
        trace[k] = math.nan if result.coverage is None else result.coverage
    return trace


@dataclass(frozen=True)
class Comparison:
    situation: str
    identical: bool
    steps_differing: int
    max_abs_diff: float


def compare(reference: Trace, other: Trace, situation: str) -> Comparison:
    diff = np.abs(other - reference)
    return Comparison(
        situation, bool(np.array_equal(other, reference)), int(np.sum(diff > 0)), float(np.max(diff))
    )


def checksum(trace: Trace) -> str:
    return hashlib.sha256(np.ascontiguousarray(trace).tobytes()).hexdigest()[:16]


def run(
    config: KortynConfig, seed: int, steps: int, other_episodes: int, source: str, profile: str = "kalibriert"
) -> tuple[Trace, list[Comparison]]:
    """Coverage trace of the reference execution and its comparison with the other situations."""
    env = make_env(config.env)
    if not isinstance(env, SimEnv):
        raise SystemExit("env_determinism checks gym environments (PushT); Meta-World runs on MuJoCo")
    try:
        if source == "policy":
            actions = policy_actions(config, env, seed, steps, profile)
        else:
            actions = pusher_actions(env, seed, steps)
        reference = coverage_trace(env, seed, actions)
        results = [compare(reference, coverage_trace(env, seed, actions), "same env instance again")]
        for other in range(1, other_episodes + 1):
            coverage_trace(env, seed + other, actions)
        results.append(
            compare(reference, coverage_trace(env, seed, actions), f"after {other_episodes} other episodes")
        )
    finally:
        env.close()
    second = make_env(config.env)
    try:
        assert isinstance(second, SimEnv)
        results.append(compare(reference, coverage_trace(second, seed, actions), "second env instance"))
    finally:
        second.close()
    return reference, results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bitwise reproducibility of the PushT simulation.")
    parser.add_argument("--config", default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--other-episodes", type=int, default=3)
    parser.add_argument(
        "--source",
        choices=("policy", "pusher"),
        default="policy",
        help="where the action sequence comes from (policy: needs the migrated PushT checkpoint)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    if args.steps < 1 or args.other_episodes < 0:
        parser.error("--steps must be >= 1 and --other-episodes >= 0")
    config = load_config(args.config)
    reference, results = run(config, args.seed, args.steps, args.other_episodes, args.source)
    LOG.info(
        "seed %d, %d steps (%s), max coverage %r, checksum %s",
        args.seed,
        args.steps,
        args.source,
        float(np.max(reference)),
        checksum(reference),
    )
    for result in results:
        LOG.info(
            "  %-28s bitwise identical: %-5s steps differing: %3d  max |diff| %.3e",
            result.situation,
            result.identical,
            result.steps_differing,
            result.max_abs_diff,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
