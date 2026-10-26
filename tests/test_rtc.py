"""Real-time chunking in the Kortyn Brain path (phase 4): LeRobot's RTC switched on for SmolVLA.

Uses the SmolVLA checkpoint of phase 3a (lerobot/smolvla_metaworld, ~0.9 GB); without it the policy tests are
skipped. The guide checks of the agents need no checkpoint.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from kortyn.agent import BrainSpec, IncompatibleBrainError, check_guidance, continues_plan, sends_guides
from kortyn.config import (
    BrainProfile,
    ConfigError,
    EnvConfig,
    MetaWorldEnvConfig,
    RtcSettings,
    SmoothingConfig,
)

os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")
CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints" / "smolvla_metaworld"
needs_checkpoint = pytest.mark.skipif(
    not (CHECKPOINT / "model.safetensors").is_file(), reason="SmolVLA checkpoint not downloaded"
)


def spec(policy_type: str, supports_guidance: bool) -> BrainSpec:
    return BrainSpec(
        n_obs_steps=1,
        max_actions=50,
        action_dim=4,
        features={},
        sampler="flow-10",
        device="cpu",
        policy_type=policy_type,
        warmup_inference_ms=0.0,
        supports_guidance=supports_guidance,
    )


def test_smoothing_modes_and_guidance_checks() -> None:
    assert SmoothingConfig(mode="rtc").mode == "rtc"
    assert sends_guides("rtc")
    assert sends_guides("inpaint")
    assert not sends_guides("none")
    check_guidance(spec("smolvla", True), "rtc", "s")
    check_guidance(spec("diffusion", True), "inpaint", "s")
    check_guidance(spec("smolvla", False), "none", "s")
    with pytest.raises(IncompatibleBrainError, match="does not support guides"):
        check_guidance(spec("smolvla", False), "rtc", "s")
    with pytest.raises(IncompatibleBrainError, match="real-time chunking"):
        check_guidance(spec("smolvla", True), "inpaint", "s")
    with pytest.raises(IncompatibleBrainError, match="inpainting"):
        check_guidance(spec("diffusion", True), "rtc", "s")
    with pytest.raises(ConfigError):
        RtcSettings(max_guidance_weight=0.0)


@pytest.fixture(scope="module")
def runner():
    from kortyn.policy_runner import PolicyRunner

    return PolicyRunner(CHECKPOINT, BrainProfile(num_inference_steps=10), device="cpu", torch_threads=4)


@pytest.fixture(scope="module")
def observation():
    from kortyn.metaworld_env import MetaWorldEnv

    env = MetaWorldEnv(
        EnvConfig(
            kind="metaworld",
            fps=80.0,
            max_episode_steps=500,
            features_map={"pixels": "observation.images.camera1", "agent_pos": "observation.state"},
            hold_action_key=None,
            metaworld=MetaWorldEnvConfig(task="push-v3"),
        )
    )
    first = env.reset(seed=1000)
    obs = {
        "observation.images.camera1": first["observation.images.camera1"][np.newaxis],
        "observation.state": np.asarray(first["observation.state"], dtype=np.float32)[np.newaxis],
    }
    task = env.task_description
    env.close()
    return obs, task


@needs_checkpoint
@pytest.mark.timeout(900)
def test_rtc_without_guide_is_bit_identical_and_with_guide_follows_the_old_plan(runner, observation):
    from kortyn.policy_runner import GuideSpec, ObservationError, _enable_rtc

    obs, task = observation
    assert runner.guidance_method == ""
    plain = runner.infer(obs, seed=7, max_actions=50, task=task).actions
    old_plan = runner.infer(obs, seed=99, max_actions=50, task=task).actions  # a different plan to follow

    settings = RtcSettings()
    _enable_rtc(runner.policy, settings, runner.max_actions)
    runner.rtc = settings
    assert runner.guidance_method == "rtc"
    assert runner.supports_guidance
    # (1) RTC on, no previous plan: exactly the unguided computation
    unguided = runner.infer(obs, seed=7, max_actions=50, task=task).actions
    assert np.array_equal(unguided, plain)

    # (2) with the rest of the old plan (25 actions, the first 5 executed before the new plan arrives)
    delay, horizon = 5, 25
    weights = np.concatenate([np.ones(delay), np.linspace(1.0, 0.0, horizon - delay + 2)[1:-1]]).astype(
        np.float32
    )
    guide = GuideSpec(offset=0, actions=old_plan[:horizon].astype(np.float32), weights=weights)
    runner.check_guide(guide)
    guided = runner.infer(obs, seed=7, max_actions=50, task=task, guide=guide).actions
    distance_guided = np.abs(guided[:delay] - old_plan[:delay]).mean()
    distance_plain = np.abs(plain[:delay] - old_plan[:delay]).mean()
    assert distance_guided < 0.5 * distance_plain
    assert np.all(np.isfinite(guided))
    # an entry before the observation step (the previous tick's action) is dropped, not misaligned
    with_previous = GuideSpec(
        offset=-1,
        actions=np.vstack([old_plan[:1], old_plan[:horizon]]).astype(np.float32),
        weights=np.concatenate([[1.0], weights]).astype(np.float32),
    )
    shifted = runner.infer(obs, seed=7, max_actions=50, task=task, guide=with_previous).actions
    assert np.array_equal(shifted, guided)

    # (3) invalid guides
    for bad in (
        GuideSpec(offset=1, actions=old_plan[:3].astype(np.float32), weights=np.ones(3, dtype=np.float32)),
        GuideSpec(offset=0, actions=old_plan[:3].astype(np.float32), weights=np.zeros(3, dtype=np.float32)),
        GuideSpec(
            offset=0, actions=np.zeros((51, 4), dtype=np.float32), weights=np.ones(51, dtype=np.float32)
        ),
    ):
        with pytest.raises(ObservationError):
            runner.check_guide(bad)


def test_overhead_benchmark_helpers() -> None:
    from kortyn.bench_rtc import relative_overhead, rtc_guide_weights

    weights = rtc_guide_weights(5, 25)
    assert weights.shape == (25,)
    assert np.all(weights[:5] == 1.0)
    assert np.all(weights[5:] < 1.0)
    assert np.all(np.diff(weights[5:]) < 0)
    with pytest.raises(ConfigError):
        rtc_guide_weights(0, 25)
    with pytest.raises(ConfigError):
        rtc_guide_weights(26, 25)
    plain = [100.0, 102.0, 98.0, 101.0, 99.0]
    overhead, (low, high) = relative_overhead(plain, [p * 1.1 for p in plain], 2000, 1)
    assert overhead == pytest.approx(0.1)
    assert low <= overhead <= high
    with pytest.raises(ConfigError):
        relative_overhead([100.0], [110.0], 100, 1)
    with pytest.raises(ConfigError):
        relative_overhead([100.0, 0.0], [110.0, 120.0], 100, 1)


def test_rtc_guides_only_continue_a_running_plan() -> None:
    action = np.zeros(7, dtype=np.float32)
    assert not continues_plan("rtc", [])  # first plan of a cold start: nothing to continue
    assert continues_plan("rtc", [action])
    assert continues_plan("inpaint", [])  # phase 2 unchanged: inpainting always sends its guide
