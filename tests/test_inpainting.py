"""Gradient-free inpainting of diffusion action chunks (tiny random policy, real LeRobot sampler parts)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kortyn.config import BrainProfile
from kortyn.inpainting import GuidanceError, blend_model_output, guided_conditional_sample
from kortyn.policy_runner import GuideSpec, ObservationError, PolicyRunner

DDIM = BrainProfile(noise_scheduler_type="DDIM", num_inference_steps=4)
DDPM = BrainProfile(noise_scheduler_type="DDPM", num_inference_steps=5)


@pytest.fixture(scope="module")
def runner(tiny_checkpoint):
    return PolicyRunner(tiny_checkpoint, DDIM, device="cpu", torch_threads=1)


@pytest.fixture(scope="module")
def ddpm_runner(tiny_checkpoint):
    return PolicyRunner(tiny_checkpoint, DDPM, device="cpu", torch_threads=1)


def history(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "observation.image": rng.integers(0, 256, size=(2, 32, 32, 3), dtype=np.uint8),
        "observation.state": rng.uniform(0, 512, size=(2, 2)).astype(np.float32),
    }


def guide_actions(count: int, seed: int = 3) -> np.ndarray:
    """Actions inside the tiny checkpoint's action range [10, 500] x [20, 480]."""
    rng = np.random.default_rng(seed)
    return np.stack([rng.uniform(10, 500, count), rng.uniform(20, 480, count)], axis=1).astype(np.float32)


def test_zero_weights_reproduce_lerobots_sampler_bit_for_bit(runner):
    observation = history()
    plain = runner.infer(observation, seed=11, max_actions=15)
    guide = GuideSpec(offset=-1, actions=guide_actions(8), weights=np.zeros(8, dtype=np.float32))
    guided = runner.infer(observation, seed=11, max_actions=15, guide=guide)
    np.testing.assert_array_equal(guided.actions, plain.actions)


@pytest.mark.parametrize("which", ["ddim", "ddpm"])
def test_frozen_actions_are_reproduced_and_the_rest_adapts(runner, ddpm_runner, which):
    policy = runner if which == "ddim" else ddpm_runner
    observation = history(1)
    actions = guide_actions(8)
    # offset -1: actions[0] is the previous tick (not part of the returned chunk), actions[1:5] are frozen
    weights = np.array([1, 1, 1, 1, 1, 0.75, 0.5, 0.25], dtype=np.float32)
    guide = GuideSpec(offset=-1, actions=actions, weights=weights)
    guided = policy.infer(observation, seed=5, max_actions=15, guide=guide).actions
    plain = policy.infer(observation, seed=5, max_actions=15).actions
    np.testing.assert_allclose(guided[:4], actions[1:5], atol=1e-2)  # env units, range ~500
    assert not np.allclose(guided[4:], plain[4:], atol=1e-3)  # the free part followed the constraint
    assert guided.shape == plain.shape == (15, 2)


def test_a_fully_weighted_guide_is_the_chunk(runner):
    actions = guide_actions(16)
    guide = GuideSpec(offset=-1, actions=actions, weights=np.ones(16, dtype=np.float32))
    result = runner.infer(history(2), seed=9, max_actions=15, guide=guide).actions
    np.testing.assert_allclose(result, actions[1:16], atol=1e-2)


def test_entries_beyond_the_horizon_are_ignored(runner):
    actions = guide_actions(30)
    guide = GuideSpec(offset=0, actions=actions, weights=np.ones(30, dtype=np.float32))
    result = runner.infer(history(3), seed=2, max_actions=15, guide=guide).actions
    np.testing.assert_allclose(result, actions[:15], atol=1e-2)


@pytest.mark.parametrize(
    ("guide", "message"),
    [
        (GuideSpec(-1, np.zeros((3, 3), np.float32), np.ones(3, np.float32)), "must be"),
        (GuideSpec(-1, np.zeros((3, 2), np.float32), np.ones(2, np.float32)), "one weight per action"),
        (GuideSpec(-1, np.zeros((0, 2), np.float32), np.ones(0, np.float32)), "one weight per action"),
        (GuideSpec(-1, np.zeros((2, 2), np.float32), np.array([1.0, 1.5], np.float32)), r"\[0, 1\]"),
        (GuideSpec(-1, np.full((2, 2), np.nan, np.float32), np.ones(2, np.float32)), "non-finite"),
        (GuideSpec(-2, np.zeros((2, 2), np.float32), np.ones(2, np.float32)), "outside the chunk"),
        (GuideSpec(15, np.zeros((2, 2), np.float32), np.ones(2, np.float32)), "outside the chunk"),
    ],
)
def test_malformed_guides_are_rejected(runner, guide, message):
    with pytest.raises(ObservationError, match=message):
        runner.infer(history(), seed=1, max_actions=15, guide=guide)


def test_runner_advertises_guidance(runner):
    assert runner.supports_guidance
    assert runner.info.supports_guidance


def test_blend_reproduces_known_x0_through_the_scheduler_update():
    from diffusers import DDIMScheduler

    scheduler = DDIMScheduler(
        num_train_timesteps=100, clip_sample=True, clip_sample_range=1.0, prediction_type="epsilon"
    )
    scheduler.set_timesteps(4)
    torch.manual_seed(0)
    sample = torch.randn(1, 6, 2)
    model_output = torch.randn(1, 6, 2)
    known = torch.rand(1, 6, 2) * 2 - 1
    weights = torch.tensor([1.0, 1.0, 0.5, 0.0, 0.0, 0.0]).reshape(1, 6, 1)
    t = scheduler.timesteps[1]
    blended = blend_model_output(scheduler, model_output, sample, t, known, weights)
    torch.testing.assert_close(blended[:, 3:], model_output[:, 3:], rtol=0, atol=0)
    x0 = scheduler.step(blended, t, sample).pred_original_sample
    torch.testing.assert_close(x0[:, :2], known[:, :2], rtol=0, atol=1e-5)
    plain_x0 = scheduler.step(model_output, t, sample).pred_original_sample
    torch.testing.assert_close(x0[:, 2], 0.5 * known[:, 2] + 0.5 * plain_x0[:, 2], rtol=0, atol=1e-5)


def test_unsupported_prediction_types_and_shapes_are_reported():
    from diffusers import DDIMScheduler

    scheduler = DDIMScheduler(num_train_timesteps=100, prediction_type="v_prediction")
    scheduler.set_timesteps(4)
    tensor = torch.zeros(1, 4, 2)
    with pytest.raises(GuidanceError, match="v_prediction"):
        blend_model_output(scheduler, tensor, tensor, scheduler.timesteps[0], tensor, torch.ones(1, 4, 1))

    class Model:
        noise_scheduler = scheduler
        num_inference_steps = 4

    with pytest.raises(GuidanceError, match="does not match"):
        guided_conditional_sample(Model(), tensor, tensor, torch.zeros(1, 5, 2), torch.zeros(1, 4, 1))
    with pytest.raises(GuidanceError, match="weights must have shape"):
        guided_conditional_sample(Model(), tensor, tensor, tensor, torch.zeros(1, 4, 2))
