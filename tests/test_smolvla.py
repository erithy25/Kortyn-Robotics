"""SmolVLA in the Kortyn Brain path: identical to LeRobot's evaluation path (needs the local checkpoint).

The checkpoint (lerobot/smolvla_metaworld, ~0.9 GB) is downloaded by the first SmolVLA experiment; without it
these tests are skipped.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from kortyn.config import BrainProfile, EnvConfig, MetaWorldEnvConfig

os.environ.setdefault("MUJOCO_GL", "egl")
CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints" / "smolvla_metaworld"
pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "model.safetensors").is_file(), reason="SmolVLA checkpoint not downloaded"
)


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
    image = first["observation.images.camera1"]
    state = first["observation.state"]
    task = env.task_description
    env.close()
    return image, state, task


@pytest.mark.timeout(600)
def test_runner_matches_lerobots_evaluation_path(runner, observation):
    import torch
    from lerobot.envs.utils import preprocess_observation

    image, state, task = observation
    ours = runner.infer(
        {"observation.images.camera1": image[None], "observation.state": state[None]},
        seed=7,
        max_actions=50,
        task=task,
    ).actions
    # LeRobot's eval rollout: preprocess_observation -> task -> policy preprocessor -> chunk -> postprocessor,
    # with the same initial flow-matching noise
    batch = preprocess_observation({"pixels": image, "agent_pos": state.astype(np.float64)})
    batch["task"] = [task]
    batch = runner.preprocessor(batch)
    noise = torch.randn((1, 50, 32), generator=torch.Generator().manual_seed(7), dtype=torch.float32)
    runner.policy.reset()
    with torch.inference_mode():
        chunk = runner.policy.predict_action_chunk(batch, noise=noise)
    expected = runner.postprocessor(chunk.reshape(50, -1)).to(torch.float32).numpy()
    assert ours.shape == (50, 4)
    np.testing.assert_array_equal(ours, expected)


@pytest.mark.timeout(600)
def test_smolvla_profile_task_and_determinism(runner, observation):
    from kortyn.policy_runner import ObservationError

    image, state, task = observation
    info = runner.info
    assert (info.policy_type, info.sampler, info.max_actions, info.n_obs_steps) == (
        "smolvla",
        "flow-10",
        50,
        1,
    )
    assert info.needs_task
    assert not info.supports_guidance
    assert runner.config.num_steps == 10
    history = {"observation.images.camera1": image[None], "observation.state": state[None]}
    with pytest.raises(ObservationError, match="language instruction"):
        runner.infer(history, seed=1, max_actions=50)
    first = runner.infer(history, seed=3, max_actions=25, task=task).actions
    again = runner.infer(history, seed=3, max_actions=25, task=task).actions
    other = runner.infer(history, seed=4, max_actions=25, task=task).actions
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)
    assert first.shape == (25, 4)
    assert np.all(np.isfinite(first))
