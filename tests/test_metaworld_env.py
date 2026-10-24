"""Meta-World wrapper: identical to LeRobot's evaluation env, lazy rendering without side effects."""

from __future__ import annotations

import os

import numpy as np
import pytest

from kortyn.config import EnvConfig, MetaWorldEnvConfig
from kortyn.sim_env import EnvError, make_env

os.environ.setdefault("MUJOCO_GL", "egl")
metaworld_env = pytest.importorskip(
    "lerobot.envs.metaworld", reason="Meta-World (lerobot[metaworld]) missing"
)


def config(task: str = "push-v3", instruction: str | None = None) -> EnvConfig:
    return EnvConfig(
        kind="metaworld",
        fps=80.0,
        max_episode_steps=500,
        features_map={"pixels": "observation.images.camera1", "agent_pos": "observation.state"},
        hold_action_key=None,
        metaworld=MetaWorldEnvConfig(task=task, instruction=instruction),
    )


@pytest.fixture(scope="module")
def env():
    created = make_env(config())
    yield created
    created.close()


def random_actions(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-1.0, 1.0, size=(n, 4)).astype(np.float32)


def test_matches_lerobots_metaworld_env_step_by_step(env):
    reference = metaworld_env.MetaworldEnv(task="push-v3", obs_type="pixels_agent_pos")
    try:
        ours = env.reset(seed=1003)
        theirs, _ = reference.reset(seed=1003)
        np.testing.assert_array_equal(ours["observation.images.camera1"], theirs["pixels"])
        np.testing.assert_array_equal(ours["observation.state"], theirs["agent_pos"].astype(np.float32))
        for k, action in enumerate(random_actions(25)):
            result = env.step(action)
            observation, reward, _terminated, _truncated, info = reference.step(action)
            assert result.reward == pytest.approx(reward, abs=0.0)
            assert result.success == bool(info["is_success"])
            np.testing.assert_array_equal(
                result.observation["observation.state"], observation["agent_pos"].astype(np.float32)
            )
            if k in (0, 12, 24):  # rendering on demand gives LeRobot's (eagerly rendered) image
                np.testing.assert_array_equal(
                    result.observation["observation.images.camera1"], observation["pixels"]
                )
    finally:
        reference.close()


def test_rendering_does_not_change_the_trajectory(env):
    actions = random_actions(40, seed=3)

    def run(render_every: int) -> list[np.ndarray]:
        observation = env.reset(seed=1010)
        states = []
        for k, action in enumerate(actions):
            if k % render_every == 0:
                assert observation["observation.images.camera1"].shape == (480, 480, 3)
            observation = env.step(action).observation
            states.append(observation["observation.state"])
        return states

    renders_before = env.renders
    eager = run(render_every=1)
    assert env.renders - renders_before == len(actions) - 1  # the reset image comes with the reset
    lazy = run(render_every=13)
    np.testing.assert_array_equal(np.asarray(eager), np.asarray(lazy))


def test_old_observations_cannot_be_rendered_and_hold_is_zero(env):
    first = env.reset(seed=1)
    assert "observation.images.camera1" in first
    second = env.step(np.zeros(4, dtype=np.float32)).observation
    _ = first["observation.images.camera1"]  # rendered at reset: still available
    env.step(np.zeros(4, dtype=np.float32))
    with pytest.raises(EnvError, match="newest"):
        _ = second["observation.images.camera1"]
    np.testing.assert_array_equal(env.hold_action(first), np.zeros(4, dtype=np.float32))
    assert env.fps == 80.0
    assert env.action_dim == 4


def test_task_description_and_instruction_override():
    default = make_env(config("button-press-v3"))
    custom = make_env(config("button-press-v3", instruction="press it"))
    try:
        assert default.task_description == metaworld_env.TASK_DESCRIPTIONS["button-press-v3"]
        assert custom.task_description == "press it"
    finally:
        default.close()
        custom.close()
    with pytest.raises(EnvError, match="unknown Meta-World task"):
        make_env(config("no-such-task-v3"))
