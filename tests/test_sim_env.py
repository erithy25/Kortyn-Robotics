import numpy as np
import pytest

from kortyn.config import EnvConfig
from kortyn.sim_env import EnvError, SimEnv


@pytest.fixture(scope="module")
def env():
    environment = SimEnv(EnvConfig())
    yield environment
    environment.close()


def test_defaults_match_lerobot_pusht_env_config():
    from lerobot.envs.factory import make_env_config

    lerobot_cfg = make_env_config("pusht")
    ours = EnvConfig()
    assert ours.gym_id == lerobot_cfg.gym_id
    assert ours.package == lerobot_cfg.package_name
    assert ours.fps == lerobot_cfg.fps
    assert ours.max_episode_steps == lerobot_cfg.episode_length
    for key in ("obs_type", "render_mode"):
        assert ours.gym_kwargs[key] == lerobot_cfg.gym_kwargs[key]
    expected_map = {raw: lerobot_cfg.features_map[raw] for raw in ("pixels", "agent_pos")}
    assert ours.features_map == expected_map


def test_reset_is_deterministic_per_seed_and_uses_lerobot_keys(env):
    first = env.reset(seed=1000)
    second = env.reset(seed=1000)
    other = env.reset(seed=1001)
    assert set(first) == {"observation.image", "observation.state"}
    assert first["observation.image"].shape == (96, 96, 3)
    assert first["observation.image"].dtype == np.uint8
    assert first["observation.state"].dtype == np.float32
    np.testing.assert_array_equal(first["observation.image"], second["observation.image"])
    np.testing.assert_array_equal(first["observation.state"], second["observation.state"])
    assert not np.array_equal(first["observation.image"], other["observation.image"])


def test_step_reports_reward_coverage_and_holds_position(env):
    observation = env.reset(seed=1000)
    hold = env.hold_action(observation)
    np.testing.assert_array_equal(hold, observation["observation.state"])
    result = env.step(hold)
    assert isinstance(result.reward, float)
    assert result.coverage is not None
    assert not result.success
    # holding the current position target keeps the agent (almost) still
    assert np.linalg.norm(result.observation["observation.state"] - observation["observation.state"]) < 1.0


def test_episode_is_truncated_at_max_steps():
    environment = SimEnv(EnvConfig(max_episode_steps=3))
    observation = environment.reset(seed=1)
    results = [environment.step(environment.hold_action(observation)) for _ in range(3)]
    environment.close()
    assert [r.truncated for r in results] == [False, False, True]


def test_unknown_env_package_is_reported():
    with pytest.raises(EnvError, match="cannot import env package"):
        SimEnv(EnvConfig(package="no_such_env_package"))
