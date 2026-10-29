"""LIBERO wrapper (phase 3): equal to LeRobot's evaluation env, pure helpers without the simulator."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from kortyn.config import ConfigError, EnvConfig, LiberoEnvConfig
from kortyn.libero_env import (
    ASSET_SUBDIRS,
    LIBERO_10_TASKS,
    ensure_libero_assets,
    ensure_libero_config,
    episode_slot,
    libero_default_paths,
    task_language,
)
from kortyn.sim_env import EnvError

HAS_LIBERO = importlib.util.find_spec("libero") is not None
needs_libero = pytest.mark.skipif(
    not HAS_LIBERO, reason="LIBERO is installed in the 'libero' environment only"
)


def libero_env_config(**libero: Any) -> EnvConfig:
    settings = {"task_ids": [0], "episodes_per_task": 50, **libero}
    return EnvConfig(
        kind="libero",
        fps=20.0,
        max_episode_steps=520,
        features_map={},
        hold_action_key=None,
        libero=LiberoEnvConfig(**settings),
    )


# --------------------------------------------------------------------------------------------- pure helpers
def test_episode_slot_matches_lerobot_episode_order() -> None:
    settings = LiberoEnvConfig(task_ids=[3, 7], episodes_per_task=50)
    assert episode_slot(settings, 1000) == (3, 0, 1000)
    assert episode_slot(settings, 1049) == (3, 49, 1049)
    assert episode_slot(settings, 1050) == (7, 0, 1000)  # episode 0 of the second task: init state 0
    assert episode_slot(settings, 1099) == (7, 49, 1049)
    # seeds outside the run wrap around (the agent's compatibility check resets with seed 0)
    assert episode_slot(settings, 1100) == episode_slot(settings, 1000)
    assert episode_slot(settings, 0) == episode_slot(settings, 1000 + (0 - 1000) % 100)


def test_episode_slot_with_offset_and_pilot_layout() -> None:
    settings = LiberoEnvConfig(task_ids=list(range(10)), episodes_per_task=10, init_state_offset=5)
    assert episode_slot(settings, 1000) == (0, 5, 1000)
    assert episode_slot(settings, 1013) == (1, 8, 1003)
    assert settings.episodes == 100


@pytest.mark.parametrize(
    "kwargs",
    [
        {"task_ids": []},
        {"task_ids": [1, 1]},
        {"task_ids": [-1]},
        {"episodes_per_task": 0},
        {"observation_height": 4},
        {"num_steps_wait": -1},
        {"cameras": {}},
        {"cameras": {"a": "x", "b": "x"}},
        {"state_key": "observation.images.camera1"},
    ],
)
def test_libero_config_rejects_invalid_values(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        LiberoEnvConfig(**kwargs)


def test_env_config_needs_libero_section() -> None:
    with pytest.raises(ConfigError, match="libero"):
        EnvConfig(kind="libero", features_map={}, hold_action_key=None)


def test_default_paths_are_libero_defaults(tmp_path: Path) -> None:
    paths = libero_default_paths(tmp_path)
    assert paths["benchmark_root"] == str(tmp_path)
    assert paths["bddl_files"].endswith("/./bddl_files")
    assert paths["init_states"].endswith("/./init_files")
    assert paths["datasets"].endswith("/../datasets")
    assert paths["assets"].endswith("/./assets")


def test_task_language_built_in_for_libero_long() -> None:
    assert task_language("libero_10", 8) == "put both moka pots on the stove"
    assert task_language("libero_10", 0) == LIBERO_10_TASKS[0]
    assert task_language("libero_10", 10) == ""  # no such task: the report shows the id only
    assert task_language("no_such_suite", 0) == ""


@needs_libero
def test_built_in_task_names_match_libero() -> None:
    from lerobot.envs.libero import _get_suite

    suite = _get_suite("libero_10")
    assert tuple(suite.get_task(i).language for i in range(len(LIBERO_10_TASKS))) == LIBERO_10_TASKS


@needs_libero
def test_ensure_config_writes_once(tmp_path: Path) -> None:
    config_file = ensure_libero_config(tmp_path / "cfg")
    written = yaml.safe_load(config_file.read_text())
    assert set(written) == {"benchmark_root", "bddl_files", "init_states", "datasets", "assets"}
    config_file.write_text("custom: 1\n")
    assert ensure_libero_config(tmp_path / "cfg") == config_file
    assert config_file.read_text() == "custom: 1\n"  # an existing config is never overwritten


@needs_libero
def test_ensure_assets_downloads_pinned_revision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kortyn.libero_env.libero_assets_dir", lambda: tmp_path / "assets")
    calls: list[dict[str, Any]] = []

    def fake_download(repo_id: str, **kwargs: Any) -> str:
        calls.append({"repo_id": repo_id, **kwargs})
        for subdir in ASSET_SUBDIRS:
            (Path(kwargs["local_dir"]) / subdir).mkdir(parents=True, exist_ok=True)
        return str(kwargs["local_dir"])

    target = ensure_libero_assets("abc123", download=fake_download)
    assert target == tmp_path / "assets"
    assert calls == [
        {
            "repo_id": "lerobot/libero-assets",
            "repo_type": "dataset",
            "revision": "abc123",
            "local_dir": str(target),
        }
    ]
    ensure_libero_assets("abc123", download=fake_download)
    assert len(calls) == 1  # present: no second download


@needs_libero
def test_ensure_assets_fails_on_incomplete_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kortyn.libero_env.libero_assets_dir", lambda: tmp_path / "assets")
    with pytest.raises(EnvError, match="lack"):
        ensure_libero_assets("abc123", download=lambda *args, **kwargs: None)


# --------------------------------------------------------------------------------------------- simulator
@pytest.fixture(scope="module")
def libero_pair() -> Any:
    """Our env and LeRobot's vector env (as ``lerobot-eval`` builds it) for task 0 of LIBERO-10."""
    if not HAS_LIBERO:
        pytest.skip("LIBERO is installed in the 'libero' environment only")
    from kortyn.libero_env import LiberoEnv, prepare_libero

    config = libero_env_config()
    assert config.libero is not None
    prepare_libero(config.libero)
    import gymnasium as gym
    from lerobot.envs.libero import create_libero_envs

    envs = create_libero_envs(
        task="libero_10",
        n_envs=1,
        gym_kwargs={
            "obs_type": "pixels_agent_pos",
            "render_mode": "rgb_array",
            "observation_height": 360,
            "observation_width": 360,
            "control_freq": 20,
            "hard_reset": True,
            "task_ids": [0],
        },
        env_cls=gym.vector.SyncVectorEnv,
    )
    reference = envs["libero_10"][0]
    ours = LiberoEnv(config)
    yield ours, reference
    ours.close()
    reference.close()


def _lerobot_policy_inputs(vector_obs: dict[str, Any]) -> dict[str, Any]:
    from lerobot.envs.utils import preprocess_observation
    from lerobot.processor.env_processor import LiberoProcessorStep

    return LiberoProcessorStep()._process_observation(preprocess_observation(vector_obs))


@needs_libero
@pytest.mark.timeout(600)
def test_matches_lerobot_libero_env_step_by_step(libero_pair: Any) -> None:
    """Same actions -> identical policy inputs (state, both cameras), rewards and success as LeRobot's env."""
    ours, reference = libero_pair
    vector_obs, _info = reference.reset(seed=[1000])
    observation = ours.reset(seed=1000)
    assert ours.task_description == reference.call("task_description")[0]
    rng = np.random.default_rng(0)
    for step in range(30):
        expected = _lerobot_policy_inputs(vector_obs)
        np.testing.assert_array_equal(
            observation["observation.state"], expected["observation.state"][0].numpy()
        )
        if step in (0, 7, 29):  # images: rendered lazily by us, every step by LeRobot's env
            for ours_key, lerobot_key in (
                ("observation.images.camera1", "observation.images.image"),
                ("observation.images.camera2", "observation.images.image2"),
            ):
                image = observation[ours_key]
                assert image.dtype == np.uint8
                assert image.shape == (360, 360, 3)
                scaled = expected[lerobot_key][0].permute(1, 2, 0) * 255
                expected_image = scaled.round().to(dtype=torch_uint8())
                np.testing.assert_array_equal(image, expected_image.numpy())
        action = rng.uniform(-1.0, 1.0, size=7).astype(np.float32)
        vector_obs, reward, terminated, _truncated, info = reference.step(action[None])
        result = ours.step(action)
        observation = result.observation
        assert result.reward == float(reward[0])
        assert result.success == bool(info["is_success"][0])
        assert result.terminated == bool(terminated[0])


def torch_uint8() -> Any:
    import torch

    return torch.uint8


@needs_libero
def test_lazy_image_of_old_step_is_refused(libero_pair: Any) -> None:
    ours, _reference = libero_pair
    first = ours.reset(seed=1001)
    assert ours.init_state_id == 1
    ours.step(ours.hold_action(first))
    with pytest.raises(EnvError, match="newest"):
        _ = first["observation.images.camera1"]


@needs_libero
def test_hold_action_keeps_gripper_command(libero_pair: Any) -> None:
    ours, _reference = libero_pair
    observation = ours.reset(seed=1002)
    hold = ours.hold_action(observation)
    np.testing.assert_array_equal(hold, [0, 0, 0, 0, 0, 0, -1])  # LeRobot's settling no-op
    close = np.zeros(7, dtype=np.float32)
    close[-1] = 1.0
    ours.step(close)
    np.testing.assert_array_equal(ours.hold_action(observation), [0, 0, 0, 0, 0, 0, 1])
    with pytest.raises(EnvError, match="shape"):
        ours.step(np.zeros(6, dtype=np.float32))


@needs_libero
def test_rejects_wrong_rate_and_episode_length() -> None:
    from kortyn.libero_env import LiberoEnv

    config = libero_env_config()
    config.max_episode_steps = 300
    with pytest.raises(EnvError, match="520"):
        LiberoEnv(config)
    config = libero_env_config()
    config.fps = 10.0
    with pytest.raises(EnvError, match="20 Hz"):
        LiberoEnv(config)
    with pytest.raises(EnvError, match="out of range"):
        LiberoEnv(libero_env_config(task_ids=[10]))


@needs_libero
@pytest.mark.timeout(600)
def test_several_envs_in_one_process_render_their_own_scene() -> None:
    """Fleet validation: each robot's camera shows its own simulator, whichever GL context is current."""
    from kortyn.libero_env import LiberoEnv

    config = libero_env_config()
    first, second = LiberoEnv(config), LiberoEnv(config)
    try:
        observation_b = second.reset(seed=1001)  # not rendered yet
        image_a = first.reset(seed=1000)["observation.images.camera1"]  # first's GL context is now current
        image_b = observation_b["observation.images.camera1"]  # must make second's context current itself
        second.close()
        image_b_alone = first.reset(seed=1001)["observation.images.camera1"]
        np.testing.assert_array_equal(image_b, image_b_alone)
        image_a_alone = first.reset(seed=1000)["observation.images.camera1"]
        np.testing.assert_array_equal(image_a, image_a_alone)
        assert not np.array_equal(image_a, image_b)  # init states 0 and 1 differ
    finally:
        first.close()


CAMERAS = ("observation.images.camera1", "observation.images.camera2")


def _camera_frames(observation: Any) -> list[np.ndarray]:
    return [np.array(observation[key]) for key in CAMERAS]


@needs_libero
@pytest.mark.timeout(900)
def test_reset_and_close_of_another_simulator_keep_this_robots_images() -> None:
    """Fleet validation: robosuite frees a destroyed simulator's GL resources in the current GL context.

    A hard reset (or close) of robot B's simulator right after robot A rendered must not delete A's textures
    and framebuffers: A's images stay those of A alone, step by step.
    """
    from kortyn.libero_env import LiberoEnv

    config = libero_env_config()
    actions = np.zeros((12, 7), dtype=np.float32)
    actions[:, 0] = 0.4  # move the end effector, so every frame differs from the last
    actions[:, -1] = -1.0

    def play(env: LiberoEnv, other: LiberoEnv | None) -> list[list[np.ndarray]]:
        frames = [_camera_frames(env.reset(seed=1000))]
        for t, action in enumerate(actions):
            if other is not None and t == 4:
                other.reset(seed=1002)  # destroys B's simulator while A's GL context is current
            if other is not None and t == 8:
                other.close()  # closes B's simulator while A's GL context is current
            frames.append(_camera_frames(env.step(action).observation))
        return frames

    alone = LiberoEnv(config)
    try:
        expected = play(alone, None)
    finally:
        alone.close()
    robot_a, robot_b = LiberoEnv(config), LiberoEnv(config)
    try:
        # B owns a simulator and rendered last, as after the agent's compatibility check
        _camera_frames(robot_b.reset(seed=1001))
        frames = play(robot_a, robot_b)
    finally:
        robot_a.close()
        robot_b.close()
    for t, (got, want) in enumerate(zip(frames, expected, strict=True)):
        for camera, (image, reference) in enumerate(zip(got, want, strict=True)):
            assert np.array_equal(image, reference), f"step {t}, camera {camera + 1}: differs from A alone"


@needs_libero
@pytest.mark.timeout(900)
def test_scene_depends_on_the_episode_seed_only() -> None:
    """LIBERO appends a property sampler per hard reset; the stove of task 8 must not move with history.

    Without the reset of the samplers, every hard reset of the same seed placed the stove (a fixture written
    into ``model.body_pos``) at a different position, so episodes depended on the simulator's history.
    """
    from kortyn.libero_env import LiberoEnv

    def stove_and_scene(env: LiberoEnv, seed: int) -> tuple[np.ndarray, np.ndarray]:
        env.reset(seed=seed)
        sim = env._sim_env.sim
        return np.array(sim.model.body_pos), np.array(sim.data.qpos)

    config = libero_env_config(task_ids=[8, 0])
    warm = LiberoEnv(config)
    try:
        first = stove_and_scene(warm, 1000)  # task 8, init state 0
        stove_and_scene(warm, 1001)  # another episode of the same simulator
        again = stove_and_scene(warm, 1000)
        task0 = stove_and_scene(warm, 1050)
    finally:
        warm.close()
    fresh = LiberoEnv(config)
    try:
        alone = stove_and_scene(fresh, 1000)
        task0_alone = stove_and_scene(fresh, 1050)
    finally:
        fresh.close()
    for expected, actual in ((first, again), (first, alone), (task0, task0_alone)):
        np.testing.assert_array_equal(expected[0], actual[0])
        np.testing.assert_array_equal(expected[1], actual[1])
