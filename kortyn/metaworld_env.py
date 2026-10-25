"""Meta-World environment for the Kortyn Agent (phase 3a), built exactly like LeRobot's evaluation env.

LeRobot's ``MetaworldEnv`` (``lerobot/envs/metaworld.py``) sets the benchmark up: ``MT1(task, seed=42)``, the
first training task, the fixed pose of the ``corner2`` camera, seeded object placement per ``reset(seed)``.
This wrapper uses it for construction and resets and steps the underlying Meta-World env directly, so the
camera image is rendered only when the agent actually sends an observation: rendering a 480x480 frame in
software costs ~90 ms, a simulation step ~1 ms. Rendering reads the simulation state without changing it,
so lazy rendering yields exactly the images and trajectories of rendering every step (checked by tests).

Lazy images require that only the newest observation is ever sent (``n_obs_steps = 1``, as for SmolVLA):
asking for the image of an older observation raises :class:`~kortyn.sim_env.EnvError`.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

from kortyn.config import EnvConfig
from kortyn.sim_env import EnvError, StepResult
from kortyn.typing_utils import Array, FloatArray

RAW_IMAGE_KEY = "pixels"
RAW_STATE_KEY = "agent_pos"
STATE_DIM = 4  # end-effector xyz + gripper opening (LeRobot: raw_obs[:4])


def default_mujoco_gl() -> str:
    """Offscreen OpenGL backend for MuJoCo without a display: EGL on Linux, CGL on macOS."""
    return "cgl" if sys.platform == "darwin" else "egl"


def configure_rendering() -> None:
    """Select the headless rendering backend unless ``MUJOCO_GL`` is set (call before importing MuJoCo)."""
    os.environ.setdefault("MUJOCO_GL", default_mujoco_gl())


class LazyObservation(dict[str, Array]):
    """Observation whose camera image is rendered on first access, from the step it belongs to."""

    def __init__(self, env: MetaWorldEnv, values: dict[str, Array], image_key: str, step_index: int) -> None:
        super().__init__(values)
        self._env = env
        self._image_key = image_key
        self._step_index = step_index

    def __contains__(self, key: object) -> bool:
        return key == self._image_key or super().__contains__(key)

    def __missing__(self, key: str) -> Array:
        if key != self._image_key:
            raise KeyError(key)
        if self._env.step_index != self._step_index:
            raise EnvError(
                f"image of step {self._step_index} requested at step {self._env.step_index}: lazily rendered "
                "observations can only be sent while they are the newest one (n_obs_steps = 1)"
            )
        image = self._env.render_image()
        self[key] = image
        return image


class MetaWorldEnv:
    """One Meta-World v3 task with LeRobot-style observation keys and lazily rendered camera images."""

    def __init__(self, config: EnvConfig) -> None:
        if config.kind != "metaworld" or config.metaworld is None:
            raise EnvError("MetaWorldEnv needs env.kind 'metaworld' with an env.metaworld section")
        configure_rendering()
        try:
            from lerobot.envs.metaworld import TASK_DESCRIPTIONS, MetaworldEnv
        except ImportError as exc:
            raise EnvError(f"Meta-World is not installed (lerobot[metaworld]): {exc}") from exc
        self.config = config
        settings = config.metaworld
        task = settings.task
        if task not in TASK_DESCRIPTIONS:
            raise EnvError(f"unknown Meta-World task {task!r}")
        missing = {RAW_IMAGE_KEY, RAW_STATE_KEY} - set(config.features_map)
        if missing:
            raise EnvError(f"env.features_map must map {sorted(missing)} for Meta-World")
        self._image_key = config.features_map[RAW_IMAGE_KEY]
        self._state_key = config.features_map[RAW_STATE_KEY]
        self._lerobot = MetaworldEnv(task=task, camera_name=settings.camera_name, obs_type="pixels_agent_pos")
        self.task = task
        self.task_description = settings.instruction or TASK_DESCRIPTIONS[task]
        shape = self._lerobot.action_space.shape
        if shape is None or len(shape) != 1:
            raise EnvError(f"expected a flat Box action space, got {self._lerobot.action_space}")
        self.action_dim = int(shape[0])
        self.step_index = 0
        self.renders = 0

    @property
    def fps(self) -> float:
        return self.config.fps

    @property
    def max_episode_steps(self) -> int:
        return self.config.max_episode_steps

    @property
    def _sim(self) -> Any:
        sim = self._lerobot._env
        if sim is None:
            raise EnvError("Meta-World env used before reset")
        return sim

    def render_image(self) -> Array:
        """Camera image of the current simulation state, as LeRobot renders it (corner2 flipped)."""
        image = np.ascontiguousarray(self._lerobot.render(), dtype=np.uint8)
        self.renders += 1
        return image

    def _observation(self, raw_state: Array, image: Array | None = None) -> LazyObservation:
        values: dict[str, Array] = {self._state_key: np.asarray(raw_state[:STATE_DIM], dtype=np.float32)}
        if image is not None:
            values[self._image_key] = np.ascontiguousarray(image, dtype=np.uint8)
        return LazyObservation(self, values, self._image_key, self.step_index)

    def reset(self, seed: int) -> LazyObservation:
        raw, _info = self._lerobot.reset(seed=seed)  # LeRobot's reset (seeds object placement, renders)
        self.step_index = 0
        return self._observation(np.asarray(raw[RAW_STATE_KEY]), np.asarray(raw[RAW_IMAGE_KEY]))

    def step(self, action: FloatArray) -> StepResult:
        values = np.asarray(action, dtype=np.float32)
        if values.shape != (self.action_dim,):
            raise EnvError(f"action must have shape ({self.action_dim},), got {values.shape}")
        raw_state, reward, done, truncated, info = self._sim.step(values)
        self.step_index += 1
        success = bool(info.get("success", 0))
        return StepResult(
            observation=self._observation(np.asarray(raw_state)),
            reward=float(reward),
            terminated=bool(done) or success,
            truncated=bool(truncated),
            success=success,
            coverage=None,
        )

    def hold_action(self, observation: dict[str, Array]) -> FloatArray:
        """Stand still: zero end-effector displacement, the gripper keeps its current command.

        The xyz part of an action moves the mocap target by an offset, so 0 keeps the arm where it is. The
        gripper part is the position target of the two finger actuators (``ctrl = [a, -a]``); a 0 there would
        move the fingers to half-closed and could drop an object, so the hold action repeats the target the
        fingers have (open after a reset). What a robot does when its driver receives no new command.
        """
        ctrl = np.asarray(self._sim.data.ctrl, dtype=np.float64)
        if self.action_dim != 4 or ctrl.shape != (2,):
            raise EnvError(
                f"hold needs Meta-World's 4-D action and two finger actuators, got action_dim "
                f"{self.action_dim} and {ctrl.shape[0]} actuators"
            )
        action = np.zeros(self.action_dim, dtype=np.float32)
        action[3] = np.float32(np.clip(ctrl[0], -1.0, 1.0))
        return action

    def close(self) -> None:
        self._lerobot.close()
