"""Gymnasium simulation wrapper for the Kortyn Agent.

Deliberately free of torch/lerobot imports: the agent process models the robot, which has no model
runtime. Raw env observations are renamed to LeRobot feature keys via ``EnvConfig.features_map``; images
stay uint8 (transport format), state vectors become float32 (as LeRobot's ``preprocess_observation`` does).
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import gymnasium as gym
import numpy as np

from kortyn.config import EnvConfig
from kortyn.typing_utils import Array, FloatArray

if TYPE_CHECKING:
    from kortyn.libero_env import LiberoEnv
    from kortyn.metaworld_env import MetaWorldEnv

Observation = dict[str, Array]


class EnvError(RuntimeError):
    """Raised when the environment cannot be created or returns unexpected data."""


@dataclass(frozen=True)
class StepResult:
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    success: bool
    coverage: float | None


class SimEnv:
    """One gymnasium environment with LeRobot-style observation keys."""

    def __init__(self, config: EnvConfig) -> None:
        self.config = config
        try:
            importlib.import_module(config.package)
        except ImportError as exc:
            raise EnvError(f"cannot import env package {config.package!r}: {exc}") from exc
        try:
            self._env = gym.make(
                config.gym_id,
                disable_env_checker=True,
                max_episode_steps=config.max_episode_steps,
                **config.gym_kwargs,
            )
        except gym.error.Error as exc:
            raise EnvError(f"cannot create env {config.gym_id!r}: {exc}") from exc
        shape = self._env.action_space.shape
        if shape is None or len(shape) != 1:
            raise EnvError(f"expected a flat Box action space, got {self._env.action_space}")
        self.action_dim = int(shape[0])

    @property
    def fps(self) -> float:
        return self.config.fps

    @property
    def max_episode_steps(self) -> int:
        return self.config.max_episode_steps

    def _map(self, raw: Any) -> Observation:
        if not isinstance(raw, dict):
            raise EnvError(f"expected a dict observation, got {type(raw).__name__}")
        observation: Observation = {}
        for raw_key, feature_key in self.config.features_map.items():
            if raw_key not in raw:
                raise EnvError(f"observation key {raw_key!r} missing from env output {sorted(raw)}")
            value = np.asarray(raw[raw_key])
            if value.dtype != np.uint8:
                value = value.astype(np.float32)
            observation[feature_key] = value
        return observation

    def reset(self, seed: int) -> Observation:
        raw, _info = self._env.reset(seed=seed)
        return self._map(raw)

    def step(self, action: FloatArray) -> StepResult:
        raw, reward, terminated, truncated, info = self._env.step(np.asarray(action, dtype=np.float32))
        coverage = info.get("coverage")
        return StepResult(
            observation=self._map(raw),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            success=bool(info.get("is_success", False)),
            coverage=None if coverage is None else float(coverage),
        )

    def hold_action(self, observation: Observation) -> FloatArray:
        """Action that keeps the robot where it is (PushT: target = current agent position)."""
        if self.config.hold_action_key is None:
            return np.zeros(self.action_dim, dtype=np.float32)
        value = np.asarray(observation[self.config.hold_action_key], dtype=np.float32)
        if value.shape != (self.action_dim,):
            raise EnvError(
                f"hold_action_key {self.config.hold_action_key!r} has shape {value.shape}, "
                f"expected ({self.action_dim},)"
            )
        return value

    def close(self) -> None:
        self._env.close()


def make_env(config: EnvConfig) -> SimEnv | MetaWorldEnv | LiberoEnv:
    """Create the environment described by ``config`` (gym env, Meta-World (phase 3a) or LIBERO (phase 3))."""
    if config.kind == "metaworld":
        from kortyn.metaworld_env import MetaWorldEnv

        return MetaWorldEnv(config)
    if config.kind == "libero":
        from kortyn.libero_env import LiberoEnv

        return LiberoEnv(config)
    return SimEnv(config)
