"""LeRobot policy wrapper used by the Kortyn Brain: observation history in, action chunk out.

This is the only module that touches torch at inference time. It reproduces LeRobot's evaluation
semantics exactly (verified by ``tests/test_policy_runner.py::test_matches_lerobot_select_action``):

* images arrive as uint8 ``(T, H, W, C)`` and are converted like ``lerobot.envs.utils.preprocess_observation``
  (float32, channel-first, ``/255``) *before* the processor pipeline, so normalization never runs on
  integer tensors;
* the batch is built explicitly as ``(1, T, ...)`` and ``policy.predict_action_chunk`` runs in its offline
  mode; ``policy.reset()`` before every request guarantees no stale internal observation queue is used;
* the initial diffusion noise comes from a CPU ``torch.Generator`` seeded per request, so results do not
  depend on the device and identical observations with identical seeds give identical chunks;
* post-processing (un-normalization) runs per action, treating the chunk steps as the batch dimension.

Phase 2 adds an optional **guide** (inpainting of the transition between plans, see ``kortyn/inpainting.py``):
the guided path rebuilds the batch exactly like ``predict_action_chunk``'s offline mode and replaces only
the sampling loop; with all guide weights 0 it reproduces the unguided chunk bit for bit.

Phase 3a adds vision-language-action policies (SmolVLA): the request's language instruction is passed as
the batch's ``task`` (tokenized by the policy's own preprocessor, as in LeRobot's evaluation), the
flow-matching noise comes from the same seeded CPU generator, and the profile's ``num_inference_steps``
sets the number of flow-matching integration steps. Phase 3: a profile can declare the camera resolution
the Brain accepts (``image_size``), since SmolVLA resizes images itself and LeRobot's LIBERO evaluation feeds
it frames of another size than its checkpoint features state.
"""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import logging
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.processor import (
    NormalizerProcessorStep,
    TokenizerProcessorStep,
    TransitionKey,
    create_transition,
)
from lerobot.utils.constants import OBS_IMAGES

from kortyn.config import BrainProfile, ConfigError
from kortyn.inpainting import GuidanceError, guided_conditional_sample
from kortyn.typing_utils import Array, FloatArray

LOG = logging.getLogger("kortyn.policy")

_INPUT_KINDS = (FeatureType.VISUAL, FeatureType.STATE, FeatureType.ENV)


class ObservationError(ValueError):
    """The observation in a request does not match the policy's input features."""


@dataclass(frozen=True)
class FeatureInfo:
    key: str
    kind: str  # "VISUAL" | "STATE" | "ENV"
    shape: tuple[int, ...]  # policy-side shape, images (C, H, W)


@dataclass(frozen=True)
class PolicyInfo:
    policy_type: str
    checkpoint: str
    device: str
    n_obs_steps: int
    max_actions: int
    action_dim: int
    input_features: tuple[FeatureInfo, ...]
    sampler: str
    lerobot_version: str
    torch_version: str
    torch_threads: int
    warmup_inference_ms: float
    supports_guidance: bool = False
    needs_task: bool = False  # the policy is conditioned on a language instruction


@dataclass(frozen=True)
class GuideSpec:
    """Inpainting guide relative to the request: ``offset`` = control step of ``actions[0]`` minus the
    observation step (``-1`` = the action executed at the previous tick)."""

    offset: int
    actions: FloatArray  # (G, action_dim), env units
    weights: FloatArray  # (G,), in [0, 1]


@dataclass(frozen=True)
class InferenceResult:
    actions: FloatArray  # (N, action_dim)
    inference_ms: float


def resolve_device(requested: str) -> str:
    """Map ``auto`` to the best available backend (mps -> cuda -> cpu) and validate explicit choices."""
    mps_available = bool(torch.backends.mps.is_available())
    cuda_available = bool(torch.cuda.is_available())
    if requested == "auto":
        if mps_available:
            return "mps"
        if cuda_available:
            return "cuda"
        return "cpu"
    if requested == "mps" and not mps_available:
        raise ConfigError("device 'mps' requested but MPS is not available on this machine")
    if requested.startswith("cuda") and not cuda_available:
        raise ConfigError(f"device {requested!r} requested but CUDA is not available")
    if requested not in ("cpu", "mps") and not requested.startswith("cuda"):
        raise ConfigError(f"unsupported device {requested!r}; use auto, cpu, mps or cuda[:N]")
    return requested


WARMUP_TASK = "warm up"  # language instruction of the warm-up inferences (content is irrelevant)


def _apply_profile(config: PreTrainedConfig, profile: BrainProfile) -> int:
    """Apply sampler/chunk settings to the policy config and return the chunk length."""
    if config.type == "smolvla":
        # Flow matching: num_inference_steps = Euler integration steps; the whole chunk is returned.
        config.num_steps = profile.num_inference_steps
        chunk = int(config.chunk_size)
        if profile.n_action_steps is not None:
            if not 1 <= profile.n_action_steps <= chunk:
                raise ConfigError(f"n_action_steps={profile.n_action_steps} not in [1, chunk_size={chunk}]")
            config.n_action_steps = profile.n_action_steps
        return chunk
    if config.type == "diffusion":
        config.noise_scheduler_type = profile.noise_scheduler_type
        config.num_inference_steps = profile.num_inference_steps
        maximum = int(config.horizon) - int(config.n_obs_steps) + 1
        chunk = maximum if profile.n_action_steps is None else profile.n_action_steps
        if not 1 <= chunk <= maximum:
            raise ConfigError(
                f"n_action_steps={chunk} not supported: diffusion horizon {config.horizon} with "
                f"n_obs_steps {config.n_obs_steps} allows at most {maximum}"
            )
        config.n_action_steps = chunk
        return chunk
    if profile.n_action_steps is not None:
        config.n_action_steps = profile.n_action_steps
    LOG.warning(
        "sampler settings of the profile only apply to diffusion policies; %s uses its own sampler",
        config.type,
    )
    return int(config.n_action_steps)


def _with_image_size(
    config: PreTrainedConfig, features: tuple[FeatureInfo, ...], image_size: Sequence[int]
) -> tuple[FeatureInfo, ...]:
    """Camera features at ``image_size`` (``[height, width]``), for policies that resize images themselves.

    SmolVLA pads every image to ``resize_imgs_with_padding`` before its vision encoder, so any camera
    resolution is a valid input; LeRobot's own evaluation feeds it whatever the environment renders.
    """
    any_config: Any = config
    if config.type != "smolvla" or getattr(any_config, "resize_imgs_with_padding", None) is None:
        raise ConfigError(
            f"profile image_size needs a policy that resizes its images itself (SmolVLA), got {config.type}"
        )
    height, width = (int(d) for d in image_size)
    return tuple(
        dataclasses.replace(feature, shape=(feature.shape[0], height, width))
        if feature.kind == "VISUAL"
        else feature
        for feature in features
    )


class PolicyRunner:
    """Owns one policy instance. Not thread-safe: call ``infer`` from a single thread."""

    def __init__(
        self,
        checkpoint_dir: Path,
        profile: BrainProfile,
        device: str = "auto",
        torch_threads: int = 1,
    ) -> None:
        import lerobot

        self.device = resolve_device(device)
        torch.set_num_threads(max(1, torch_threads))
        # Raises if inter-op parallelism was already initialized in this process (e.g. a second runner in
        # tests); keeping the existing setting is harmless.
        with contextlib.suppress(RuntimeError):
            torch.set_num_interop_threads(1)

        config = PreTrainedConfig.from_pretrained(checkpoint_dir)
        self.max_actions = _apply_profile(config, profile)
        config.device = self.device
        self.config = config
        self.policy = get_policy_class(config.type).from_pretrained(
            checkpoint_dir, config=config, strict=True
        )
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            config,
            pretrained_path=str(checkpoint_dir),
            preprocessor_overrides={"device_processor": {"device": self.device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        self.n_obs_steps = int(config.n_obs_steps)
        self.action_dim = int(config.output_features["action"].shape[0])
        self.features = tuple(
            FeatureInfo(key=key, kind=feature.type.value, shape=tuple(int(d) for d in feature.shape))
            for key, feature in config.input_features.items()
            if feature.type in _INPUT_KINDS
        )
        if profile.image_size is not None:
            self.features = _with_image_size(config, self.features, profile.image_size)
        accepts_noise = "noise" in inspect.signature(self.policy.predict_action_chunk).parameters
        horizon = getattr(config, "horizon", None)
        self._noise_shape: tuple[int, int, int] | None = None
        if accepts_noise and horizon is not None:
            self._noise_shape = (1, int(horizon), self.action_dim)
        elif accepts_noise and config.type == "smolvla":
            any_config: Any = config
            self._noise_shape = (1, int(any_config.chunk_size), int(any_config.max_action_dim))
        self._action_normalizer = next(
            (step for step in self.preprocessor.steps if isinstance(step, NormalizerProcessorStep)), None
        )
        self.needs_task = any(isinstance(step, TokenizerProcessorStep) for step in self.preprocessor.steps)
        if config.type == "diffusion":
            sampler = profile.sampler_label
        elif config.type == "smolvla":
            sampler = f"flow-{profile.num_inference_steps}"
        else:
            sampler = str(getattr(config, "type", "policy"))
        self._info = PolicyInfo(
            policy_type=str(config.type),
            checkpoint=str(checkpoint_dir),
            device=self.device,
            n_obs_steps=self.n_obs_steps,
            max_actions=self.max_actions,
            action_dim=self.action_dim,
            input_features=self.features,
            sampler=sampler,
            lerobot_version=str(getattr(lerobot, "__version__", "unknown")),
            torch_version=str(torch.__version__),
            torch_threads=int(torch.get_num_threads()),
            warmup_inference_ms=0.0,
            supports_guidance=self.supports_guidance,
            needs_task=self.needs_task,
        )
        LOG.info(
            "loaded %s from %s on %s (n_obs_steps=%d, chunk=%d, sampler=%s)",
            config.type,
            checkpoint_dir,
            self.device,
            self.n_obs_steps,
            self.max_actions,
            sampler,
        )

    # ------------------------------------------------------------------ info
    @property
    def info(self) -> PolicyInfo:
        return self._info

    @property
    def supports_guidance(self) -> bool:
        """Inpainting needs the diffusion sampler and an explicit initial noise."""
        return str(self.config.type) == "diffusion" and self._noise_shape is not None

    def dummy_observation(self) -> dict[str, Array]:
        """A valid observation history of zeros (used for warm-up and tests)."""
        observation: dict[str, Array] = {}
        for feature in self.features:
            if feature.kind == "VISUAL":
                c, h, w = feature.shape
                observation[feature.key] = np.zeros((self.n_obs_steps, h, w, c), dtype=np.uint8)
            else:
                observation[feature.key] = np.zeros((self.n_obs_steps, *feature.shape), dtype=np.float32)
        return observation

    def warmup(self, iterations: int) -> float:
        """Run ``iterations`` inferences (kernel selection, MPS shader compilation); return median ms."""
        task = WARMUP_TASK if self.needs_task else None
        timings = [
            self.infer(self.dummy_observation(), seed=i, max_actions=self.max_actions, task=task).inference_ms
            for i in range(iterations)
        ]
        median_ms = float(statistics.median(timings)) if timings else 0.0
        self._info = dataclasses.replace(self._info, warmup_inference_ms=median_ms)
        return median_ms

    # ------------------------------------------------------------------ inference
    def validate(self, observation: Mapping[str, Array]) -> None:
        """Raise ``ObservationError`` unless ``observation`` matches the policy inputs exactly."""
        expected_keys = {f.key for f in self.features}
        missing = expected_keys - set(observation)
        if missing:
            raise ObservationError(f"missing observation keys: {sorted(missing)}")
        unexpected = set(observation) - expected_keys
        if unexpected:
            raise ObservationError(f"unexpected observation keys: {sorted(unexpected)}")
        for feature in self.features:
            value = observation[feature.key]
            if feature.kind == "VISUAL":
                c, h, w = feature.shape
                expected_shape: tuple[int, ...] = (self.n_obs_steps, h, w, c)
                if value.dtype != np.uint8:
                    raise ObservationError(f"{feature.key}: expected uint8 images, got {value.dtype}")
            else:
                expected_shape = (self.n_obs_steps, *feature.shape)
                if value.dtype not in (np.float32, np.float64):
                    raise ObservationError(f"{feature.key}: expected float32 values, got {value.dtype}")
            if tuple(value.shape) != expected_shape:
                raise ObservationError(
                    f"{feature.key}: expected shape {expected_shape} (n_obs_steps={self.n_obs_steps}), "
                    f"got {tuple(value.shape)}"
                )

    def _to_batch(self, observation: Mapping[str, Array]) -> dict[str, Any]:
        batch: dict[str, Any] = {}
        for feature in self.features:
            value = observation[feature.key]
            if feature.kind == "VISUAL":
                # copy: decoded request arrays are read-only views on the protobuf payload
                tensor = torch.from_numpy(np.array(value, dtype=np.uint8, copy=True)).permute(0, 3, 1, 2)
                tensor = tensor.to(torch.float32).div_(255.0)  # (T, C, H, W) in [0, 1]
            else:
                tensor = torch.from_numpy(np.asarray(value, dtype=np.float32).copy())  # (T, D)
            if self.n_obs_steps == 1:
                tensor = tensor[0]
            batch[feature.key] = tensor.unsqueeze(0)  # (1, T, ...) or (1, ...) for single-frame policies
        return batch

    def check_guide(self, guide: GuideSpec) -> None:
        """Raise ``ObservationError`` unless the guide can be applied to this policy's chunk."""
        if not self.supports_guidance:
            raise ObservationError(f"inpainting guides are not supported for {self.config.type} policies")
        actions = np.asarray(guide.actions)
        weights = np.asarray(guide.weights)
        if actions.ndim != 2 or actions.shape[1] != self.action_dim:
            raise ObservationError(f"guide actions must be (G, {self.action_dim}), got {actions.shape}")
        if weights.shape != (actions.shape[0],) or actions.shape[0] == 0:
            raise ObservationError(
                f"guide needs one weight per action, got {weights.shape} for {actions.shape[0]} actions"
            )
        if not (np.all(np.isfinite(actions)) and np.all(np.isfinite(weights))):
            raise ObservationError("guide contains non-finite values")
        if weights.min() < 0.0 or weights.max() > 1.0:
            raise ObservationError("guide weights must be in [0, 1]")
        first = guide.offset + self.n_obs_steps - 1
        if first < 0 or first >= int(self.config.horizon):
            raise ObservationError(
                f"guide starts at step offset {guide.offset}, outside the chunk "
                f"[{1 - self.n_obs_steps}, {int(self.config.horizon) - self.n_obs_steps}]"
            )

    def _guide_tensors(self, guide: GuideSpec) -> tuple[torch.Tensor, torch.Tensor]:
        """(known, weights) on the horizon grid: normalized guide actions and ``(1, horizon, 1)`` weights."""
        horizon = int(self.config.horizon)
        first = guide.offset + self.n_obs_steps - 1
        count = min(len(guide.weights), horizon - first)  # entries beyond the horizon have no counterpart
        actions = torch.as_tensor(np.asarray(guide.actions[:count], dtype=np.float32), device=self.device)
        if self._action_normalizer is not None:
            actions = self._action_normalizer(create_transition(action=actions))[TransitionKey.ACTION]
        known = torch.zeros((1, horizon, self.action_dim), dtype=torch.float32, device=self.device)
        weights = torch.zeros((1, horizon, 1), dtype=torch.float32, device=self.device)
        known[0, first : first + count] = actions.to(device=self.device, dtype=torch.float32)
        weights[0, first : first + count, 0] = torch.as_tensor(
            np.asarray(guide.weights[:count], dtype=np.float32), device=self.device
        )
        return known, weights

    def _guided_chunk(self, batch: dict[str, Any], noise: torch.Tensor, guide: GuideSpec) -> torch.Tensor:
        """``predict_action_chunk`` (offline mode) with the inpainting sampler."""
        config: Any = self.config
        batch = dict(batch)
        if config.image_features:
            for key in config.image_features:
                if batch[key].ndim == 4:
                    batch[key] = batch[key].unsqueeze(1)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in config.image_features], dim=-4)
        model: Any = self.policy.diffusion
        global_cond = model._prepare_global_conditioning(batch)
        known, weights = self._guide_tensors(guide)
        try:
            actions = guided_conditional_sample(model, global_cond, noise, known, weights)
        except GuidanceError as exc:
            raise ObservationError(str(exc)) from exc
        start = self.n_obs_steps - 1
        return actions[:, start : start + int(config.n_action_steps)]

    def infer(
        self,
        observation: Mapping[str, Array],
        seed: int,
        max_actions: int,
        guide: GuideSpec | None = None,
        task: str | None = None,
    ) -> InferenceResult:
        if not 1 <= max_actions <= self.max_actions:
            raise ObservationError(f"max_actions must be in [1, {self.max_actions}], got {max_actions}")
        if guide is not None:
            self.check_guide(guide)
        if self.needs_task and not task:
            raise ObservationError(f"{self.config.type} policies need a language instruction (task)")
        start = time.perf_counter()
        self.policy.reset()
        raw_batch = self._to_batch(observation)
        if self.needs_task:
            raw_batch["task"] = [task]
        batch = self.preprocessor(raw_batch)
        torch.manual_seed(seed % (2**63))
        kwargs: dict[str, Any] = {}
        if self._noise_shape is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed % (2**63))
            noise = torch.randn(self._noise_shape, generator=generator, dtype=torch.float32)
            kwargs["noise"] = noise.to(self.device)
        with torch.inference_mode():
            if guide is None:
                chunk = self.policy.predict_action_chunk(batch, **kwargs)
            else:
                chunk = self._guided_chunk(batch, kwargs["noise"], guide)
            if chunk.ndim == 2:
                chunk = chunk.unsqueeze(0)
            chunk = chunk[:, :max_actions]
            steps = int(chunk.shape[1])
            actions = self.postprocessor(chunk.reshape(steps, -1))
        array = actions.detach().to(device="cpu", dtype=torch.float32).numpy()  # synchronizes the device
        return InferenceResult(actions=array, inference_ms=(time.perf_counter() - start) * 1000.0)
