"""Shared fixtures: a tiny randomly initialized diffusion policy saved in LeRobot's processor format.

The fixtures exercise the real LeRobot code paths (config, policy, processors) without downloading the
1 GB PushT checkpoint, so the whole test suite stays fast and offline.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # tiny fits: threads only oversubscribe a busy CPU

from pathlib import Path

import pytest

TINY_IMAGE = (3, 32, 32)  # C, H, W
TINY_STATE_DIM = 2
TINY_ACTION_DIM = 2


def tiny_diffusion_config():
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

    return DiffusionConfig(
        input_features={
            "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=TINY_IMAGE),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(TINY_STATE_DIM,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(TINY_ACTION_DIM,))},
        normalization_mapping={
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        },
        n_obs_steps=2,
        horizon=16,
        n_action_steps=8,
        down_dims=(16, 32),
        kernel_size=3,
        n_groups=4,
        diffusion_step_embed_dim=16,
        spatial_softmax_num_keypoints=4,
        crop_shape=(28, 28),
        crop_is_random=True,
        vision_backbone="resnet18",
        pretrained_backbone_weights=None,
        use_group_norm=True,
        num_train_timesteps=20,
        noise_scheduler_type="DDIM",
        num_inference_steps=4,
        device="cpu",
    )


def tiny_stats():
    import torch

    return {
        "observation.image": {
            "mean": torch.tensor([0.4, 0.5, 0.6]).reshape(3, 1, 1),
            "std": torch.tensor([0.2, 0.25, 0.3]).reshape(3, 1, 1),
        },
        "observation.state": {"min": torch.tensor([0.0, 0.0]), "max": torch.tensor([512.0, 512.0])},
        "action": {"min": torch.tensor([10.0, 20.0]), "max": torch.tensor([500.0, 480.0])},
    }


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory) -> Path:
    """Processor-format checkpoint directory of a tiny random diffusion policy."""
    import torch
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    torch.manual_seed(0)
    config = tiny_diffusion_config()
    policy = DiffusionPolicy(config)
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=tiny_stats())
    directory = tmp_path_factory.mktemp("tiny_diffusion")
    policy.save_pretrained(directory)
    preprocessor.save_pretrained(directory)
    postprocessor.save_pretrained(directory)
    return directory


@pytest.fixture(scope="session")
def legacy_checkpoint(tmp_path_factory, tiny_checkpoint) -> Path:
    """The tiny policy re-packed in the pre-0.4 LeRobot format (normalization buffers in the weights)."""
    import json

    from safetensors.torch import load_file, save_file

    stats = tiny_stats()
    state = load_file(tiny_checkpoint / "model.safetensors")
    for feature, prefix in (
        ("observation.image", "normalize_inputs"),
        ("observation.state", "normalize_inputs"),
        ("action", "normalize_targets"),
        ("action", "unnormalize_outputs"),
    ):
        buffer_name = feature.replace(".", "_")
        for stat_name, value in stats[feature].items():
            state[f"{prefix}.buffer_{buffer_name}.{stat_name}"] = value.clone()
    directory = tmp_path_factory.mktemp("legacy_diffusion")
    save_file(state, directory / "model.safetensors")
    config = json.loads((tiny_checkpoint / "config.json").read_text())
    config["crop_shape"] = list(config["crop_shape"])  # legacy JSON stores tuples as lists
    (directory / "config.json").write_text(json.dumps(config))
    return directory
