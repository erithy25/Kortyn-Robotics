"""Download a LeRobot checkpoint at a pinned revision and migrate it to the processor-pipeline format.

``lerobot/diffusion_pusht`` predates LeRobot 0.4: its normalization statistics are buffers inside
``model.safetensors`` and it ships no ``policy_preprocessor.json``. LeRobot's own migration script
(``lerobot/processor/migrate_policy_normalization.py``) crashes on it in 0.6.1 because it passes the raw
JSON config (``crop_shape`` as a list) into the dataclass. This module performs the same steps with
LeRobot's own helper functions, but decodes the config through ``PreTrainedConfig.from_pretrained``:

1. extract the normalization statistics from the state dict,
2. remove the normalization buffers,
3. load the clean weights strictly (no missing or unexpected keys),
4. build pre/post-processor pipelines from the statistics,
5. save model + processors locally.

Heavy imports (torch, lerobot) are local to the functions so that importing this module is cheap.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from kortyn.config import CheckpointConfig

LOG = logging.getLogger("kortyn.checkpoint")

PROCESSOR_FILES = ("policy_preprocessor.json", "policy_postprocessor.json")
MARKER_FILE = "kortyn_migration.json"


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be downloaded, migrated or validated."""


@dataclass(frozen=True)
class MigrationReport:
    source: str
    output_dir: Path
    normalized_features: tuple[str, ...]
    removed_keys: int


def is_processor_format(path: Path) -> bool:
    return all((path / name).is_file() for name in PROCESSOR_FILES) and (path / "model.safetensors").is_file()


def migrate_legacy_checkpoint(source_dir: Path, output_dir: Path) -> MigrationReport:
    """Migrate a local pre-0.4 checkpoint directory (config.json + model.safetensors) to ``output_dir``."""
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import get_policy_class, make_pre_post_processors
    from lerobot.processor.migrate_policy_normalization import (
        clean_state_dict,
        extract_normalization_stats,
        remove_normalization_layers,
    )
    from safetensors.torch import load_file

    weights = source_dir / "model.safetensors"
    if not weights.is_file() or not (source_dir / "config.json").is_file():
        raise CheckpointError(f"{source_dir} must contain config.json and model.safetensors")

    config = PreTrainedConfig.from_pretrained(source_dir)
    config.device = "cpu"
    state_dict = load_file(weights)
    stats = extract_normalization_stats(state_dict)
    if not stats:
        raise CheckpointError(
            f"no normalization buffers found in {weights}; this does not look like a legacy checkpoint"
        )
    clean = clean_state_dict(remove_normalization_layers(state_dict), remove_str="._orig_mod")

    policy = get_policy_class(config.type)(config)
    missing, unexpected = policy.load_state_dict(clean, strict=False)
    if missing or unexpected:
        raise CheckpointError(f"weights do not match the policy: missing={missing} unexpected={unexpected}")
    policy.to(torch.float32)
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=stats)

    tmp_dir = output_dir.with_name(output_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    policy.save_pretrained(tmp_dir)
    preprocessor.save_pretrained(tmp_dir)
    postprocessor.save_pretrained(tmp_dir)
    report = MigrationReport(
        source=str(source_dir),
        output_dir=output_dir,
        normalized_features=tuple(sorted(stats)),
        removed_keys=len(state_dict) - len(clean),
    )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    tmp_dir.rename(output_dir)
    return report


def ensure_checkpoint(config: CheckpointConfig) -> Path:
    """Return a local directory with a processor-format checkpoint, downloading/migrating if needed."""
    output_dir = Path(config.local_dir)
    marker = output_dir / MARKER_FILE
    if is_processor_format(output_dir) and marker.is_file():
        recorded = json.loads(marker.read_text(encoding="utf-8"))
        if recorded.get("repo_id") == config.repo_id and recorded.get("revision") == config.revision:
            return output_dir
        LOG.warning("checkpoint in %s was built from %s; rebuilding", output_dir, recorded)

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError

    LOG.info("downloading %s@%s", config.repo_id, config.revision)
    try:
        snapshot = Path(
            snapshot_download(
                config.repo_id,
                revision=config.revision,
                allow_patterns=["config.json", "model.safetensors", "*.json", "*.safetensors"],
            )
        )
    except (HfHubHTTPError, OSError) as exc:
        raise CheckpointError(f"could not download {config.repo_id}@{config.revision}: {exc}") from exc

    if is_processor_format(snapshot):
        LOG.info("checkpoint already uses the processor format; copying")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(snapshot, output_dir, ignore=shutil.ignore_patterns("*.mp4", "*.gif", "*.png"))
        features: tuple[str, ...] = ()
    else:
        LOG.info("migrating legacy checkpoint to the processor format")
        report = migrate_legacy_checkpoint(snapshot, output_dir)
        features = report.normalized_features
        LOG.info("migrated: normalized features %s, removed %d buffer keys", features, report.removed_keys)

    marker.write_text(
        json.dumps(
            {"repo_id": config.repo_id, "revision": config.revision, "normalized_features": list(features)},
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_dir
