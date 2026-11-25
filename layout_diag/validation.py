"""Combined dataset/config/annotation validation entry point."""

from __future__ import annotations

from pathlib import Path

from .config import Config, load_config
from .dataset import load_dataset
from .models import Dataset, FailureLog


def validate_workspace(
    dataset_root: str | Path,
    config_path: str | Path,
) -> tuple[Dataset, Config, FailureLog]:
    failures = FailureLog()
    dataset = load_dataset(dataset_root, failures)
    config = load_config(config_path, failures)

    if dataset.resolution != config.resolution:
        failures.add(
            "E_IMAGE_SIZE",
            "workspace",
            "resolution",
            "dataset and config resolutions disagree",
            dataset=list(dataset.resolution),
            config=list(config.resolution),
        )
    if dataset.pixel_mode != config.pixel_mode:
        failures.add(
            "E_IMAGE_MODE",
            "workspace",
            "pixel_mode",
            "dataset and config pixel modes disagree",
            dataset=dataset.pixel_mode,
            config=config.pixel_mode,
        )
    for entry in dataset.images:
        versions = dataset.keypoint_versions.get(entry.name, {})
        if config.keypoint_version not in versions:
            failures.add(
                "E_KEYPOINT_VERSION",
                "image",
                entry.name,
                "active keypoint version is not available",
                active=config.keypoint_version,
                available=sorted(versions),
            )
    return dataset, config, failures


def validate_documents(dataset_root: str | Path, config_path: str | Path) -> dict:
    dataset, config, failures = validate_workspace(dataset_root, config_path)
    return {
        "dataset": {
            "root": dataset.root,
            "content_hash": dataset.content_hash,
            "images": [entry.name for entry in dataset.images],
            "scanned_pngs": list(dataset.scanned_pngs),
            "stats": dataset.stats(),
        },
        "config": {
            "path": config.path,
            "content_hash": config.content_hash(),
            "tier": config.tier,
            "active_keypoint_version": config.keypoint_version,
        },
        "failures": failures.as_dicts(),
        "ok": not failures,
    }
