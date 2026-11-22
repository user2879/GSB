"""Incremental edit plan and deterministic workspace materialization."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .datagen import generate_sample, write_sample
from .hashing import sha256_text


@dataclass
class EditBatch:
    batch_id: str
    edits: list[dict]

    @property
    def config_updates(self) -> dict:
        nested: dict = {}
        for edit in self.edits:
            if edit.get("op") == "update_config":
                _merge_nested(nested, edit.get("config", {}))
        return nested


def _merge_nested(target: dict, source: dict) -> dict:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_nested(target[key], value)
        else:
            target[key] = value
    return target


@dataclass
class EditPlan:
    base_config_path: Path
    base_sample_dir: Path
    batches: list[EditBatch]

    @classmethod
    def load(cls, path: str | Path) -> "EditPlan":
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        base_dir = Path(path).resolve().parent
        config_path = Path(data["base_config"])
        if not config_path.is_absolute():
            config_path = base_dir / config_path
        sample_dir = Path(data["base_sample_dir"])
        if not sample_dir.is_absolute():
            sample_dir = base_dir / sample_dir
        batches = [
            EditBatch(batch_id=str(batch.get("id", idx)), edits=list(batch.get("edits", [])))
            for idx, batch in enumerate(data.get("batches", []))
        ]
        return cls(base_config_path=config_path, base_sample_dir=sample_dir, batches=batches)


class Workspace:
    """Materializes the sample directory state for a prefix of edit batches."""

    def __init__(self, root: str | Path, base_config: Config, base_sample_dir: str | Path):
        self.root = Path(root)
        self.samples_dir = self.root / "samples"
        self.root.mkdir(parents=True, exist_ok=True)
        if self.samples_dir.exists():
            shutil.rmtree(self.samples_dir)
        shutil.copytree(base_sample_dir, self.samples_dir)
        self.base_config = base_config

    def apply_batches(self, batches: list[EditBatch], *, max_index: int | None = None) -> Config:
        """Replay edits 0..max_index deterministically onto the workspace."""
        config = self.base_config
        limit = len(batches) if max_index is None else max_index + 1
        for batch in batches[:limit]:
            for edit in batch.edits:
                config = self._apply_edit(edit, config)
        return config

    def _apply_edit(self, edit: dict, config: Config) -> Config:
        op = edit.get("op")
        if op in ("add_sample", "replace_sample"):
            sample = edit["sample"]
            name = sample["name"]
            index = int(sample.get("index", int(sha256_text(name)[:8], 16) % 10_000))
            variant = int(sample.get("variant", 0))
            width, height = config.image_size
            png_bytes, annotation = generate_sample(
                name,
                index,
                tuple(config.labels),
                dict(config.positive_ratios),
                width=width,
                height=height,
                base_seed=config.seed,
                variant=variant,
            )
            write_sample(self.samples_dir, name, png_bytes, annotation)
            return config
        if op == "delete_sample":
            for suffix in (".png", ".json"):
                path = self.samples_dir / f"{edit['sample']}{suffix}"
                if path.exists():
                    path.unlink()
            return config
        if op == "add_label":
            label = edit["label"]
            if label in config.labels:
                return config
            labels = list(config.labels) + [label]
            ratios = dict(config.positive_ratios)
            ratios[label] = float(edit.get("ratio", 0.5))
            return config.with_updates({"labels": labels, "positive_ratios": ratios})
        if op == "remove_label":
            label = edit["label"]
            if label not in config.labels:
                return config
            labels = [item for item in config.labels if item != label]
            ratios = {key: value for key, value in config.positive_ratios.items() if key != label}
            label_thresholds = {
                key: value for key, value in config.label_thresholds.items() if key != label
            }
            updates = {"labels": labels, "positive_ratios": ratios}
            if label_thresholds:
                updates["label_thresholds"] = label_thresholds
            return config.with_updates(updates)
        if op == "update_config":
            return config.with_updates(edit.get("config", {}))
        raise ValueError(f"unknown edit op {op!r}")
