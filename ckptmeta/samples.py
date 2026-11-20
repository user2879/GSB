from __future__ import annotations

import io
import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .canonical import atomic_write_bytes, pretty_json_bytes, sha256_hex
from .config import InteropConfig, load_config, step_id
from .formats import decode_manifest_data, write_manifest
from .generator import materialize_content
from .ledger import Ledger, build_ledger


SAMPLE_CONFIG: dict[str, Any] = {
    "seed": 20261120,
    "shard_count": 4,
    "partition_mode": "modulo",
    "steps": [0, 10, 20],
    "sample_count": 64,
    "input_dim": 8,
    "hidden_dim": 6,
    "classes": 3,
}


@dataclass(frozen=True)
class FixturePaths:
    workspace: Path
    config: Path
    ledger: Path
    direct: Path
    historical_v1: Path
    historical_v2: Path
    crash_catalog: Path
    content: Path
    target_step: int

    @property
    def direct_target(self) -> Path:
        return self.direct / f"{step_id(self.target_step)}.v3.json"

    @property
    def v1_target(self) -> Path:
        return self.historical_v1 / f"{step_id(self.target_step)}.v1.json"

    @property
    def v2_target(self) -> Path:
        return self.historical_v2 / f"{step_id(self.target_step)}.v2.json"


def _force_remove(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        def onerror(function: Any, failing_path: str, exc_info: Any) -> None:
            os.chmod(failing_path, stat.S_IWRITE | stat.S_IREAD)
            function(failing_path)

        shutil.rmtree(path, onerror=onerror)
    else:
        try:
            os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        except OSError:
            pass
        path.unlink(missing_ok=True)


def clean_fixture_workspace(workspace: str | Path) -> None:
    root = Path(workspace)
    for name in (
        "direct",
        "historical_v1",
        "historical_v2",
        "crash_catalog",
        "ledger",
        "content",
        "logs",
        "snapshots",
    ):
        _force_remove(root / name)
    _force_remove(root / "config.json")


def _v1_record(view: dict[str, Any]) -> dict[str, Any]:
    return {
        "format_version": 1,
        "global_step": view["global_step"],
        "shard_count": view["shard_count"],
        "shards": [
            {
                "shard_index": shard["shard_index"],
                "parameter_digest": shard["parameter_digest"],
            }
            for shard in view["shards"]
        ],
    }


def _v2_record(view: dict[str, Any]) -> dict[str, Any]:
    return {
        "format_version": 2,
        "global_step": view["global_step"],
        "shard_count": view["shard_count"],
        "rng_state": dict(view["rng_state"]),
        "shards": [
            {
                "shard_index": shard["shard_index"],
                "parameter_digest": shard["parameter_digest"],
                "momentum_digest": shard["momentum_digest"],
            }
            for shard in view["shards"]
        ],
    }


def _write_record(path: Path, record: dict[str, Any]) -> None:
    decoded = decode_manifest_data(record, source_path=path)
    write_manifest(path, decoded)


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _write_content(workspace: Path, config: InteropConfig, step: int, ledger: Ledger) -> dict[str, Any]:
    content = materialize_content(config, step)
    directory = workspace / "content" / step_id(step)
    arrays: dict[str, np.ndarray] = {
        "dataset_X": content.features,
        "dataset_y": content.labels,
        **{f"parameter_{name}": value for name, value in sorted(content.parameters.items())},
        **{f"momentum_{name}": value for name, value in sorted(content.momentum.items())},
    }
    array_records: list[dict[str, Any]] = []
    for name, array in arrays.items():
        data = _npy_bytes(array)
        atomic_write_bytes(directory / f"{name}.npy", data)
        array_records.append(
            {
                "name": name,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": sha256_hex(data),
            }
        )
    summary = {
        "global_step": step,
        "arrays": array_records,
        "ledger_view_digest": ledger.entry(step).view_digest,
    }
    atomic_write_bytes(directory / "content-summary.json", pretty_json_bytes(summary))
    return summary


def generate_fixtures(workspace: str | Path, *, clean: bool = True) -> FixturePaths:
    root = Path(workspace).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if clean:
        clean_fixture_workspace(root)

    config_path = root / "config.json"
    atomic_write_bytes(config_path, pretty_json_bytes(SAMPLE_CONFIG))
    config = load_config(config_path)
    ledger = build_ledger(config)
    ledger_path = root / "ledger" / "truth-ledger.json"
    ledger.save(ledger_path)

    direct = root / "direct"
    v1 = root / "historical_v1"
    v2 = root / "historical_v2"
    crash = root / "crash_catalog"
    for directory in (direct, v1, v2, crash):
        directory.mkdir(parents=True, exist_ok=True)

    target_step = config.steps[-1]
    for config_step in config.steps:
        view = ledger.entry(config_step).view
        manifest = decode_manifest_data(
            {"format_version": 3, **view},
            source_path=direct / f"{step_id(config_step)}.v3.json",
        )
        write_manifest(direct / f"{step_id(config_step)}.v3.json", manifest)

    for directory in (v1, v2):
        for config_step in config.steps:
            if config_step == target_step:
                continue
            view = ledger.entry(config_step).view
            manifest = decode_manifest_data(
                {"format_version": 3, **view},
                source_path=directory / f"{step_id(config_step)}.v3.json",
            )
            write_manifest(directory / f"{step_id(config_step)}.v3.json", manifest)

    genesis_view = ledger.entry(0).view
    write_manifest(
        crash / f"{step_id(0)}.v3.json",
        decode_manifest_data(
            {"format_version": 3, **genesis_view},
            source_path=crash / f"{step_id(0)}.v3.json",
        ),
    )

    _write_record(v1 / f"{step_id(target_step)}.v1.json", _v1_record(ledger.entry(target_step).view))
    _write_record(v2 / f"{step_id(target_step)}.v2.json", _v2_record(ledger.entry(target_step).view))

    # The crash fixture has two low-version records so the first committed item
    # can be observed before the process is forcibly stopped.
    first_step = config.steps[1]
    _write_record(crash / f"{step_id(first_step)}.v1.json", _v1_record(ledger.entry(first_step).view))
    _write_record(crash / f"{step_id(target_step)}.v2.json", _v2_record(ledger.entry(target_step).view))

    content_summary = _write_content(root, config, target_step, ledger)
    atomic_write_bytes(
        root / "fixture-summary.json",
        pretty_json_bytes(
            {
                "target_step": target_step,
                "config_digest": ledger.config_digest,
                "ledger_digest": ledger.ledger_digest,
                "target_view_digest": ledger.entry(target_step).view_digest,
                "content_arrays": content_summary["arrays"],
            }
        ),
    )
    return FixturePaths(
        workspace=root,
        config=config_path,
        ledger=ledger_path,
        direct=direct,
        historical_v1=v1,
        historical_v2=v2,
        crash_catalog=crash,
        content=root / "content",
        target_step=target_step,
    )
