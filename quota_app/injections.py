from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .codec import load_document
from .config_workspace import CHECKPOINT_FILENAME, open_workspace
from .errors import AppError, E_FORMAT, E_ORDER, E_UPGRADE
from .events import load_events


INJECTION_KINDS = ("corrupt-config", "missing-checkpoint", "timestamp-regression")


@dataclass
class InjectionResult:
    kind: str
    observed_code: str | None
    error: dict[str, Any] | None
    workspace: str
    expected_code: str

    @property
    def passed(self) -> bool:
        return self.observed_code == self.expected_code

    def as_dict(self) -> dict[str, Any]:
        return {
            "injection": self.kind,
            "expected_code": self.expected_code,
            "observed_code": self.observed_code,
            "passed": self.passed,
            "error": self.error,
            "workspace": self.workspace,
        }


def _copy_sample_workspace(config_dir: str | Path, destination: Path) -> Path:
    source = Path(config_dir)
    if not source.exists():
        raise FileNotFoundError(source)
    target = destination / "configs"
    shutil.copytree(source, target)
    return target


def _config_files(config_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(config_dir.glob("*.json"))
        if path.name != CHECKPOINT_FILENAME
    ]


def _ensure_legacy_entry(config_dir: Path) -> None:
    files = _config_files(config_dir)
    if not files:
        raise FileNotFoundError("no configuration entries to downgrade")
    target = files[-1]
    raw = json.loads(target.read_text(encoding="utf-8-sig"))
    volumes = raw.get("volumes", [])
    legacy_volumes = [
        {"id": volume.get("id", volume.get("volume_id")), "capacity_bytes": volume.get("capacity_bytes", 0)}
        for volume in volumes
        if isinstance(volume, dict)
    ]
    if not legacy_volumes:
        legacy_volumes = [{"id": "legacy", "capacity_bytes": 1}]
    legacy = {
        "format_version": 1,
        "revision": int(raw.get("revision", 1)),
        "name": "injected-legacy",
        "volumes": legacy_volumes,
    }
    target.write_text(json.dumps(legacy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _make_timestamp_regression(events_path: str | Path, target: Path) -> Path:
    source = Path(events_path)
    lines = [line for line in source.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    records = [json.loads(line) for line in lines]
    if not records:
        records = [
            {"seq": 1, "timestamp_ms": 10, "volume_id": "root", "operation_count": 1, "write_bytes": 1},
            {"seq": 2, "timestamp_ms": 20, "volume_id": "root", "operation_count": 1, "write_bytes": 1},
        ]
    index = 1 if len(records) > 1 else 0
    timestamp_key = "timestamp_ms" if "timestamp_ms" in records[index] else "timestamp"
    if index > 0:
        previous_key = "timestamp_ms" if "timestamp_ms" in records[index - 1] else "timestamp"
        previous_timestamp = int(records[index - 1].get(previous_key, 10))
        if previous_timestamp <= 0:
            records[index - 1][previous_key] = 10
            previous_timestamp = 10
        records[index][timestamp_key] = max(0, previous_timestamp - 1)
    else:
        records[index][timestamp_key] = 0
    output = target / "events-regression.ndjson"
    output.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return output


def run_injection(
    kind: str,
    *,
    config_dir: str | Path,
    events_path: str | Path,
) -> InjectionResult:
    if kind not in INJECTION_KINDS:
        raise ValueError(f"unknown injection kind: {kind}")

    with tempfile.TemporaryDirectory(prefix="quota-injection-") as temporary:
        root = Path(temporary)
        copied_config = _copy_sample_workspace(config_dir, root)
        observed_code: str | None = None
        error_payload: dict[str, Any] | None = None
        expected = {
            "corrupt-config": E_FORMAT,
            "missing-checkpoint": E_UPGRADE,
            "timestamp-regression": E_ORDER,
        }[kind]

        try:
            if kind == "corrupt-config":
                files = _config_files(copied_config)
                if not files:
                    raise FileNotFoundError("no configuration file available")
                files[0].write_text('{"format_version": 3, "volumes": [', encoding="utf-8")
                load_document(files[0])
            elif kind == "missing-checkpoint":
                checkpoint = copied_config / CHECKPOINT_FILENAME
                if checkpoint.exists():
                    checkpoint.unlink()
                has_low_version = False
                for path in _config_files(copied_config):
                    raw = json.loads(path.read_text(encoding="utf-8-sig"))
                    version = raw.get("format_version", raw.get("version", raw.get("schema_version")))
                    if isinstance(version, int) and version < 3:
                        has_low_version = True
                        break
                if not has_low_version:
                    _ensure_legacy_entry(copied_config)
                result = open_workspace(copied_config)
                upgrade_errors = [error for error in result.upgrade.errors if error.code == E_UPGRADE]
                if not upgrade_errors:
                    raise AssertionError("missing checkpoint did not produce E_UPGRADE")
                raise upgrade_errors[0]
            else:
                regression = _make_timestamp_regression(events_path, root)
                load_events(regression)
        except AppError as exc:
            observed_code = exc.code
            error_payload = exc.as_dict()

        return InjectionResult(
            kind=kind,
            observed_code=observed_code,
            error=error_payload,
            workspace=str(copied_config),
            expected_code=expected,
        )
