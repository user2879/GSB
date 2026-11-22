from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from .canonical import canonical_bytes, sha256_bytes, true_int
from .constants import (
    ALLOWED_ELAPSED_SEMANTICS,
    ALLOWED_SCAN_SEMANTICS,
    DB_EXEC_STATS,
    KNOWN_VERSIONS,
    SOURCE_TYPES,
    TYPE_ALIASES,
)
from .errors import source_error


SOURCE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
SCHEMA_NAME = "slowdiag.sources/v1"


@dataclass(frozen=True)
class SourceConfig:
    id: str
    type: str
    format_version: int
    sample_rows: int
    sample_period_ms: int
    elapsed_semantics: str
    scan_semantics: str
    data_path: str
    resolved_data_path: Path

    def config_fingerprint(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "format_version": self.format_version,
            "sample_rows": self.sample_rows,
            "sample_period_ms": self.sample_period_ms,
            "elapsed_semantics": self.elapsed_semantics,
            "scan_semantics": self.scan_semantics,
        }

    def fingerprint_sha256(self) -> str:
        return sha256_bytes(canonical_bytes(self.config_fingerprint()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "format_version": self.format_version,
            "sample_rows": self.sample_rows,
            "sample_period_ms": self.sample_period_ms,
            "elapsed_semantics": self.elapsed_semantics,
            "scan_semantics": self.scan_semantics,
            "data_path": self.data_path,
        }

    def with_data_path(self, path: Path, workspace: Path) -> "SourceConfig":
        try:
            display = path.resolve().relative_to(workspace.resolve()).as_posix()
        except ValueError:
            display = str(path.resolve())
        return replace(self, data_path=display, resolved_data_path=path.resolve())


@dataclass(frozen=True)
class DatasetConfig:
    query_count: int
    seed: int
    slow_elapsed_ms: int
    slow_scanned_rows: int
    slow_rule: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_count": self.query_count,
            "seed": self.seed,
            "slow_thresholds": {
                "elapsed_ms": self.slow_elapsed_ms,
                "scanned_rows": self.slow_scanned_rows,
                "rule": self.slow_rule,
            },
        }


@dataclass(frozen=True)
class MigrationConfig:
    replay_count: int
    reducer: str
    trim_percent: int
    min_evidence: int
    value_tolerance_pct: int
    pairs: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "replay_count": self.replay_count,
            "reducer": self.reducer,
            "trim_percent": self.trim_percent,
            "min_evidence": self.min_evidence,
            "value_tolerance_pct": self.value_tolerance_pct,
            "pairs": [{"from": a, "to": b} for a, b in self.pairs],
        }


@dataclass(frozen=True)
class WorkbenchConfig:
    schema: str
    dataset: DatasetConfig
    migration: MigrationConfig
    sources: tuple[SourceConfig, ...]
    workspace: Path

    def sorted_sources(self) -> tuple[SourceConfig, ...]:
        return tuple(sorted(self.sources, key=lambda item: item.id))

    def source_by_id(self, source_id: str) -> SourceConfig:
        for source in self.sources:
            if source.id == source_id:
                return source
        raise KeyError(source_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "dataset": self.dataset.to_dict(),
            "migration": self.migration.to_dict(),
            "sources": [item.to_dict() for item in self.sources],
        }

    def policy_fingerprint(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset.to_dict(),
            "migration": self.migration.to_dict(),
        }

    def fingerprint_sha256(self) -> str:
        return sha256_bytes(canonical_bytes(self.policy_fingerprint()))


def _expect_object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise source_error("expected a JSON object", path)
    return value


def _expect_int(value: Any, path: str, *, positive: bool = False) -> int:
    if not true_int(value):
        raise source_error("expected an integer (booleans are not accepted)", path)
    if positive and value <= 0:
        raise source_error("expected a positive integer", path)
    if not positive and value < 0:
        raise source_error("expected a non-negative integer", path)
    return value


def _expect_str(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise source_error("expected a non-empty string", path)
    return value.strip()


def _validate_source(
    raw: Any,
    index: int,
    workspace: Path,
    used_ids: dict[str, int],
    used_paths: dict[str, int],
) -> SourceConfig:
    prefix = f"sources[{index}]"
    obj = _expect_object(raw, prefix)

    source_id = _expect_str(obj.get("id"), f"{prefix}.id")
    if not SOURCE_ID_RE.fullmatch(source_id):
        raise source_error(
            "source id must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}",
            f"{prefix}.id",
        )
    if source_id in used_ids:
        raise source_error(
            f"duplicate source id; first declared at sources[{used_ids[source_id]}]",
            f"{prefix}.id",
            source_id=source_id,
        )

    raw_type = _expect_str(obj.get("type"), f"{prefix}.type")
    source_type = TYPE_ALIASES.get(raw_type)
    if source_type is None:
        raise source_error(
            f"unsupported source type; expected one of {', '.join(SOURCE_TYPES)}",
            f"{prefix}.type",
            source_id=source_id,
        )

    format_version = _expect_int(
        obj.get("format_version"), f"{prefix}.format_version", positive=True
    )
    if format_version not in KNOWN_VERSIONS[source_type]:
        raise source_error(
            f"unknown declared format version; supported: "
            f"{', '.join(str(v) for v in KNOWN_VERSIONS[source_type])}",
            f"{prefix}.format_version",
            source_id=source_id,
        )

    sample_rows = _expect_int(
        obj.get("sample_rows"), f"{prefix}.sample_rows", positive=True
    )
    sample_period_ms = _expect_int(
        obj.get("sample_period_ms"), f"{prefix}.sample_period_ms", positive=True
    )

    elapsed_semantics = _expect_str(
        obj.get("elapsed_semantics"), f"{prefix}.elapsed_semantics"
    )
    if elapsed_semantics not in ALLOWED_ELAPSED_SEMANTICS[source_type]:
        raise source_error(
            "illegal elapsed semantics for source type; allowed: "
            + ", ".join(ALLOWED_ELAPSED_SEMANTICS[source_type]),
            f"{prefix}.elapsed_semantics",
            source_id=source_id,
        )

    scan_semantics = _expect_str(obj.get("scan_semantics"), f"{prefix}.scan_semantics")
    if scan_semantics not in ALLOWED_SCAN_SEMANTICS[source_type]:
        raise source_error(
            "illegal scan semantics for source type; allowed: "
            + ", ".join(ALLOWED_SCAN_SEMANTICS[source_type]),
            f"{prefix}.scan_semantics",
            source_id=source_id,
        )

    default_path = f"sources/{source_id}/records.sqlite3"
    display_path = obj.get("data_path", default_path)
    if not isinstance(display_path, str) or not display_path.strip():
        raise source_error(
            "expected a non-empty data path", f"{prefix}.data_path", source_id=source_id
        )
    display_path = display_path.strip().replace("\\", "/")
    candidate = Path(display_path)
    if not candidate.is_absolute():
        candidate = workspace / candidate
    resolved = candidate.resolve(strict=False)
    workspace_resolved = workspace.resolve(strict=False)
    try:
        resolved.relative_to(workspace_resolved)
    except ValueError:
        raise source_error(
            "data_path must remain inside the workspace",
            f"{prefix}.data_path",
            source_id=source_id,
        ) from None
    path_key = str(resolved).casefold()
    if path_key in used_paths:
        raise source_error(
            f"data_path collides with sources[{used_paths[path_key]}]",
            f"{prefix}.data_path",
            source_id=source_id,
        )
    reserved_dirs = (workspace_resolved / "ledger", workspace_resolved / "snapshots")
    for reserved in reserved_dirs:
        try:
            resolved.relative_to(reserved.resolve(strict=False))
            raise source_error(
                "data_path must not be inside a reserved ledger/snapshot directory",
                f"{prefix}.data_path",
                source_id=source_id,
            )
        except ValueError:
            pass
    if resolved.exists() and resolved.is_dir():
        raise source_error(
            "data_path points to a directory, not a SQLite file",
            f"{prefix}.data_path",
            source_id=source_id,
        )

    used_ids[source_id] = index
    used_paths[path_key] = index
    try:
        relative_display = resolved.relative_to(workspace_resolved).as_posix()
    except ValueError:
        relative_display = resolved.as_posix()
    return SourceConfig(
        id=source_id,
        type=source_type,
        format_version=format_version,
        sample_rows=sample_rows,
        sample_period_ms=sample_period_ms,
        elapsed_semantics=elapsed_semantics,
        scan_semantics=scan_semantics,
        data_path=relative_display,
        resolved_data_path=resolved,
    )


def _parse_dataset(raw: Any) -> DatasetConfig:
    if raw is None:
        raw = {}
    obj = _expect_object(raw, "dataset")
    query_count = _expect_int(
        obj.get("query_count", 8), "dataset.query_count", positive=True
    )
    seed = _expect_int(obj.get("seed", 20261122), "dataset.seed")
    thresholds = obj.get("slow_thresholds", {})
    if thresholds is None:
        thresholds = {}
    thresholds = _expect_object(thresholds, "dataset.slow_thresholds")
    slow_elapsed = _expect_int(
        thresholds.get("elapsed_ms", 250),
        "dataset.slow_thresholds.elapsed_ms",
        positive=True,
    )
    slow_scans = _expect_int(
        thresholds.get("scanned_rows", 12000),
        "dataset.slow_thresholds.scanned_rows",
        positive=True,
    )
    rule = thresholds.get("rule", "either")
    if rule not in ("either", "both"):
        raise source_error(
            "slow threshold rule must be 'either' or 'both'",
            "dataset.slow_thresholds.rule",
        )
    return DatasetConfig(query_count, seed, slow_elapsed, slow_scans, rule)


def _parse_migration(
    raw: Any, source_ids: tuple[str, ...]
) -> MigrationConfig:
    if raw is None:
        raw = {}
    obj = _expect_object(raw, "migration")
    replay_count = _expect_int(
        obj.get("replay_count", 3), "migration.replay_count", positive=True
    )
    reducer = obj.get("reducer", "median")
    if reducer not in ("median", "trimmed_mean"):
        raise source_error(
            "reducer must be 'median' or 'trimmed_mean'", "migration.reducer"
        )
    trim_percent = _expect_int(
        obj.get("trim_percent", 20), "migration.trim_percent"
    )
    if trim_percent > 49:
        raise source_error(
            "trim_percent must be between 0 and 49", "migration.trim_percent"
        )
    min_evidence = _expect_int(
        obj.get("min_evidence", 2), "migration.min_evidence", positive=True
    )
    tolerance = _expect_int(
        obj.get("value_tolerance_pct", 18), "migration.value_tolerance_pct"
    )
    if tolerance > 100:
        raise source_error(
            "value_tolerance_pct must be between 0 and 100",
            "migration.value_tolerance_pct",
        )

    pairs_raw = obj.get("pairs")
    pairs: list[tuple[str, str]] = []
    if pairs_raw is None:
        ids = sorted(source_ids)
        pairs = [(a, b) for a in ids for b in ids if a != b]
    else:
        if not isinstance(pairs_raw, list):
            raise source_error("migration.pairs must be an array", "migration.pairs")
        seen: set[tuple[str, str]] = set()
        for index, item in enumerate(pairs_raw):
            prefix = f"migration.pairs[{index}]"
            pair_obj = _expect_object(item, prefix)
            left = _expect_str(pair_obj.get("from"), f"{prefix}.from")
            right = _expect_str(pair_obj.get("to"), f"{prefix}.to")
            if left not in source_ids:
                raise source_error(
                    f"unknown source in migration pair: {left}", f"{prefix}.from"
                )
            if right not in source_ids:
                raise source_error(
                    f"unknown source in migration pair: {right}", f"{prefix}.to"
                )
            if left == right:
                raise source_error(
                    "migration pair must reference two distinct sources",
                    f"{prefix}.to",
                )
            if (left, right) in seen:
                raise source_error(
                    "duplicate migration pair", f"{prefix}.from"
                )
            seen.add((left, right))
            pairs.append((left, right))
        pairs.sort()
    return MigrationConfig(
        replay_count=replay_count,
        reducer=reducer,
        trim_percent=trim_percent,
        min_evidence=min_evidence,
        value_tolerance_pct=tolerance,
        pairs=tuple(pairs),
    )


def load_config_data(raw: Any, workspace: Path | str) -> WorkbenchConfig:
    workspace_path = Path(workspace).resolve(strict=False)
    if isinstance(raw, list):
        raw = {"sources": raw}
    obj = _expect_object(raw, "$")
    schema = obj.get("schema", SCHEMA_NAME)
    if schema != SCHEMA_NAME:
        raise source_error(
            f"unsupported schema; expected {SCHEMA_NAME}", "$.schema"
        )
    sources_raw = obj.get("sources")
    if not isinstance(sources_raw, list) or not sources_raw:
        raise source_error("sources must be a non-empty array", "$.sources")

    used_ids: dict[str, int] = {}
    used_paths: dict[str, int] = {}
    sources: list[SourceConfig] = []
    for index, source_raw in enumerate(sources_raw):
        sources.append(
            _validate_source(source_raw, index, workspace_path, used_ids, used_paths)
        )

    dataset = _parse_dataset(obj.get("dataset"))
    source_ids = tuple(source.id for source in sources)
    migration = _parse_migration(obj.get("migration"), source_ids)
    return WorkbenchConfig(
        schema=SCHEMA_NAME,
        dataset=dataset,
        migration=migration,
        sources=tuple(sources),
        workspace=workspace_path,
    )


def load_config(path: Path | str, workspace: Path | str) -> WorkbenchConfig:
    config_path = Path(path)
    try:
        raw_text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise source_error(
            f"cannot read source declaration: {exc}", "$", details={"file": str(path)}
        ) from exc
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise source_error(
            f"invalid JSON at line {exc.lineno}, column {exc.colno}", "$"
        ) from exc
    return load_config_data(raw, workspace)


def default_config_dict() -> dict[str, Any]:
    return {
        "schema": SCHEMA_NAME,
        "dataset": {
            "query_count": 8,
            "seed": 20261122,
            "slow_thresholds": {
                "elapsed_ms": 250,
                "scanned_rows": 12000,
                "rule": "either",
            },
        },
        "migration": {
            "replay_count": 3,
            "reducer": "median",
            "trim_percent": 20,
            "min_evidence": 2,
            "value_tolerance_pct": 18,
        },
        "sources": [
            {
                "id": "app-prod",
                "type": "app_log_summary",
                "format_version": 2,
                "sample_rows": 24,
                "sample_period_ms": 100,
                "elapsed_semantics": "p95",
                "scan_semantics": "estimated",
                "data_path": "sources/app-prod/records.sqlite3",
            },
            {
                "id": "db-prod",
                "type": DB_EXEC_STATS,
                "format_version": 2,
                "sample_rows": 24,
                "sample_period_ms": 250,
                "elapsed_semantics": "p95",
                "scan_semantics": "exact",
                "data_path": "sources/db-prod/records.sqlite3",
            },
            {
                "id": "slow-prod",
                "type": "slow_log_excerpt",
                "format_version": 2,
                "sample_rows": 24,
                "sample_period_ms": 500,
                "elapsed_semantics": "max",
                "scan_semantics": "exact",
                "data_path": "sources/slow-prod/records.sqlite3",
            },
            {
                "id": "slow-canary",
                "type": "slow_log_excerpt",
                "format_version": 1,
                "sample_rows": 6,
                "sample_period_ms": 1000,
                "elapsed_semantics": "max",
                "scan_semantics": "sampled",
                "data_path": "sources/slow-canary/records.sqlite3",
            },
        ],
    }


def dump_config(config: WorkbenchConfig) -> str:
    return json.dumps(config.to_dict(), ensure_ascii=False, indent=2) + "\n"


def iter_source_ids(config: WorkbenchConfig) -> Iterable[str]:
    for source in config.sorted_sources():
        yield source.id
