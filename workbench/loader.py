from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import RuntimeConfig, patch_config, validate_config
from .errors import ConfigError, PlanError, ShardError
from .model import (
    CandidateRow,
    LoadResult,
    MaterializedInput,
    Shard,
    ShardDescriptor,
    ValidationFailure,
)
from .util import canonical_json, file_hash, is_finite_number, read_json, stable_hash


SHARD_FIELDS = (
    "request_id",
    "channel",
    "item_id",
    "score",
    "features",
    "shard_seq",
    "coverage_start_ms",
    "coverage_end_ms",
)


def discover_base_shards(samples_dir: str | Path) -> list[ShardDescriptor]:
    root = Path(samples_dir)
    shard_root = root / "shards"
    if not shard_root.exists():
        raise ShardError(f"shard directory does not exist: {shard_root}")
    paths = sorted(
        (path for path in shard_root.rglob("*.jsonl") if path.is_file()),
        key=lambda path: path.as_posix(),
    )
    if not paths:
        raise ShardError(f"no JSONL shards found under {shard_root}")
    return [
        ShardDescriptor(
            path=str(path.resolve()),
            origin="base",
            **_peek_meta(path),
        )
        for path in paths
    ]


def _peek_meta(path: str | Path) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            first = handle.readline()
        row = json.loads(first)
        channel = row.get("channel")
        shard_seq = row.get("shard_seq")
        start = row.get("coverage_start_ms")
        end = row.get("coverage_end_ms")
        if isinstance(channel, str) and isinstance(shard_seq, int):
            return {
                "channel": channel,
                "shard_seq": shard_seq,
                "coverage_start_ms": start if isinstance(start, int) else None,
                "coverage_end_ms": end if isinstance(end, int) else None,
            }
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    return {}


def load_plan(samples_dir: str | Path) -> dict[str, Any]:
    path = Path(samples_dir) / "roll_plan.json"
    if not path.exists():
        return {"version": 1, "steps": []}
    plan = read_json(path)
    if not isinstance(plan, dict) or not isinstance(plan.get("steps"), list):
        raise PlanError("roll_plan.json must contain a steps array")
    seen: set[int] = set()
    for step in plan["steps"]:
        if not isinstance(step, dict) or not isinstance(step.get("step"), int):
            raise PlanError("each plan step must have an integer step")
        if step["step"] in seen:
            raise PlanError(f"duplicate plan step {step['step']}")
        seen.add(step["step"])
        ops = step.get("operations")
        if not isinstance(ops, list):
            raise PlanError(f"plan step {step['step']} must contain operations")
        for operation in ops:
            if not isinstance(operation, dict) or operation.get("op") not in {
                "append",
                "replace",
                "revoke",
                "update_config",
            }:
                raise PlanError(f"invalid operation in step {step['step']}: {operation!r}")
    plan["steps"] = sorted(plan["steps"], key=lambda item: item["step"])
    return plan


def _resolve_source(samples_dir: str | Path, source: str) -> str:
    path = Path(source)
    if not path.is_absolute():
        path = Path(samples_dir) / path
    return str(path.resolve())


def materialize_input(
    samples_dir: str | Path,
    up_to_step: int | None = None,
    *,
    config_override: str | Path | None = None,
) -> MaterializedInput:
    samples = Path(samples_dir)
    plan = load_plan(samples)
    config_path = Path(config_override) if config_override else samples / "rerank_config.json"
    if not config_path.exists():
        raise ConfigError(f"config file does not exist: {config_path}")
    config = read_json(config_path)
    if not isinstance(config, dict):
        raise ConfigError("config file must contain a JSON object")
    validate_config(config)
    active = discover_base_shards(samples)
    applied_steps: list[int] = []
    operations: list[dict[str, Any]] = []
    available = [step["step"] for step in plan["steps"]]
    if (
        up_to_step is not None
        and available
        and up_to_step not in available
        and up_to_step >= min(available)
    ):
        raise PlanError(f"unknown plan step {up_to_step}; available={available}")
    for step in plan["steps"]:
        if up_to_step is not None and step["step"] > up_to_step:
            break
        for operation_index, operation in enumerate(step["operations"]):
            op = operation["op"]
            record = {
                "step": step["step"],
                "operation_index": operation_index,
                "op": op,
            }
            if op == "append":
                source = operation.get("source")
                if not isinstance(source, str) or not source:
                    raise PlanError(f"append in step {step['step']} requires source")
                path = _resolve_source(samples, source)
                descriptor = ShardDescriptor(
                    path=path,
                    injection_id=operation.get("injection_id"),
                    origin=f"step:{step['step']}",
                    **_peek_meta(path),
                )
                active.append(descriptor)
                record.update({"source": path, "injection_id": descriptor.injection_id})
            elif op == "replace":
                source = operation.get("source")
                channel = operation.get("channel")
                start = operation.get("start_ms")
                end = operation.get("end_ms")
                if not isinstance(source, str) or not isinstance(channel, str):
                    raise PlanError(f"replace in step {step['step']} requires channel and source")
                if not isinstance(start, int) or not isinstance(end, int) or start >= end:
                    raise PlanError(f"replace in step {step['step']} has invalid interval")
                removed = []
                kept = []
                for descriptor in active:
                    same_channel = descriptor.channel == channel
                    overlaps = (
                        descriptor.coverage_start_ms is not None
                        and descriptor.coverage_end_ms is not None
                        and descriptor.coverage_start_ms < end
                        and start < descriptor.coverage_end_ms
                    )
                    if same_channel and overlaps:
                        removed.append(descriptor.path)
                    else:
                        kept.append(descriptor)
                active = kept
                path = _resolve_source(samples, source)
                active.append(
                    ShardDescriptor(
                        path=path,
                        injection_id=operation.get("injection_id"),
                        origin=f"step:{step['step']}",
                        **_peek_meta(path),
                    )
                )
                record.update(
                    {
                        "channel": channel,
                        "start_ms": start,
                        "end_ms": end,
                        "source": path,
                        "removed": sorted(removed),
                        "injection_id": operation.get("injection_id"),
                    }
                )
            elif op == "revoke":
                channel = operation.get("channel")
                shard_seq = operation.get("shard_seq")
                if not isinstance(channel, str) or not isinstance(shard_seq, int):
                    raise PlanError(f"revoke in step {step['step']} requires channel and shard_seq")
                removed = []
                kept = []
                for descriptor in active:
                    if descriptor.channel == channel and descriptor.shard_seq == shard_seq:
                        removed.append(descriptor.path)
                    else:
                        kept.append(descriptor)
                active = kept
                record.update(
                    {
                        "channel": channel,
                        "shard_seq": shard_seq,
                        "removed": sorted(removed),
                    }
                )
            elif op == "update_config":
                patch = operation.get("patch")
                if not isinstance(patch, dict):
                    raise PlanError(f"update_config in step {step['step']} requires object patch")
                config = patch_config(config, patch)
                validate_config(config)
                record.update({"patch": patch})
            operations.append(record)
        applied_steps.append(step["step"])
    active.sort(key=lambda descriptor: descriptor.path)
    source_versions = []
    for descriptor in active:
        try:
            digest = file_hash(descriptor.path)
        except OSError:
            digest = "missing"
        source_versions.append({"path": descriptor.path, "hash": digest})
    return MaterializedInput(
        descriptors=active,
        config_raw=config,
        applied_steps=applied_steps,
        operations=operations,
        effective_plan_hash=stable_hash(
            {
                "config": config,
                "applied_steps": applied_steps,
                "operations": operations,
                "active_sources": [item.path for item in active],
                "source_versions": source_versions,
            }
        ),
    )


def _parse_file(
    descriptor: ShardDescriptor,
    cfg: RuntimeConfig,
) -> tuple[list[Shard], list[ValidationFailure]]:
    path = descriptor.path
    failures: list[ValidationFailure] = []
    groups: dict[tuple[str, int], list[CandidateRow]] = {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line_no, raw_line in enumerate(handle, 1):
                if not raw_line.strip():
                    continue
                try:
                    row = json.loads(raw_line)
                except (json.JSONDecodeError, TypeError) as exc:
                    failures.append(
                        ValidationFailure(
                            code=ShardError.code,
                            source=path,
                            reason=f"invalid JSON: {exc}",
                            line_no=line_no,
                            injection_id=descriptor.injection_id,
                        )
                    )
                    continue
                try:
                    candidate = _validate_row(row, cfg, path, line_no, descriptor.injection_id)
                except ValueError as exc:
                    failures.append(
                        ValidationFailure(
                            code=ShardError.code,
                            source=path,
                            reason=str(exc),
                            shard_id=(
                                f"{row.get('channel')}#{row.get('shard_seq')}"
                                if isinstance(row, dict)
                                else None
                            ),
                            line_no=line_no,
                            injection_id=descriptor.injection_id,
                        )
                    )
                    continue
                groups.setdefault((candidate.channel, candidate.shard_seq), []).append(candidate)
    except OSError as exc:
        failures.append(
            ValidationFailure(
                code=ShardError.code,
                source=path,
                reason=f"cannot read shard: {exc}",
                injection_id=descriptor.injection_id,
            )
        )
        return [], failures
    if failures:
        for (channel, shard_seq) in sorted(groups):
            failures.append(
                ValidationFailure(
                    code=ShardError.code,
                    source=path,
                    reason="shard contains invalid rows",
                    shard_id=f"{channel}#{shard_seq}",
                    injection_id=descriptor.injection_id,
                )
            )
        return [], failures
    shards: list[Shard] = []
    for (channel, shard_seq), rows in sorted(groups.items()):
        rows.sort(
            key=lambda row: (
                row.request_id,
                row.item_id,
                row.score,
                row.line_no,
            )
        )
        starts = {row.coverage_start_ms for row in rows}
        ends = {row.coverage_end_ms for row in rows}
        if len(starts) != 1 or len(ends) != 1:
            failures.append(
                ValidationFailure(
                    code=ShardError.code,
                    source=path,
                    reason="coverage interval differs within shard",
                    shard_id=f"{channel}#{shard_seq}",
                    injection_id=descriptor.injection_id,
                )
            )
            continue
        start = next(iter(starts))
        end = next(iter(ends))
        shard = Shard(
            shard_id=f"{channel}#{shard_seq}",
            channel=channel,
            shard_seq=shard_seq,
            coverage_start_ms=start,
            coverage_end_ms=end,
            rows=rows,
            fingerprint=stable_hash(
                {
                    "shard_id": f"{channel}#{shard_seq}",
                    "rows": [row.semantic_dict() for row in rows],
                }
            ),
            source=path,
            injection_id=descriptor.injection_id,
        )
        shards.append(shard)
    if not shards and not failures:
        failures.append(
            ValidationFailure(
                code=ShardError.code,
                source=path,
                reason="shard is empty",
                injection_id=descriptor.injection_id,
            )
        )
    return shards, failures


def _validate_row(
    row: Any,
    cfg: RuntimeConfig,
    source: str,
    line_no: int,
    injection_id: str | None,
) -> CandidateRow:
    if not isinstance(row, dict):
        raise ValueError("row must be a JSON object")
    missing = [field for field in SHARD_FIELDS if field not in row]
    if missing:
        raise ValueError(f"missing fields: {', '.join(missing)}")
    request_id = row["request_id"]
    channel = row["channel"]
    item_id = row["item_id"]
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("request_id must be a non-empty string")
    if not isinstance(channel, str) or not channel.strip():
        raise ValueError("channel must be a non-empty string")
    if not isinstance(item_id, str) or not item_id.strip():
        raise ValueError("item_id must be a non-empty string")
    shard_seq = row["shard_seq"]
    if not isinstance(shard_seq, int) or isinstance(shard_seq, bool) or shard_seq < 0:
        raise ValueError("shard_seq must be a non-negative integer")
    start = row["coverage_start_ms"]
    end = row["coverage_end_ms"]
    if not isinstance(start, int) or isinstance(start, bool):
        raise ValueError("coverage_start_ms must be an integer")
    if not isinstance(end, int) or isinstance(end, bool):
        raise ValueError("coverage_end_ms must be an integer")
    if start < 0 or end <= start:
        raise ValueError("coverage interval must satisfy 0 <= start < end")
    if start % cfg.alignment_ms or end % cfg.alignment_ms:
        raise ValueError(
            f"coverage interval must align to {cfg.alignment_ms} ms"
        )
    score = row["score"]
    if not is_finite_number(score):
        raise ValueError("score must be a finite number")
    features = row["features"]
    if not isinstance(features, list) or len(features) != cfg.feature_dim:
        raise ValueError(f"features must contain exactly {cfg.feature_dim} numbers")
    if any(not is_finite_number(value) for value in features):
        raise ValueError("features must contain only finite numbers")
    return CandidateRow(
        request_id=request_id,
        channel=channel,
        item_id=item_id,
        score=float(score),
        features=tuple(float(value) for value in features),
        shard_seq=shard_seq,
        coverage_start_ms=start,
        coverage_end_ms=end,
        line_no=line_no,
        source=source,
        injection_id=injection_id,
    )


def load_shards(
    materialized: MaterializedInput,
    cfg: RuntimeConfig,
    samples_dir: str | Path,
) -> LoadResult:
    shards: list[Shard] = []
    failures: list[ValidationFailure] = []
    for descriptor in materialized.descriptors:
        parsed, errors = _parse_file(descriptor, cfg)
        shards.extend(parsed)
        failures.extend(errors)

    by_identity: dict[tuple[str, int], list[Shard]] = {}
    for shard in shards:
        by_identity.setdefault((shard.channel, shard.shard_seq), []).append(shard)
    duplicate_ids: set[str] = set()
    for (channel, shard_seq), group in sorted(by_identity.items()):
        sources = sorted({shard.source for shard in group})
        if len(sources) > 1:
            duplicate_ids.add(f"{channel}#{shard_seq}")
            for source in sources:
                failures.append(
                    ValidationFailure(
                        code=ShardError.code,
                        source=source,
                        reason=f"shard_seq {shard_seq} is duplicated in channel {channel}",
                        shard_id=f"{channel}#{shard_seq}",
                    )
                )
    shards = [shard for shard in shards if shard.shard_id not in duplicate_ids]

    overlap_ids: set[str] = set()
    by_channel: dict[str, list[Shard]] = {}
    for shard in shards:
        by_channel.setdefault(shard.channel, []).append(shard)
    for channel, channel_shards in sorted(by_channel.items()):
        ordered = sorted(
            channel_shards,
            key=lambda shard: (shard.coverage_start_ms, shard.coverage_end_ms, shard.shard_seq),
        )
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                if right.coverage_start_ms >= left.coverage_end_ms:
                    break
                overlap_ids.add(left.shard_id)
                overlap_ids.add(right.shard_id)
                failures.append(
                    ValidationFailure(
                        code=ShardError.code,
                        source=left.source,
                        reason=(
                            f"coverage overlaps {right.shard_id} in channel {channel}: "
                            f"[{left.coverage_start_ms},{left.coverage_end_ms}) vs "
                            f"[{right.coverage_start_ms},{right.coverage_end_ms})"
                        ),
                        shard_id=left.shard_id,
                        injection_id=left.injection_id,
                    )
                )
    shards = [shard for shard in shards if shard.shard_id not in overlap_ids]
    for shard in shards:
        if shard.fingerprint != stable_hash(
            {
                "shard_id": shard.shard_id,
                "rows": [row.semantic_dict() for row in shard.rows],
            }
        ):
            raise AssertionError("internal shard fingerprint mismatch")
    shards.sort(
        key=lambda shard: (
            shard.channel,
            shard.coverage_start_ms,
            shard.coverage_end_ms,
            shard.shard_seq,
            shard.source,
        )
    )
    failures.sort(
        key=lambda failure: (
            failure.source,
            failure.shard_id or "",
            failure.line_no if failure.line_no is not None else -1,
            failure.reason,
            failure.code,
        )
    )
    attributes, attr_failures, attr_fp = load_attributes(samples_dir)
    return LoadResult(
        shards=shards,
        failures=failures,
        attributes=attributes,
        attributes_fingerprint=attr_fp,
        attributes_failures=attr_failures,
    )


def load_attributes(samples_dir: str | Path) -> tuple[dict[str, dict[str, Any]], list[ValidationFailure], str]:
    path = Path(samples_dir) / "items.json"
    if not path.exists():
        raise ConfigError(f"item attribute table does not exist: {path}")
    raw = read_json(path)
    failures: list[ValidationFailure] = []
    attributes: dict[str, dict[str, Any]] = {}
    if isinstance(raw, dict):
        entries = [{"item_id": key, **(value if isinstance(value, dict) else {})} for key, value in raw.items()]
    elif isinstance(raw, list):
        entries = raw
    else:
        raise ConfigError("items.json must be an object or array")
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            failures.append(
                ValidationFailure(
                    code="E_ITEM_ATTRIBUTE_INVALID",
                    source=str(path),
                    reason="attribute entry must be an object",
                    line_no=index + 1,
                )
            )
            continue
        item_id = entry.get("item_id")
        category = entry.get("category")
        price_band = entry.get("price_band")
        if (
            not isinstance(item_id, str)
            or not item_id
            or not isinstance(category, str)
            or not category
            or not isinstance(price_band, str)
            or not price_band
        ):
            failures.append(
                ValidationFailure(
                    code="E_ITEM_ATTRIBUTE_INVALID",
                    source=str(path),
                    reason="item_id, category and price_band must be non-empty strings",
                    line_no=index + 1,
                )
            )
            continue
        attributes[item_id] = {"category": category, "price_band": price_band}
    failures.sort(key=lambda failure: (failure.source, failure.line_no or -1, failure.reason))
    return attributes, failures, stable_hash(attributes)


def initial_sources_hash(samples_dir: str | Path) -> str:
    samples = Path(samples_dir)
    managed = [
        samples / "rerank_config.json",
        samples / "items.json",
        samples / "roll_plan.json",
        samples / "injections.json",
    ]
    managed.extend(sorted((samples / "shards").rglob("*.jsonl")))
    return stable_hash(
        [
            {
                "path": path.relative_to(samples).as_posix(),
                "hash": file_hash(path),
            }
            for path in managed
            if path.exists()
        ]
    )
