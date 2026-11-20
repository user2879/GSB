"""Strict input validation. Invalid batches are rejected before mutation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .constants import ESTIMATOR_VERSION, P95_METHOD, REPLAY_MODEL_VERSION, TRUNCATION_DIRECTION
from .errors import LogValidationError, ParameterError, SnapshotError
from .models import AggregationSpec, MasterRecord, ReplayParams
from .util import read_json


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LogValidationError(f"{field} 必须是对象", field=field)
    return value


def _strict_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LogValidationError(f"{field} 必须是整数", field=field)
    return value


def validate_master_records(raw_records: Any) -> list[MasterRecord]:
    if not isinstance(raw_records, list) or not raw_records:
        raise LogValidationError("主库日志不能为空", seq=0, field="records")
    records: list[MasterRecord] = []
    seen_txn_ids: set[str] = set()
    expected_seq = 1
    for index, raw in enumerate(raw_records):
        if not isinstance(raw, dict):
            raise LogValidationError(
                "日志记录必须是对象", seq=index + 1, field=f"records[{index}]"
            )
        try:
            seq = _strict_int(raw.get("commit_seq"), "commit_seq")
        except LogValidationError as exc:
            exc.seq = index + 1
            raise
        if seq <= 0:
            raise LogValidationError("commit_seq 必须为正整数", seq=seq, field="commit_seq")
        if seq != expected_seq:
            reason = "重复" if seq < expected_seq else "不连续"
            raise LogValidationError(
                f"commit_seq {reason}",
                seq=seq,
                field="commit_seq",
                details={"expected_seq": expected_seq, "actual_seq": seq},
            )
        txn_id = raw.get("txn_id")
        if not isinstance(txn_id, str) or not txn_id.strip():
            raise LogValidationError("txn_id 必须为非空字符串", seq=seq, field="txn_id")
        txn_id = txn_id.strip()
        if txn_id in seen_txn_ids:
            raise LogValidationError(
                "txn_id 必须唯一", seq=seq, field="txn_id", details={"txn_id": txn_id}
            )
        try:
            commit_ts_ms = _strict_int(raw.get("commit_ts_ms"), "commit_ts_ms")
            txn_bytes = _strict_int(raw.get("txn_bytes"), "txn_bytes")
        except LogValidationError as exc:
            exc.seq = seq
            raise
        if commit_ts_ms < 0:
            raise LogValidationError(
                "commit_ts_ms 必须非负", seq=seq, field="commit_ts_ms"
            )
        if txn_bytes <= 0:
            raise LogValidationError("txn_bytes 必须为正整数", seq=seq, field="txn_bytes")
        records.append(
            MasterRecord(
                commit_seq=seq,
                txn_id=txn_id,
                commit_ts_ms=commit_ts_ms,
                txn_bytes=txn_bytes,
            )
        )
        seen_txn_ids.add(txn_id)
        expected_seq += 1
    return records


def load_master_records(path: str | Path) -> list[MasterRecord]:
    try:
        document = read_json(path)
    except Exception as exc:  # noqa: BLE001 - translate parser errors to E_LOG
        raise LogValidationError(
            f"无法读取日志 JSON: {exc}", seq=0, field="file", details={"path": str(path)}
        ) from exc
    if isinstance(document, dict):
        document = document.get("records")
    return validate_master_records(document)


def _config_int(mapping: dict[str, Any], key: str, field: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ParameterError(f"{field} 必须是整数", field=field)
    return value


def validate_aggregation_spec(raw: Any) -> AggregationSpec:
    if not isinstance(raw, dict):
        raise ParameterError("aggregation 必须是对象", field="aggregation")
    window_size = _config_int(raw, "window_size", "aggregation.window_size")
    sample_cap = _config_int(raw, "max_samples_per_window", "aggregation.max_samples_per_window")
    min_samples = _config_int(raw, "min_samples", "aggregation.min_samples")
    ceiling = _config_int(raw, "latency_ceiling_ms", "aggregation.latency_ceiling_ms")
    scale_power = _config_int(raw, "scale_power", "aggregation.scale_power")
    threshold = _config_int(
        raw, "dominant_threshold_pct", "aggregation.dominant_threshold_pct"
    )
    p95_method = raw.get("p95_method", P95_METHOD)
    truncation = raw.get("truncation_direction", TRUNCATION_DIRECTION)
    version = raw.get("version", ESTIMATOR_VERSION)
    if window_size <= 0:
        raise ParameterError("window_size 必须为正整数", field="aggregation.window_size")
    if sample_cap <= 0 or sample_cap > window_size:
        raise ParameterError(
            "max_samples_per_window 必须在 1..window_size 之间",
            field="aggregation.max_samples_per_window",
        )
    if min_samples <= 0:
        raise ParameterError("min_samples 必须为正整数", field="aggregation.min_samples")
    if ceiling < 0:
        raise ParameterError("latency_ceiling_ms 必须非负", field="aggregation.latency_ceiling_ms")
    if scale_power < 0 or scale_power > 12:
        raise ParameterError("scale_power 必须在 0..12", field="aggregation.scale_power")
    if threshold <= 0 or threshold > 100:
        raise ParameterError(
            "dominant_threshold_pct 必须在 1..100", field="aggregation.dominant_threshold_pct"
        )
    if p95_method != P95_METHOD:
        raise ParameterError(
            f"不支持的 p95_method: {p95_method!r}", field="aggregation.p95_method"
        )
    if truncation != TRUNCATION_DIRECTION:
        raise ParameterError(
            f"不支持的 truncation_direction: {truncation!r}",
            field="aggregation.truncation_direction",
        )
    if version != ESTIMATOR_VERSION:
        raise ParameterError(
            f"估算口径版本不一致: {version!r}", field="aggregation.version"
        )
    return AggregationSpec(
        window_size=window_size,
        max_samples_per_window=sample_cap,
        min_samples=min_samples,
        latency_ceiling_ms=ceiling,
        scale_power=scale_power,
        p95_method=p95_method,
        truncation_direction=truncation,
        dominant_threshold_pct=threshold,
        version=version,
    )


def validate_replay_params(raw: Any, spec: AggregationSpec) -> ReplayParams:
    if not isinstance(raw, dict):
        raise ParameterError("replay 必须是对象", field="replay")
    rate = _config_int(raw, "rate_rps", "replay.rate_rps")
    concurrency = _config_int(raw, "concurrency", "replay.concurrency")
    base_ms = _config_int(raw, "clock_base_ms", "replay.clock_base_ms")
    clock_mode = raw.get("clock_mode", "virtual_monotonic_ns")
    model_version = raw.get("model_version", REPLAY_MODEL_VERSION)
    if rate <= 0:
        raise ParameterError("rate_rps 必须为正整数", field="replay.rate_rps")
    if concurrency <= 0:
        raise ParameterError("concurrency 必须为正整数", field="replay.concurrency")
    if base_ms < 0:
        raise ParameterError("clock_base_ms 必须非负", field="replay.clock_base_ms")
    if clock_mode != "virtual_monotonic_ns":
        raise ParameterError(
            f"不支持的 clock_mode: {clock_mode!r}", field="replay.clock_mode"
        )
    if model_version != REPLAY_MODEL_VERSION:
        raise ParameterError(
            f"回放模型版本不一致: {model_version!r}", field="replay.model_version"
        )
    return ReplayParams(
        rate_rps=rate,
        concurrency=concurrency,
        clock_mode=clock_mode,
        clock_base_ms=base_ms,
        estimator_digest=spec.digest,
    )


def load_replay_config(path: str | Path) -> tuple[ReplayParams, AggregationSpec, dict[str, Any]]:
    try:
        document = read_json(path)
    except Exception as exc:  # noqa: BLE001 - report as E_PARAM
        raise ParameterError(
            f"无法读取回放配置: {exc}", field="file", details={"path": str(path)}
        ) from exc
    if not isinstance(document, dict):
        raise ParameterError("配置文件根节点必须是对象", field="file")
    spec = validate_aggregation_spec(document.get("aggregation"))
    params = validate_replay_params(document.get("replay"), spec)
    return params, spec, document


def validate_progress(last_replayed_seq: int, last_log_seq: int) -> None:
    if isinstance(last_replayed_seq, bool) or not isinstance(last_replayed_seq, int):
        raise LogValidationError("回放游标必须是整数", seq=None, field="last_replayed_seq")
    if last_replayed_seq < 0:
        raise LogValidationError("回放游标必须非负", seq=last_replayed_seq, field="last_replayed_seq")
    if last_replayed_seq > last_log_seq:
        raise LogValidationError(
            "回放进度不得超前于日志末尾",
            seq=last_replayed_seq,
            field="last_replayed_seq",
            details={"last_log_seq": last_log_seq},
        )


def validate_snapshot_envelope(document: Any, expected_format: int) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise SnapshotError("快照根节点必须是对象", field="snapshot")
    fmt = document.get("snapshot_format_version")
    if fmt != expected_format:
        raise SnapshotError(
            "快照格式版本不兼容",
            field="snapshot_format_version",
            details={"expected": expected_format, "actual": fmt},
        )
    required = (
        "db_schema_version",
        "log",
        "run",
        "replay_params",
        "estimator",
        "progress",
        "batches",
        "samples",
        "estimation",
    )
    missing = [key for key in required if key not in document]
    if missing:
        raise SnapshotError(
            "快照字段缺失", field=missing[0], details={"missing_fields": missing}
        )
    return document
