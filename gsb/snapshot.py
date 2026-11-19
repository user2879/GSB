from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import GSBError
from .generation import scenario_data_digest
from .model import parse_scenario
from .session import Session
from .util import atomic_write_text, canonical_json, sha256_json


SNAPSHOT_FORMAT = "gsb-snapshot-v1"


def _bucket_lower(payload: dict[str, Any]) -> Any:
    buckets = payload.get("buckets") or []
    if buckets:
        return buckets[0].get("lower")
    return None


def save_snapshot(session: Session, path: str | Path) -> dict[str, Any]:
    if session.current is None:
        session.run()
    assert session.current is not None
    result = session.current
    payload = {
        "format": SNAPSHOT_FORMAT,
        "scenario": session.spec.name,
        "spec": session.spec.to_dict(),
        "spec_digest": session.spec.digest,
        "parameter_version": session.parameter_version,
        "parameters": {
            **session.spec.join.version_parameters,
            "top_k": session.spec.join.top_k,
            "parameter_version": session.parameter_version,
        },
        "data": {
            "scenario_data_digest": scenario_data_digest(session.samples),
            "tables": {
                name: {
                    "content_digest": sample.content_digest,
                    "file_digest": sample.file_digest,
                    "rows": sample.rows,
                }
                for name, sample in sorted(session.samples.items())
            },
        },
        "histograms": {
            "left": result.histograms["left"].to_dict(),
            "right": result.histograms["right"].to_dict(),
        },
        "sample_ledgers": {
            "left": result.ledgers["left"].to_dict(include_entries=False),
            "right": result.ledgers["right"].to_dict(include_entries=False),
        },
        "estimates": {
            "histogram": {
                "estimate_fixed": result.histogram_estimate["estimate_fixed"],
                "estimate": result.histogram_estimate["estimate"],
                "overlap_pair_count": result.histogram_estimate["overlap_pair_count"],
                "quantization_error_upper": result.histogram_estimate["quantization_error_upper"],
                "path_digest": result.histogram_estimate["path_digest"],
            },
            "sampling": {
                "estimate_fixed": result.sampling_estimate["estimate_fixed"],
                "estimate": result.sampling_estimate["estimate"],
                "sample_hit_pairs": result.sampling_estimate["sample_hit_pairs"],
                "quantization_error_upper": result.sampling_estimate["quantization_error_upper"],
            },
            "truth_rows": result.truth["row_count"],
        },
        "attribution": {
            "digest": result.attribution["digest"],
            "pair_count": result.attribution["pair_count"],
            "top_k": result.attribution["top_k"],
        },
        "decision": dict(result.decision),
        "truth": {
            "digest": result.truth["digest"],
            "row_count": result.truth["row_count"],
            "common_key_count": result.truth["common_key_count"],
            "pair_rows": result.truth["pair_rows"],
            "left_bucket_rows": result.truth["left_bucket_rows"],
            "right_bucket_rows": result.truth["right_bucket_rows"],
        },
        "analysis_digest": result.analysis_digest,
    }
    payload["snapshot_digest"] = sha256_json(payload)
    atomic_write_text(path, canonical_json(payload, pretty=True))
    session.logger.log(
        "INFO",
        "OK",
        "snapshot_saved",
        {
            "path": str(path),
            "parameter_version": session.parameter_version,
            "snapshot_digest": payload["snapshot_digest"],
        },
    )
    return {
        "path": str(path),
        "snapshot_digest": payload["snapshot_digest"],
        "parameter_version": session.parameter_version,
        "analysis_digest": result.analysis_digest,
    }


def load_snapshot_payload(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise GSBError("E_VERSION", "无法读取快照", {"path": str(source), "reason": str(exc)}) from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GSBError(
            "E_VERSION",
            "快照不是合法 JSON",
            {"path": str(source), "line": exc.lineno, "column": exc.colno},
        ) from exc
    if not isinstance(payload, dict):
        raise GSBError("E_VERSION", "快照根节点必须为对象", {"path": str(source)})
    if payload.get("format") != SNAPSHOT_FORMAT:
        raise GSBError("E_VERSION", "不支持的快照格式", {"format": payload.get("format")})
    for field in ("parameter_version", "spec", "histograms", "sample_ledgers", "estimates", "decision", "analysis_digest"):
        if field not in payload:
            raise GSBError("E_VERSION", "快照缺少关键字段", {"path": str(source), "field": field})
    snapshot_digest = payload.get("snapshot_digest")
    if snapshot_digest is not None:
        content = dict(payload)
        content.pop("snapshot_digest", None)
        actual_digest = sha256_json(content)
        payload["_snapshot_digest_valid"] = actual_digest == snapshot_digest
        payload["_snapshot_digest_actual"] = actual_digest
    else:
        payload["_snapshot_digest_valid"] = None
        payload["_snapshot_digest_actual"] = None
    return payload


def _replay_error(
    message: str,
    *,
    field: str,
    expected: Any,
    actual: Any,
    bucket_lower: Any = None,
    side: str | None = None,
    extra: dict[str, Any] | None = None,
) -> GSBError:
    details = {
        "bucket_lower": bucket_lower,
        "field": field,
        "expected": expected,
        "actual": actual,
    }
    if side is not None:
        details["side"] = side
    if extra:
        details.update(extra)
    return GSBError("E_REPLAY", message, details)


def _compare_histograms(snapshot: dict[str, Any], result: Any) -> None:
    for side in ("left", "right"):
        expected = snapshot["histograms"][side]
        actual = result.histograms[side].to_dict()
        expected_buckets = expected.get("buckets", [])
        actual_buckets = actual.get("buckets", [])
        if len(expected_buckets) != len(actual_buckets):
            raise _replay_error(
                "重算桶数量不一致",
                field="bucket_count",
                expected=len(expected_buckets),
                actual=len(actual_buckets),
                bucket_lower=_bucket_lower(expected),
                side=side,
            )
        for expected_bucket, actual_bucket in zip(expected_buckets, actual_buckets):
            for field in (
                "index",
                "lower",
                "upper",
                "lower_inclusive",
                "upper_inclusive",
                "row_count",
                "null_count",
                "digest",
            ):
                if expected_bucket.get(field) != actual_bucket.get(field):
                    raise _replay_error(
                        "重算直方图桶字段不一致",
                        field=field,
                        expected=expected_bucket.get(field),
                        actual=actual_bucket.get(field),
                        bucket_lower=expected_bucket.get("lower"),
                        side=side,
                        extra={"bucket_index": expected_bucket.get("index")},
                    )
        for field in ("digest", "row_count", "null_count", "edges", "boundary_rule"):
            if expected.get(field) != actual.get(field):
                raise _replay_error(
                    "重算直方图摘要不一致",
                    field=field,
                    expected=expected.get(field),
                    actual=actual.get(field),
                    bucket_lower=_bucket_lower(expected),
                    side=side,
                )


def _compare_scalar_paths(snapshot: dict[str, Any], result: Any) -> None:
    comparisons = [
        (
            "histogram.estimate_fixed",
            snapshot["estimates"]["histogram"]["estimate_fixed"],
            result.histogram_estimate["estimate_fixed"],
        ),
        (
            "sampling.estimate_fixed",
            snapshot["estimates"]["sampling"]["estimate_fixed"],
            result.sampling_estimate["estimate_fixed"],
        ),
        ("truth.row_count", snapshot["estimates"]["truth_rows"], result.truth["row_count"]),
        (
            "histogram.path_digest",
            snapshot["estimates"]["histogram"].get("path_digest"),
            result.histogram_estimate["path_digest"],
        ),
        ("attribution.digest", snapshot["attribution"]["digest"], result.attribution["digest"]),
        ("decision.digest", snapshot["decision"]["digest"], result.decision["digest"]),
        ("analysis.digest", snapshot["analysis_digest"], result.analysis_digest),
    ]
    for field, expected, actual in comparisons:
        if expected != actual:
            raise _replay_error(
                "快照重算结果不可复现",
                field=field,
                expected=expected,
                actual=actual,
                bucket_lower=_bucket_lower(snapshot["histograms"]["left"]),
            )
    for side in ("left", "right"):
        expected = snapshot["sample_ledgers"][side]
        actual = result.ledgers[side].to_dict(include_entries=False)
        for field in ("digest", "selected_rows_digest", "sample_size", "derived_seed", "key_count"):
            if expected.get(field) != actual.get(field):
                raise _replay_error(
                    "重算样本账本不一致",
                    field=f"{side}.{field}",
                    expected=expected.get(field),
                    actual=actual.get(field),
                    bucket_lower=_bucket_lower(snapshot["histograms"]["left"]),
                    side=side,
                )


def restore_snapshot(
    path: str | Path,
    workdir: str | Path,
    *,
    logger: Any = None,
) -> tuple[Session, dict[str, Any]]:
    payload = load_snapshot_payload(path)
    spec = parse_scenario(payload["spec"], source=f"snapshot:{path}")
    snapshot_version = payload.get("parameter_version")
    if not isinstance(snapshot_version, int) or isinstance(snapshot_version, bool):
        raise GSBError("E_VERSION", "快照参数版本缺失或非法", {"parameter_version": snapshot_version})
    if spec.join.parameter_version != snapshot_version:
        raise GSBError(
            "E_VERSION",
            "快照参数版本与表声明版本不一致",
            {"snapshot_version": snapshot_version, "spec_version": spec.join.parameter_version},
        )
    if "parameter_version" not in payload.get("parameters", {}):
        raise GSBError("E_VERSION", "快照参数段缺少 parameter_version", {"path": str(path)})
    restored = Session(spec, workdir, logger=logger)
    for table_name, expected in payload.get("data", {}).get("tables", {}).items():
        actual = restored.samples[table_name]
        if expected.get("content_digest") != actual.content_digest:
            raise GSBError(
                "E_SAMPLE",
                "快照样本数据摘要与本地重生成结果不一致",
                {
                    "table": table_name,
                    "expected_content_digest": expected.get("content_digest"),
                    "actual_content_digest": actual.content_digest,
                },
            )
    result = restored.run()
    _compare_histograms(payload, result)
    _compare_scalar_paths(payload, result)
    if payload.get("_snapshot_digest_valid") is False:
        raise _replay_error(
            "快照内容摘要不匹配",
            field="snapshot_digest",
            expected=payload.get("snapshot_digest"),
            actual=payload.get("_snapshot_digest_actual"),
            bucket_lower=_bucket_lower(payload["histograms"]["left"]),
        )
    replay = {
        "status": "OK",
        "parameter_version": snapshot_version,
        "analysis_digest": result.analysis_digest,
        "histogram_digest": result.histograms["left"].digest,
        "sampling_ledger_digest": result.ledgers["left"].digest,
        "attribution_digest": result.attribution["digest"],
        "decision_digest": result.decision["digest"],
        "snapshot_digest": payload.get("snapshot_digest"),
        "snapshot_digest_valid": payload.get("_snapshot_digest_valid"),
        "compared": [
            "histogram buckets",
            "histogram digests",
            "sample ledger summaries",
            "histogram estimate",
            "sampling estimate",
            "truth rows",
            "attribution digest",
            "decision digest",
            "analysis digest",
        ],
    }
    restored.logger.log(
        "INFO",
        "OK",
        "snapshot_replayed",
        {
            "path": str(path),
            "parameter_version": snapshot_version,
            "analysis_digest": result.analysis_digest,
        },
    )
    return restored, replay
