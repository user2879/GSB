from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

from .backend import ForwardEngine
from .ledger import HitRecord, LedgerStore
from .metrics import (
    STAT_FIELDS,
    aggregate_hits,
    compare_stats,
)
from .routing import decide_route, stage_for_request
from .spec import GatewaySpec, parse_spec
from .util import FORMAT_VERSION, canonical_json, content_digest, utc_epoch_ms


@dataclass(frozen=True)
class ConsistencyResult:
    ok: bool
    code: str
    message: str
    rule_id: str = "-"
    stage_id: str = "-"
    field_name: str = "-"
    snapshot_value: Any = None
    recomputed_value: Any = None
    stats_digest: str = "-"
    hit_count: int = 0
    aggregation_path: str = "-"

    def canonical(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "code": self.code,
            "message": self.message,
            "first_mismatch": {
                "rule_id": self.rule_id,
                "stage_id": self.stage_id,
                "field": self.field_name,
                "snapshot_value": self.snapshot_value,
                "recomputed_value": self.recomputed_value,
            },
            "stats_digest": self.stats_digest,
            "hit_count": self.hit_count,
            "aggregation_path": self.aggregation_path,
        }


@dataclass(frozen=True)
class Snapshot:
    snapshot_id: str
    rebuild_id: str
    source_ledger_version: int
    observed_ledger_version: int
    excluded_hits: int
    format_version: int
    content_digest: str
    body: dict[str, Any]
    spec_digest: str
    spec_json: str
    aggregation_path: str
    rebuild_elapsed_ms: int
    budget_ms: int
    latency_exceeded: bool
    created_at_ms: int

    @property
    def stats(self) -> list[dict[str, Any]]:
        return list(self.body["stats"])

    @property
    def body_json(self) -> str:
        return canonical_json(self.body)

    def to_ledger_payload(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "source_ledger_version": self.source_ledger_version,
            "format_version": self.format_version,
            "content_digest": self.content_digest,
            "body_json": self.body_json,
            "spec_digest": self.spec_digest,
            "spec_json": self.spec_json,
            "aggregation_path": self.aggregation_path,
            "budget_ms": self.budget_ms,
            "created_at_ms": self.created_at_ms,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Snapshot":
        body = row.get("body")
        if body is None:
            body = json.loads(row["body_json"])
        return cls(
            snapshot_id=str(row["snapshot_id"]),
            rebuild_id=str(row["rebuild_id"]),
            source_ledger_version=int(row["source_ledger_version"]),
            observed_ledger_version=int(row["observed_ledger_version"]),
            excluded_hits=int(row["excluded_hits"]),
            format_version=int(row["format_version"]),
            content_digest=str(row["content_digest"]),
            body=body,
            spec_digest=str(row["spec_digest"]),
            spec_json=str(row["spec_json"]),
            aggregation_path=str(row["aggregation_path"]),
            rebuild_elapsed_ms=int(row["rebuild_elapsed_ms"]),
            budget_ms=int(row["budget_ms"]),
            latency_exceeded=bool(row["latency_exceeded"]),
            created_at_ms=int(row["created_at_ms"]),
        )


def make_body(
    *,
    spec_digest: str,
    source_ledger_version: int,
    observed_ledger_version: int,
    excluded_hits: int,
    stats: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "spec_digest": spec_digest,
        "source_ledger_version": source_ledger_version,
        "observed_ledger_version": observed_ledger_version,
        "excluded_hits": excluded_hits,
        "stats": list(stats),
    }


def materialize_snapshot(
    *,
    snapshot_id: str,
    rebuild_id: str,
    spec: GatewaySpec,
    source_ledger_version: int,
    observed_ledger_version: int,
    excluded_hits: int,
    stats: Sequence[dict[str, Any]],
    aggregation_path: str,
    rebuild_elapsed_ms: int,
    budget_ms: int,
    created_at_ms: int | None = None,
) -> Snapshot:
    body = make_body(
        spec_digest=spec.canonical_digest,
        source_ledger_version=source_ledger_version,
        observed_ledger_version=observed_ledger_version,
        excluded_hits=excluded_hits,
        stats=stats,
    )
    return Snapshot(
        snapshot_id=snapshot_id,
        rebuild_id=rebuild_id,
        source_ledger_version=source_ledger_version,
        observed_ledger_version=observed_ledger_version,
        excluded_hits=excluded_hits,
        format_version=FORMAT_VERSION,
        content_digest=content_digest(body),
        body=body,
        spec_digest=spec.canonical_digest,
        spec_json=spec.canonical_json(),
        aggregation_path=aggregation_path,
        rebuild_elapsed_ms=rebuild_elapsed_ms,
        budget_ms=budget_ms,
        latency_exceeded=rebuild_elapsed_ms > budget_ms,
        created_at_ms=created_at_ms if created_at_ms is not None else utc_epoch_ms(),
    )


def rebuild_stats_from_ledger(
    spec: GatewaySpec,
    ledger: LedgerStore,
    source_ledger_version: int,
    *,
    aggregation_path: str | None = None,
) -> tuple[list[dict[str, Any]], str, int]:
    hits = ledger.hits_upto(source_ledger_version)
    stats, selected_path = aggregate_hits(
        hits, spec, path=aggregation_path
    )
    return stats, selected_path, len(hits)


def _replay_hit(
    hit: HitRecord, spec: GatewaySpec, forward_engine: ForwardEngine
) -> tuple[bool, str, Any, Any]:
    try:
        route = spec.route_for_feature(hit.feature)
    except Exception as exc:
        return False, "feature_route", hit.rule_id, str(exc)
    try:
        stage = stage_for_request(route, hit.request_seq)
    except Exception as exc:
        return False, "request_stage", hit.stage_id, str(exc)
    decision = decide_route(spec, route, hit.request_seq, hit.client_key)
    comparisons = (
        ("rule_id", hit.rule_id, decision.rule_id),
        ("stage_id", hit.stage_id, stage.stage_id),
        ("stage_weight_bp", hit.stage_weight_bp, stage.gray_weight_bp),
        ("hash_bucket", hit.hash_bucket, decision.hash_bucket),
        ("side", hit.side, decision.side),
        ("backend_version", hit.backend_version, decision.backend_version),
    )
    for field_name, stored, recomputed in comparisons:
        if stored != recomputed:
            return False, field_name, stored, recomputed
    model = spec.model(hit.backend_version)
    output_digest, _value = forward_engine.forward_one(model, hit.sample)
    if output_digest != hit.output_digest:
        return False, "output_digest", hit.output_digest, output_digest
    if hit.format_version != FORMAT_VERSION:
        return False, "format_version", hit.format_version, FORMAT_VERSION
    return True, "-", None, None


def authoritative_check(
    snapshot: Snapshot,
    spec: GatewaySpec,
    ledger: LedgerStore,
    forward_engine: ForwardEngine,
) -> ConsistencyResult:
    if snapshot.spec_digest != spec.canonical_digest:
        return ConsistencyResult(
            ok=False,
            code="E_SNAPSHOT",
            message="snapshot spec digest differs from authoritative spec",
            rule_id="-",
            stage_id="-",
            field_name="spec_digest",
            snapshot_value=snapshot.spec_digest,
            recomputed_value=spec.canonical_digest,
        )
    hits = ledger.hits_upto(snapshot.source_ledger_version)
    for hit in hits:
        ok, field_name, stored, recomputed = _replay_hit(hit, spec, forward_engine)
        if not ok:
            return ConsistencyResult(
                ok=False,
                code="E_SNAPSHOT",
                message="authoritative replay differs from ledger hit",
                rule_id=hit.rule_id,
                stage_id=hit.stage_id,
                field_name=field_name,
                snapshot_value=stored,
                recomputed_value=recomputed,
                hit_count=len(hits),
            )
    expected_stats, path = aggregate_hits(hits, spec)
    ok, rule_id, stage_id, field_name, left, right = compare_stats(
        snapshot.stats, expected_stats
    )
    expected_digest = content_digest(expected_stats)
    if not ok:
        return ConsistencyResult(
            ok=False,
            code="E_SNAPSHOT",
            message=(
                "snapshot statistics differ from authoritative replay at "
                f"{rule_id}/{stage_id}/{field_name}"
            ),
            rule_id=rule_id,
            stage_id=stage_id,
            field_name=field_name,
            snapshot_value=left,
            recomputed_value=right,
            stats_digest=expected_digest,
            hit_count=len(hits),
            aggregation_path=path,
        )
    return ConsistencyResult(
        ok=True,
        code="OK",
        message=(
            f"snapshot statistics match authoritative replay for {len(hits)} hits"
        ),
        stats_digest=expected_digest,
        hit_count=len(hits),
        aggregation_path=path,
    )


def replay_saved_snapshot(
    snapshot: Snapshot,
    ledger: LedgerStore,
) -> ConsistencyResult:
    try:
        recomputed_body_digest = content_digest(snapshot.body)
    except Exception as exc:
        return ConsistencyResult(
            ok=False,
            code="E_REPLAY",
            message=f"saved snapshot body cannot be canonicalized: {exc}",
            field_name="body_json",
        )
    if recomputed_body_digest != snapshot.content_digest:
        return ConsistencyResult(
            ok=False,
            code="E_REPLAY",
            message="saved snapshot body digest does not match content_digest",
            field_name="content_digest",
            snapshot_value=snapshot.content_digest,
            recomputed_value=recomputed_body_digest,
        )
    try:
        stored_spec = parse_spec(json.loads(snapshot.spec_json))
    except Exception as exc:
        return ConsistencyResult(
            ok=False,
            code="E_REPLAY",
            message=f"saved snapshot spec cannot be parsed: {exc}",
            field_name="spec_json",
        )
    if stored_spec.canonical_digest != snapshot.spec_digest:
        return ConsistencyResult(
            ok=False,
            code="E_REPLAY",
            message="saved snapshot spec digest differs from saved spec parameters",
            field_name="spec_digest",
            snapshot_value=snapshot.spec_digest,
            recomputed_value=stored_spec.canonical_digest,
        )
    hits = ledger.hits_upto(snapshot.source_ledger_version)
    stats, path = aggregate_hits(hits, stored_spec)
    excluded = ledger.count_hits_between(
        snapshot.source_ledger_version, snapshot.observed_ledger_version
    )
    rebuilt_body = make_body(
        spec_digest=snapshot.spec_digest,
        source_ledger_version=snapshot.source_ledger_version,
        observed_ledger_version=snapshot.observed_ledger_version,
        excluded_hits=excluded,
        stats=stats,
    )
    ok, rule_id, stage_id, field_name, saved_value, rebuilt_value = compare_stats(
        snapshot.stats, stats
    )
    if not ok:
        return ConsistencyResult(
            ok=False,
            code="E_REPLAY",
            message=(
                "recovered snapshot differs at "
                f"{rule_id}/{stage_id}/{field_name}"
            ),
            rule_id=rule_id,
            stage_id=stage_id,
            field_name=field_name,
            snapshot_value=saved_value,
            recomputed_value=rebuilt_value,
            hit_count=len(hits),
            aggregation_path=path,
        )
    if excluded != snapshot.excluded_hits:
        return ConsistencyResult(
            ok=False,
            code="E_REPLAY",
            message="recovered snapshot excluded-hit count differs",
            field_name="excluded_hits",
            snapshot_value=snapshot.excluded_hits,
            recomputed_value=excluded,
            hit_count=len(hits),
            aggregation_path=path,
        )
    rebuilt_digest = content_digest(rebuilt_body)
    if rebuilt_digest != snapshot.content_digest:
        return ConsistencyResult(
            ok=False,
            code="E_REPLAY",
            message="recovered snapshot content digest differs",
            field_name="content_digest",
            snapshot_value=snapshot.content_digest,
            recomputed_value=rebuilt_digest,
            hit_count=len(hits),
            aggregation_path=path,
        )
    return ConsistencyResult(
        ok=True,
        code="OK",
        message=f"snapshot replay matched for {len(hits)} committed hits",
        stats_digest=content_digest(stats),
        hit_count=len(hits),
        aggregation_path=path,
    )


def diff_snapshots(previous: Snapshot | None, current: Snapshot) -> list[dict[str, Any]]:
    if previous is None:
        return [
            {
                "rule_id": row["rule_id"],
                "stage_id": row["stage_id"],
                "changed_fields": ["created"],
                "before": None,
                "after": {field: row[field] for field in STAT_FIELDS},
            }
            for row in current.stats
        ]
    before_map = {
        (row["rule_id"], row["stage_id"]): row for row in previous.stats
    }
    changes: list[dict[str, Any]] = []
    for row in current.stats:
        key = (row["rule_id"], row["stage_id"])
        before = before_map.get(key)
        if before is None:
            changed_fields = ["created"]
        else:
            changed_fields = [
                field for field in STAT_FIELDS if before.get(field) != row.get(field)
            ]
        if changed_fields:
            changes.append(
                {
                    "rule_id": row["rule_id"],
                    "stage_id": row["stage_id"],
                    "changed_fields": changed_fields,
                    "before": (
                        None
                        if before is None
                        else {field: before.get(field) for field in STAT_FIELDS}
                    ),
                    "after": {field: row.get(field) for field in STAT_FIELDS},
                }
            )
    current_keys = {(row["rule_id"], row["stage_id"]) for row in current.stats}
    for key, before in before_map.items():
        if key not in current_keys:
            changes.append(
                {
                    "rule_id": key[0],
                    "stage_id": key[1],
                    "changed_fields": ["removed"],
                    "before": {field: before.get(field) for field in STAT_FIELDS},
                    "after": None,
                }
            )
    return changes

