from __future__ import annotations

import hashlib
import json
import math
import shutil
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from .cache import (
    build_bucket_refs,
    empty_cache,
    invalidation_summary,
    metrics_cache_key,
    metrics_lookup,
    metrics_store,
    normalize_cache,
)
from .canonical import (
    atomic_write_json,
    atomic_write_text,
    fingerprint,
    sorted_failures,
    stable_float,
)
from .config import LabConfig
from .conversion import (
    conversion_key,
    convert_v1_to_v2,
    convert_v2_to_v1,
)
from .errors import ConfigError, LabError, SimulatedTermination, SnapshotError
from .features import FeatureParams, extract_pair_features, feature_cache_key
from .formats import (
    ParseResult,
    V1Record,
    V2Record,
    encode_v1_record,
    parse_raw_pairs,
    parse_v1_file,
    parse_v2_file,
    v2_record_to_dict,
)
from .injections import InjectionSet
from .models import FeatureResult, RawPair, ScoreResult
from .scoring import score_features, scoring_cache_key
from .snapshot import SnapshotStore
from .strategy import StrategyDecision, select_strategy


def _failure(
    code: str,
    message: str,
    *,
    stage: str,
    source_version: str,
    pair_id: str | None = None,
    severity: str = "error",
    injection_id: str | None = None,
    **details: Any,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "code": code,
        "message": message,
        "stage": stage,
        "source_version": source_version,
        "severity": severity,
    }
    if pair_id is not None:
        result["pair_id"] = pair_id
    if injection_id is not None:
        result["injection_id"] = injection_id
    result.update(details)
    return result


def _exception_failure(
    exc: BaseException,
    *,
    stage: str,
    source_version: str,
    pair_id: str,
    injection_id: str | None = None,
) -> dict[str, Any]:
    if isinstance(exc, LabError):
        return _failure(
            exc.code,
            exc.message,
            stage=stage,
            source_version=source_version,
            pair_id=pair_id,
            injection_id=injection_id or exc.details.get("injection_id"),
            details=exc.details,
        )
    return _failure(
        "E_INTERNAL",
        f"{type(exc).__name__}: {exc}",
        stage=stage,
        source_version=source_version,
        pair_id=pair_id,
        injection_id=injection_id,
    )


def _task_order(pair_id: str, stage: str) -> str:
    return hashlib.sha256(f"{pair_id}:{stage}".encode("utf-8")).hexdigest()


def _parallel_results(
    items: list[RawPair],
    worker: Callable[[RawPair], dict[str, Any]],
    *,
    stage: str,
    concurrency: int,
) -> list[dict[str, Any]]:
    ordered = sorted(items, key=lambda pair: (_task_order(pair.pair_id, stage), pair.pair_id))
    if not ordered:
        return []
    workers = max(1, min(concurrency, len(ordered)))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"rewardlab-{stage}") as pool:
        futures = [pool.submit(worker, pair) for pair in ordered]
        results = [future.result() for future in futures]
    return sorted(results, key=lambda result: result["pair_id"])


class ExperimentLab:
    def __init__(
        self,
        data_dir: str | Path,
        *,
        config_path: str | Path | None = None,
        state_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        injections_path: str | Path | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.config_path = Path(config_path) if config_path else self.data_dir / "config.json"
        self.raw_path = self.data_dir / "preference_pairs.jsonl"
        self.v1_path = self.data_dir / "exchange_v1.txt"
        self.v2_path = self.data_dir / "exchange_v2.json"
        self.state_dir = Path(state_dir) if state_dir else self.data_dir / ".lab_state"
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / "output"
        self.injections_path = (
            Path(injections_path) if injections_path else self.data_dir / "injections.json"
        )
        if not self.injections_path.exists():
            self.injections_path = None
        self.snapshot_store = SnapshotStore(self.state_dir)

    def load_config(self) -> LabConfig:
        return LabConfig.load(self.config_path)

    def _load_inputs(self, config: LabConfig) -> dict[str, Any]:
        raw = parse_raw_pairs(self.raw_path)
        v1 = parse_v1_file(self.v1_path)
        v2 = parse_v2_file(self.v2_path, config)
        return {"raw": raw, "v1": v1, "v2": v2}

    def _selected_exchange(self, source_version: str, inputs: dict[str, Any]) -> ParseResult:
        return inputs[source_version]

    @staticmethod
    def _exchange_fields(
        record: V1Record | V2Record, config: LabConfig
    ) -> dict[str, Any]:
        if isinstance(record, V1Record):
            precision = (
                "milli_exact"
                if "milli_exact" in config.precision_labels
                else config.precision_labels[0]
            )
            return {
                "exchange_reward_score": record.score_milli / 1000.0,
                "exchange_reward_delta": record.delta_milli / 1000.0,
                "reward_label": config.v1_default_label,
                "precision_label": precision,
                "source_version": "v1",
            }
        return {
            "exchange_reward_score": stable_float(record.reward_score),
            "exchange_reward_delta": stable_float(record.reward_delta),
            "reward_label": record.reward_label,
            "precision_label": record.precision_label,
            "source_version": "v2",
        }

    def _feature_phase(
        self,
        *,
        pairs: list[RawPair],
        config: LabConfig,
        strategy: StrategyDecision,
        source_version: str,
        old_cache: dict[str, Any],
        new_cache: dict[str, Any],
        injections: InjectionSet,
        hits: dict[str, bool],
        failures: list[dict[str, Any]],
    ) -> tuple[dict[str, FeatureResult], list[str], int]:
        params = FeatureParams.from_config(
            config,
            precision_tier=strategy.precision_tier,
            hash_buckets=strategy.effective_hash_buckets,
        )
        results: dict[str, FeatureResult] = {}
        recomputed: list[str] = []
        reused = 0

        def work(pair: RawPair) -> dict[str, Any]:
            injection = injections.at(pair.pair_id, "features")
            if injection is not None:
                hits[injection.injection_id] = True
                if injection.kind == "zero_hash_buckets":
                    zero_params = FeatureParams(
                        tokenizer_version=params.tokenizer_version,
                        precision_tier=params.precision_tier,
                        hash_buckets=0,
                        feature_dim=params.feature_dim,
                        normalize_text=params.normalize_text,
                    )
                    try:
                        extract_pair_features(pair, zero_params)
                    except BaseException as exc:
                        return {
                            "pair_id": pair.pair_id,
                            "ok": False,
                            "failure": _exception_failure(
                                exc,
                                stage="features",
                                source_version=source_version,
                                pair_id=pair.pair_id,
                                injection_id=injection.injection_id,
                            ),
                        }
            try:
                value = extract_pair_features(pair, params)
                return {"pair_id": pair.pair_id, "ok": True, "value": value}
            except BaseException as exc:
                return {
                    "pair_id": pair.pair_id,
                    "ok": False,
                    "failure": _exception_failure(
                        exc,
                        stage="features",
                        source_version=source_version,
                        pair_id=pair.pair_id,
                        injection_id=injection.injection_id if injection else None,
                    ),
                }

        pending: list[RawPair] = []
        for pair in pairs:
            injection = injections.at(pair.pair_id, "features")
            if injection is not None:
                hits[injection.injection_id] = True
            key = feature_cache_key(
                pair,
                params,
                source_version=source_version,
                injection_fingerprint=injections.fingerprint(pair.pair_id, "features"),
            )
            entry = old_cache.get("features", {}).get(pair.pair_id)
            if (
                injection is None
                and isinstance(entry, dict)
                and entry.get("key") == key
                and isinstance(entry.get("value"), dict)
            ):
                try:
                    value = FeatureResult.from_dict(entry["value"])
                    results[pair.pair_id] = value
                    new_cache["features"][pair.pair_id] = {
                        "key": key,
                        "value": value.content_dict(),
                    }
                    reused += 1
                    continue
                except (KeyError, TypeError, ValueError):
                    pass
            pending.append(pair)
            recomputed.append(pair.pair_id)

        pairs_by_id = {pair.pair_id: pair for pair in pending}
        for result in _parallel_results(
            pending, work, stage="features", concurrency=config.concurrency
        ):
            pair_id = str(result["pair_id"])
            if result["ok"]:
                value = result["value"]
                assert isinstance(value, FeatureResult)
                pair = pairs_by_id[pair_id]
                key = feature_cache_key(
                    pair,
                    params,
                    source_version=source_version,
                    injection_fingerprint=injections.fingerprint(pair_id, "features"),
                )
                results[pair_id] = value
                new_cache["features"][pair_id] = {"key": key, "value": value.content_dict()}
            else:
                failures.append(result["failure"])
        return results, recomputed, reused

    def _scoring_phase(
        self,
        *,
        pairs_by_id: dict[str, RawPair],
        exchange_records: dict[str, V1Record | V2Record],
        features: dict[str, FeatureResult],
        config: LabConfig,
        strategy: StrategyDecision,
        source_version: str,
        old_cache: dict[str, Any],
        new_cache: dict[str, Any],
        injections: InjectionSet,
        hits: dict[str, bool],
        failures: list[dict[str, Any]],
    ) -> tuple[dict[str, dict[str, Any]], list[str], int]:
        records: dict[str, dict[str, Any]] = {}
        recomputed: list[str] = []
        reused = 0
        score_values: dict[str, ScoreResult] = {}
        pending: list[FeatureResult] = []
        for pair_id in sorted(features):
            feature = features[pair_id]
            injection = injections.at(pair_id, "scoring")
            if injection is not None:
                hits[injection.injection_id] = True
            key = scoring_cache_key(
                feature,
                config,
                effective_hash_buckets=strategy.effective_hash_buckets,
                precision_tier=strategy.precision_tier,
                effective_threshold=strategy.effective_threshold,
                source_version=source_version,
                injection_fingerprint=injections.fingerprint(pair_id, "scoring"),
            )
            entry = old_cache.get("scoring", {}).get(pair_id)
            if (
                injection is None
                and isinstance(entry, dict)
                and entry.get("key") == key
                and isinstance(entry.get("value"), dict)
            ):
                try:
                    value = ScoreResult.from_dict(entry["value"])
                    score_values[pair_id] = value
                    new_cache["scoring"][pair_id] = {"key": key, "value": value.content_dict()}
                    reused += 1
                    continue
                except (KeyError, TypeError, ValueError):
                    pass
            pending.append(feature)
            recomputed.append(pair_id)

        def work(feature: FeatureResult) -> dict[str, Any]:
            pair_id = feature.pair_id
            injection = injections.at(pair_id, "scoring")
            weights_override: dict[str, float] | None = None
            boundary_equal = False
            if injection is not None:
                hits[injection.injection_id] = True
                if injection.kind == "non_finite_weight":
                    weights_override = {"chosen": float("nan")}
                elif injection.kind == "boundary_equal_threshold":
                    boundary_equal = True
            try:
                value = score_features(
                    feature,
                    config,
                    effective_threshold=strategy.effective_threshold,
                    weights_override=weights_override,
                    boundary_equal_threshold=boundary_equal,
                    injection_id=injection.injection_id if injection else None,
                )
                return {"pair_id": pair_id, "ok": True, "value": value}
            except BaseException as exc:
                return {
                    "pair_id": pair_id,
                    "ok": False,
                    "failure": _exception_failure(
                        exc,
                        stage="scoring",
                        source_version=source_version,
                        pair_id=pair_id,
                        injection_id=injection.injection_id if injection else None,
                    ),
                }

        for result in _parallel_results(
            [pairs_by_id[feature.pair_id] for feature in pending],
            lambda pair: work(features[pair.pair_id]),
            stage="scoring",
            concurrency=config.concurrency,
        ):
            pair_id = str(result["pair_id"])
            if result["ok"]:
                value = result["value"]
                assert isinstance(value, ScoreResult)
                score_values[pair_id] = value
                key = scoring_cache_key(
                    features[pair_id],
                    config,
                    effective_hash_buckets=strategy.effective_hash_buckets,
                    precision_tier=strategy.precision_tier,
                    effective_threshold=strategy.effective_threshold,
                    source_version=source_version,
                    injection_fingerprint=injections.fingerprint(pair_id, "scoring"),
                )
                new_cache["scoring"][pair_id] = {"key": key, "value": value.content_dict()}
            else:
                failures.append(result["failure"])

        for pair_id in sorted(score_values):
            score = score_values[pair_id]
            selected = self._exchange_fields(exchange_records[pair_id], config)
            record = {
                "pair_id": pair_id,
                "feature_buckets": {str(key): features[pair_id].buckets[key] for key in sorted(features[pair_id].buckets)},
                "feature_bucket_count": len(features[pair_id].buckets),
                "chosen_score": score.chosen_score,
                "rejected_score": score.rejected_score,
                "reward_score": score.chosen_score,
                "reward_delta": score.reward_delta,
                "model_reward_score": score.chosen_score,
                "model_reward_delta": score.reward_delta,
                "source_reward_score": selected["exchange_reward_score"],
                "source_reward_delta": selected["exchange_reward_delta"],
                "effective_threshold": score.effective_threshold,
                "boundary": score.boundary,
                **selected,
            }
            records[pair_id] = record
            injection = injections.at(pair_id, "scoring")
            if injection is not None and injection.kind == "boundary_equal_threshold":
                failures.append(
                    _failure(
                        "W_BOUNDARY_EQUAL_THRESHOLD",
                        "reward delta is exactly equal to the effective boundary threshold",
                        stage="scoring",
                        source_version=source_version,
                        pair_id=pair_id,
                        severity="warning",
                        injection_id=injection.injection_id,
                        reward_delta=score.reward_delta,
                        effective_threshold=score.effective_threshold,
                    )
                )
        return records, recomputed, reused

    @staticmethod
    def _aggregate_up_losses(rows: list[dict[str, Any]], restored_count: int) -> dict[str, Any]:
        score_losses = [float(row.get("score_loss", 0.0)) for row in rows]
        delta_losses = [float(row.get("delta_loss", 0.0)) for row in rows]
        return {
            "direction": "v1_to_v2",
            "record_count": len(rows),
            "restored_from_sidecar": restored_count,
            "exact_milli_conversions": len(rows) - restored_count,
            "max_abs_score_loss": max(score_losses, default=0.0),
            "max_abs_delta_loss": max(delta_losses, default=0.0),
            "total_abs_score_loss": math.fsum(score_losses),
            "total_abs_delta_loss": math.fsum(delta_losses),
            "rows": sorted(rows, key=lambda row: row["pair_id"]),
        }

    @staticmethod
    def _aggregate_down_losses(rows: list[dict[str, Any]]) -> dict[str, Any]:
        score_residuals = [abs(Decimal(row["score_residual_milli"])) for row in rows]
        delta_residuals = [abs(Decimal(row["delta_residual_milli"])) for row in rows]
        return {
            "direction": "v2_to_v1",
            "record_count": len(rows),
            "restored_from_sidecar": 0,
            "exact_milli_conversions": len(rows),
            "max_abs_score_residual_milli": str(max(score_residuals, default=Decimal(0))),
            "max_abs_delta_residual_milli": str(max(delta_residuals, default=Decimal(0))),
            "total_abs_score_residual_milli": str(sum(score_residuals, Decimal(0))),
            "total_abs_delta_residual_milli": str(sum(delta_residuals, Decimal(0))),
            "rows": sorted(rows, key=lambda row: row["pair_id"]),
        }

    def _conversion_phase(
        self,
        *,
        v1_records: dict[str, V1Record],
        v2_records: dict[str, V2Record],
        config: LabConfig,
        source_version: str,
        old_cache: dict[str, Any],
        new_cache: dict[str, Any],
        injections: InjectionSet,
        hits: dict[str, bool],
        failures: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], list[str], int]:
        recomputed_directions: list[str] = []
        reused_directions = 0
        conversion_cache: dict[str, dict[str, Any]] = {}

        down_records: list[V1Record] = []
        down_rows: list[dict[str, Any]] = []
        sidecar_records: dict[str, Any] = {}
        for pair_id in sorted(v2_records):
            record = v2_records[pair_id]
            injection = injections.at(pair_id, "conversion")
            apply_injection = (
                injection is not None
                and injection.kind == "missing_sidecar"
                and source_version == "v2"
            )
            if apply_injection:
                hits[injection.injection_id] = True
            key = conversion_key(
                record,
                source_version="v2",
                target_version="v1",
                config=config,
            ) + (":" + injections.fingerprint(pair_id, "conversion") if apply_injection else "")
            previous = old_cache.get("conversion", {}).get(pair_id, {}).get("v2_to_v1")
            if (
                not apply_injection
                and isinstance(previous, dict)
                and previous.get("key") == key
                and isinstance(previous.get("value"), dict)
            ):
                value = previous["value"]
                reused_directions += 1
            else:
                recomputed_directions.append(f"{pair_id}:v2_to_v1")
                outcome = convert_v2_to_v1(
                    [record],
                    config,
                    force_missing_sidecar_pair=pair_id if apply_injection else None,
                    force_missing_sidecar_injection_id=(
                        injection.injection_id if apply_injection and injection else None
                    ),
                )
                if outcome.failures:
                    failures.extend(outcome.failures)
                    continue
                converted = outcome.records[0]
                value = {
                    "record": {
                        "pair_id": converted.pair_id,
                        "score_milli": converted.score_milli,
                        "delta_milli": converted.delta_milli,
                    },
                    "sidecar_entry": outcome.sidecar["records"][pair_id],
                    "loss_row": outcome.losses["rows"][0],
                }
            conversion_cache.setdefault(pair_id, {})["v2_to_v1"] = {
                "key": key,
                "value": value,
            }
            down_records.append(
                V1Record(
                    pair_id,
                    int(value["record"]["score_milli"]),
                    int(value["record"]["delta_milli"]),
                )
            )
            sidecar_records[pair_id] = value["sidecar_entry"]
            down_rows.append(value["loss_row"])

        sidecar = {
            "schema_version": 1,
            "direction": "v2_to_v1",
            "rules": {
                "rounding": "ROUND_HALF_UP",
                "residual_unit": "milli",
                "labels_preserved": True,
            },
            "records": {key: sidecar_records[key] for key in sorted(sidecar_records)},
        }
        sidecar_fingerprint = fingerprint(sidecar, prefix="sidecar:")

        up_records: list[V2Record] = []
        up_rows: list[dict[str, Any]] = []
        restored_count = 0
        for pair_id in sorted(v1_records):
            record = v1_records[pair_id]
            injection = injections.at(pair_id, "conversion")
            apply_injection = (
                injection is not None
                and injection.kind == "missing_sidecar"
                and source_version == "v1"
            )
            if apply_injection:
                hits[injection.injection_id] = True
            pair_sidecar_fingerprint = fingerprint(
                sidecar["records"].get(pair_id), prefix="sidecar-entry:"
            )
            key = (
                conversion_key(
                    record,
                    source_version="v1",
                    target_version="v2",
                    config=config,
                )
                + ":"
                + pair_sidecar_fingerprint
                + (":" + injections.fingerprint(pair_id, "conversion") if apply_injection else "")
            )
            previous = old_cache.get("conversion", {}).get(pair_id, {}).get("v1_to_v2")
            if (
                not apply_injection
                and isinstance(previous, dict)
                and previous.get("key") == key
                and isinstance(previous.get("value"), dict)
            ):
                value = previous["value"]
                reused_directions += 1
            else:
                recomputed_directions.append(f"{pair_id}:v1_to_v2")
                outcome = convert_v1_to_v2(
                    [record],
                    config,
                    sidecar=sidecar,
                    force_missing_sidecar_pair=pair_id if apply_injection else None,
                    force_missing_sidecar_injection_id=(
                        injection.injection_id if apply_injection and injection else None
                    ),
                )
                if outcome.failures:
                    failures.extend(outcome.failures)
                    continue
                converted = outcome.records[0]
                value = {
                    "record": v2_record_to_dict(converted),
                    "loss_row": outcome.losses["rows"][0],
                    "restored_from_sidecar": pair_id in set(outcome.restored_pair_ids),
                }
            conversion_cache.setdefault(pair_id, {})["v1_to_v2"] = {
                "key": key,
                "value": value,
            }
            up_records.append(
                V2Record(
                    pair_id=pair_id,
                    reward_score=float(value["record"]["reward_score"]),
                    reward_delta=float(value["record"]["reward_delta"]),
                    reward_label=str(value["record"]["reward_label"]),
                    precision_label=str(value["record"]["precision_label"]),
                )
            )
            up_rows.append(value["loss_row"])
            if value.get("restored_from_sidecar"):
                restored_count += 1

        new_cache["conversion"] = conversion_cache
        comparison_rows: list[dict[str, Any]] = []
        for pair_id in sorted(set(v1_records) & set(v2_records)):
            v1 = v1_records[pair_id]
            v2 = v2_records[pair_id]
            expected_v2 = v1.score_milli / 1000.0
            expected_delta = v1.delta_milli / 1000.0
            comparison_rows.append(
                {
                    "pair_id": pair_id,
                    "v1_to_v2_abs_score_difference": abs(v2.reward_score - expected_v2),
                    "v1_to_v2_abs_delta_difference": abs(v2.reward_delta - expected_delta),
                }
            )
        comparison = {
            "pair_count": len(comparison_rows),
            "max_v1_v2_abs_score_difference": max(
                (row["v1_to_v2_abs_score_difference"] for row in comparison_rows),
                default=0.0,
            ),
            "max_v1_v2_abs_delta_difference": max(
                (row["v1_to_v2_abs_delta_difference"] for row in comparison_rows),
                default=0.0,
            ),
            "total_v1_v2_abs_score_difference": math.fsum(
                row["v1_to_v2_abs_score_difference"] for row in comparison_rows
            ),
            "total_v1_v2_abs_delta_difference": math.fsum(
                row["v1_to_v2_abs_delta_difference"] for row in comparison_rows
            ),
            "rows": comparison_rows,
        }
        artifact = {
            "schema_version": 1,
            "source_version": source_version,
            "v1_to_v2": {
                "losses": self._aggregate_up_losses(up_rows, restored_count),
                "records": [v2_record_to_dict(record) for record in up_records],
            },
            "v2_to_v1": {
                "losses": self._aggregate_down_losses(down_rows),
                "records": [
                    {
                        "pair_id": record.pair_id,
                        "score_milli": record.score_milli,
                        "delta_milli": record.delta_milli,
                    }
                    for record in down_records
                ],
                "sidecar": sidecar,
            },
            "cross_version_comparison": comparison,
            "sidecar_fingerprint": sidecar_fingerprint,
        }
        return artifact, recomputed_directions, reused_directions

    def _evaluation_metrics(
        self,
        *,
        records: dict[str, dict[str, Any]],
        failures: list[dict[str, Any]],
        bucket_refs: dict[str, list[str]],
        config: LabConfig,
        old_cache: dict[str, Any],
        new_cache: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        ordered_records = [records[key] for key in sorted(records)]
        payload = {
            "records": ordered_records,
            "failures": sorted_failures(failures),
            "bucket_refs": bucket_refs,
        }
        key = metrics_cache_key(payload, fingerprint(config.stage_params("metrics"), prefix="metrics-params:"))
        cached = metrics_lookup(old_cache, key)
        if cached is not None:
            metrics_store(new_cache, key, cached)
            return dict(cached), True
        deltas = [float(record["reward_delta"]) for record in ordered_records]
        boundary_count = sum(1 for record in ordered_records if record["boundary"])
        metrics = {
            "record_count": len(ordered_records),
            "boundary_sample_count": boundary_count,
            "boundary_sample_ratio": boundary_count / len(ordered_records) if ordered_records else 0.0,
            "mean_reward_delta": math.fsum(deltas) / len(deltas) if deltas else 0.0,
            "mean_abs_reward_delta": (
                math.fsum(abs(delta) for delta in deltas) / len(deltas) if deltas else 0.0
            ),
            "unique_feature_buckets": len(bucket_refs),
            "failure_count": len(failures),
            "error_failure_count": sum(1 for failure in failures if failure.get("severity") == "error"),
        }
        metrics_store(new_cache, key, metrics)
        return metrics, False

    def _strategy_phase(
        self,
        *,
        pairs: list[RawPair],
        config: LabConfig,
        old_cache: dict[str, Any],
        new_cache: dict[str, Any],
    ) -> tuple[StrategyDecision, bool]:
        key = fingerprint(
            {
                "pairs": [pair.content_dict() for pair in sorted(pairs, key=lambda item: item.pair_id)],
                "feature_params": {
                    "tokenizer_version": config.tokenizer_version,
                    "feature_dim": config.feature_dim,
                    "hash_buckets": config.hash_buckets,
                    "normalize_text": config.normalize_text,
                },
                "scoring_weights": config.scoring_weights,
                "reward_delta_threshold": config.reward_delta_threshold,
                "strategy_tiers": [tier.to_dict() for tier in config.strategy_tiers],
                "forced_strategy_tier": config.forced_strategy_tier,
                "strategy_schema": 1,
            },
            prefix="strategy-pilot:",
        )
        previous = old_cache.get("strategy")
        if (
            isinstance(previous, dict)
            and previous.get("key") == key
            and isinstance(previous.get("value"), dict)
        ):
            try:
                decision = StrategyDecision.from_dict(previous["value"])
                new_cache["strategy"] = {"key": key, "value": decision.to_dict()}
                return decision, True
            except (KeyError, TypeError, ValueError):
                pass
        decision = select_strategy(pairs, config)
        new_cache["strategy"] = {"key": key, "value": decision.to_dict()}
        return decision, False

    @staticmethod
    def _write_conversion_artifacts(output_dir: Path, artifact: dict[str, Any]) -> dict[str, str]:
        output_dir.mkdir(parents=True, exist_ok=True)
        v1_text = "".join(
            encode_v1_record(
                V1Record(
                    pair_id=row["pair_id"],
                    score_milli=int(row["score_milli"]),
                    delta_milli=int(row["delta_milli"]),
                )
            )
            + "\n"
            for row in artifact["v2_to_v1"]["records"]
        )
        v2_text = json.dumps(
            artifact["v1_to_v2"]["records"],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        paths = {
            "converted_v2": "converted_v2.json",
            "converted_v1": "converted_v1.txt",
            "sidecar": "sidecar_v2_to_v1.json",
        }
        atomic_write_text(output_dir / "converted_v2.json", v2_text + "\n")
        atomic_write_text(output_dir / "converted_v1.txt", v1_text)
        atomic_write_json(output_dir / "sidecar_v2_to_v1.json", artifact["v2_to_v1"]["sidecar"])
        return paths

    def run(
        self,
        *,
        resume: bool = False,
        force_recompute: bool = False,
        crash_after_stage: str | None = None,
        source_version_override: str | None = None,
        expected_global_version: int | None = None,
        expected_source_version: str | None = None,
    ) -> dict[str, Any]:
        config = self.load_config()
        if source_version_override is not None:
            if source_version_override not in {"v1", "v2"}:
                raise ConfigError("source version override must be v1 or v2", field="source_version")
            config = config.with_updates({"source_version": source_version_override})
        config_hash = fingerprint(config.to_dict(), prefix="config:")
        injections = InjectionSet.load(self.injections_path)
        inputs = self._load_inputs(config)
        raw_pairs: dict[str, RawPair] = inputs["raw"].records
        v1_records: dict[str, V1Record] = inputs["v1"].records
        v2_records: dict[str, V2Record] = inputs["v2"].records
        source_version = config.source_version
        selected_records = self._selected_exchange(source_version, inputs).records
        failures = [
            *inputs["raw"].failures,
            *inputs["v1"].failures,
            *inputs["v2"].failures,
        ]
        hits: dict[str, bool] = {}
        injection_messages: dict[str, str] = {}

        if resume:
            recovery = self.snapshot_store.recover(
                config_hash=config_hash,
                expected_global_version=expected_global_version,
                expected_source_version=expected_source_version,
            )
            previous_payload = recovery.get("payload")
            recovery_point = recovery.get("recovery_point")
        else:
            previous_payload, snapshot_error = self.snapshot_store.load_latest_unchecked()
            previous_manifest = (
                previous_payload.get("manifest") if isinstance(previous_payload, dict) else None
            )
            recovery_point = None
            if snapshot_error:
                if previous_manifest is None:
                    try:
                        previous_manifest = self.snapshot_store.latest_manifest()
                    except SnapshotError:
                        previous_manifest = None
                failures.append(
                    _failure(
                        snapshot_error.split(":", 1)[0],
                        "latest snapshot was ignored because it is incomplete",
                        stage="snapshot",
                        source_version=source_version,
                        severity="warning",
                    )
                )
        if resume:
            previous_manifest = (
                previous_payload.get("manifest") if isinstance(previous_payload, dict) else None
            )
        old_cache = (
            normalize_cache(previous_payload.get("cache"))
            if isinstance(previous_payload, dict) and not force_recompute
            else empty_cache()
        )
        new_cache = empty_cache()
        global_version = int((previous_manifest or {}).get("global_version", 0)) + 1
        round_cursor = int((previous_manifest or {}).get("completed_round_cursor", 0)) + 1

        valid_pairs: list[RawPair] = []
        exchange_for_pair: dict[str, V1Record | V2Record] = {}
        for pair_id in sorted(raw_pairs):
            pair = raw_pairs[pair_id]
            selected = selected_records.get(pair_id)
            if selected is None:
                failures.append(
                    _failure(
                        "E_FORMAT_INVALID",
                        f"chosen source {source_version} has no valid record for pair",
                        stage="source_merge",
                        source_version=source_version,
                        pair_id=pair_id,
                        field="pair_id",
                    )
                )
                continue
            valid_pairs.append(pair)
            exchange_for_pair[pair_id] = selected

        for item in injections.items:
            if item.pair_id not in raw_pairs:
                failures.append(
                    _failure(
                        "E_INJECTION_TARGET_NOT_FOUND",
                        f"injection target {item.pair_id} is not present in raw preference pairs",
                        stage=item.stage,
                        source_version=source_version,
                        pair_id=item.pair_id,
                        injection_id=item.injection_id,
                    )
                )
                injection_messages[item.injection_id] = "target pair not found"

        strategy, _strategy_reused = self._strategy_phase(
            pairs=valid_pairs,
            config=config,
            old_cache=old_cache,
            new_cache=new_cache,
        )
        features, features_recomputed, features_reused = self._feature_phase(
            pairs=valid_pairs,
            config=config,
            strategy=strategy,
            source_version=source_version,
            old_cache=old_cache,
            new_cache=new_cache,
            injections=injections,
            hits=hits,
            failures=failures,
        )
        output_records, scores_recomputed, scores_reused = self._scoring_phase(
            pairs_by_id=raw_pairs,
            exchange_records=exchange_for_pair,
            features=features,
            config=config,
            strategy=strategy,
            source_version=source_version,
            old_cache=old_cache,
            new_cache=new_cache,
            injections=injections,
            hits=hits,
            failures=failures,
        )
        conversion, conversion_recomputed, conversion_reused = self._conversion_phase(
            v1_records=v1_records,
            v2_records=v2_records,
            config=config,
            source_version=source_version,
            old_cache=old_cache,
            new_cache=new_cache,
            injections=injections,
            hits=hits,
            failures=failures,
        )

        feature_values = {
            pair_id: feature.content_dict() for pair_id, feature in features.items()
        }
        bucket_refs = build_bucket_refs(feature_values)
        new_cache["bucket_refs"] = bucket_refs
        new_cache["pair_content_fingerprints"] = {
            pair_id: fingerprint(
                {
                    "raw": raw_pairs[pair_id].content_dict(),
                    "exchange": exchange_for_pair[pair_id].content_dict(),
                    "source_version": source_version,
                },
                prefix="pair-content:",
            )
            for pair_id in sorted(exchange_for_pair)
        }
        metrics, metrics_reused = self._evaluation_metrics(
            records=output_records,
            failures=failures,
            bucket_refs=bucket_refs,
            config=config,
            old_cache=old_cache,
            new_cache=new_cache,
        )

        ordered_failures = sorted_failures(failures)
        recomputed = {
            "features": features_recomputed,
            "scoring": scores_recomputed,
            "conversion": conversion_recomputed,
            "metrics": [] if metrics_reused else ["global"],
        }
        reused = {
            "features": features_reused,
            "scoring": scores_reused,
            "conversion": conversion_reused,
        }
        affected_buckets: set[int] = set()
        old_pair_fps = old_cache.get("pair_content_fingerprints", {})
        for pair_id in new_cache["pair_content_fingerprints"]:
            if old_pair_fps.get(pair_id) != new_cache["pair_content_fingerprints"][pair_id]:
                affected_buckets.update(
                    int(bucket) for bucket in feature_values.get(pair_id, {}).get("buckets", {})
                )
        for pair_id in features_recomputed:
            affected_buckets.update(
                int(bucket) for bucket in feature_values.get(pair_id, {}).get("buckets", {})
            )
            old_feature = old_cache.get("features", {}).get(pair_id, {})
            if isinstance(old_feature, dict):
                affected_buckets.update(
                    int(bucket)
                    for bucket in old_feature.get("value", {}).get("buckets", {})
                )
        for pair_id in set(old_pair_fps) - set(new_cache["pair_content_fingerprints"]):
            old_feature = old_cache.get("features", {}).get(pair_id, {})
            if isinstance(old_feature, dict):
                affected_buckets.update(
                    int(bucket)
                    for bucket in old_feature.get("value", {}).get("buckets", {})
                )
        invalidations = invalidation_summary(
            old_cache,
            new_cache,
            recomputed=recomputed,
            reused=reused,
            affected_buckets=sorted(affected_buckets),
        )

        boundary_samples = [
            {
                "pair_id": pair_id,
                "reward_delta": output_records[pair_id]["reward_delta"],
                "effective_threshold": output_records[pair_id]["effective_threshold"],
                "source_version": output_records[pair_id]["source_version"],
            }
            for pair_id in sorted(output_records)
            if output_records[pair_id]["boundary"]
        ]
        bucket_summary = [
            {
                "bucket": int(bucket),
                "count": sum(
                    int(feature_values[pair_id]["buckets"][bucket])
                    for pair_id in pair_ids
                    if bucket in feature_values.get(pair_id, {}).get("buckets", {})
                ),
                "pair_ids": pair_ids,
            }
            for bucket, pair_ids in bucket_refs.items()
        ]
        injection_report = injections.build_report(hits, injection_messages)
        injection_codes = sorted(
            {
                item["business_code"]
                for item in injection_report
                if item["hit"]
            }
        )
        failure_codes = sorted({str(item["code"]) for item in ordered_failures})
        conversion_artifact = conversion
        data_fingerprint_value = fingerprint(
            {
                "raw": {key: raw_pairs[key].content_dict() for key in sorted(raw_pairs)},
                "v1": {key: v1_records[key].content_dict() for key in sorted(v1_records)},
                "v2": {key: v2_records[key].content_dict() for key in sorted(v2_records)},
                "source_version": source_version,
            },
            prefix="dataset:",
        )
        core = {
            "source_version": source_version,
            "strategy": strategy.to_dict(),
            "records": [output_records[pair_id] for pair_id in sorted(output_records)],
            "boundary_samples": boundary_samples,
            "feature_bucket_summary": bucket_summary,
            "failures": ordered_failures,
            "conversion": conversion_artifact,
            "metrics": metrics,
            "injections": injection_report,
        }
        core_fingerprint = fingerprint(core, prefix="report-core:")
        artifact_paths = self._write_conversion_artifacts(self.output_dir, conversion_artifact)
        conversion_artifact["artifact_paths"] = artifact_paths
        report: dict[str, Any] = {
            "schema_version": 1,
            "deterministic_core_fingerprint": core_fingerprint,
            "config_hash": config_hash,
            "data_fingerprint": data_fingerprint_value,
            "source_version": source_version,
            "input": {
                "raw_pair_count": len(raw_pairs),
                "valid_pair_count": len(valid_pairs),
                "v1_record_count": len(v1_records),
                "v2_record_count": len(v2_records),
            },
            "strategy": strategy.to_dict(),
            "records": core["records"],
            "boundary_samples": boundary_samples,
            "feature_bucket_summary": bucket_summary,
            "feature_bucket_refs": bucket_refs,
            "failures": ordered_failures,
            "failure_business_codes": failure_codes,
            "conversion": conversion_artifact,
            "metrics": metrics,
            "invalidation": invalidations,
            "injections": injection_report,
            "injection_business_codes": injection_codes,
            "recovery_point": recovery_point,
            "snapshot": {
                "global_version": global_version,
                "completed_round_cursor": round_cursor,
                "config_hash": config_hash,
                "data_fingerprint": data_fingerprint_value,
                "source_version": source_version,
                "snapshot_commit_interval": config.snapshot_commit_interval,
                "checkpoint_due": round_cursor % config.snapshot_commit_interval == 0,
            },
        }

        stage_payloads = {
            "features": {
                "by_pair": feature_values,
                "bucket_refs": bucket_refs,
            },
            "scoring": {
                "by_pair": {
                    pair_id: output_records[pair_id] for pair_id in sorted(output_records)
                },
                "strategy": strategy.to_dict(),
            },
            "conversion": conversion_artifact,
            "cache": new_cache,
        }
        if crash_after_stage is not None:
            if crash_after_stage not in stage_payloads:
                raise ConfigError(
                    f"unknown crash stage: {crash_after_stage}",
                    field="crash_after_stage",
                )
            self.snapshot_store.write_uncommitted_crash_temp(
                crash_after_stage, stage_payloads[crash_after_stage]
            )
            raise SimulatedTermination(
                f"simulated termination after writing {crash_after_stage} temp artifact"
            )

        manifest = self.snapshot_store.commit(
            stage_payloads,
            global_version=global_version,
            config_hash=config_hash,
            data_fingerprint=data_fingerprint_value,
            source_version=source_version,
            completed_round_cursor=round_cursor,
            commit_interval=config.snapshot_commit_interval,
            extra_manifest={
                "report_core_fingerprint": report["deterministic_core_fingerprint"],
            },
        )
        report["snapshot"]["generation"] = manifest["generation"]
        report["snapshot"]["checkpoint_due"] = manifest["checkpoint_due"]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        atomic_write_text(self.output_dir / "report.json", report_text)
        atomic_write_json(self.output_dir / "latest_report.json", report)
        return report

    def apply_edit(self, edit: dict[str, Any]) -> None:
        operation = edit.get("op")
        if operation == "replace_pair":
            self.replace_pair(str(edit["pair_id"]), **dict(edit.get("changes", {})))
        elif operation == "delete_pair":
            self.delete_pair(str(edit["pair_id"]))
        elif operation == "switch_version":
            self.update_config({"source_version": str(edit["version"])})
        elif operation == "force_strategy_tier":
            self.update_config({"forced_strategy_tier": edit.get("tier")})
        elif operation == "set_weights":
            self.update_config({"scoring_weights": dict(edit["weights"])})
        elif operation == "set_threshold":
            self.update_config({"reward_delta_threshold": float(edit["value"])})
        elif operation == "update_config":
            self.update_config(dict(edit["updates"]))
        else:
            raise ConfigError(f"unknown edit operation: {operation}", field="op")

    def _raw_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with self.raw_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    def _v2_rows(self) -> list[dict[str, Any]]:
        value = json.loads(self.v2_path.read_text(encoding="utf-8"))
        if isinstance(value, dict) and isinstance(value.get("records"), list):
            return list(value["records"])
        if isinstance(value, list):
            return list(value)
        if isinstance(value, dict):
            return [value]
        raise ConfigError("cannot edit unsupported v2 structure", field="exchange_v2")

    def replace_pair(self, pair_id: str, **changes: Any) -> None:
        raw_rows = self._raw_rows()
        row = next((item for item in raw_rows if item.get("pair_id") == pair_id), None)
        if row is None:
            row = {
                "pair_id": pair_id,
                "prompt": "amber anchor",
                "chosen": "cedar circle",
                "rejected": "delta drift",
                "annotator_id": "ann-01",
                "timestamp": "2026-01-01T00:00:00Z",
            }
            raw_rows.append(row)
        for field in ("prompt", "chosen", "rejected", "annotator_id", "timestamp"):
            if field in changes:
                row[field] = str(changes[field])
        raw_rows.sort(key=lambda item: str(item["pair_id"]))
        atomic_write_text(
            self.raw_path,
            "".join(
                json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
                for item in raw_rows
            ),
        )

        v1_parse = parse_v1_file(self.v1_path)
        existing = v1_parse.records.get(pair_id, V1Record(pair_id, 500, 100))
        new_v1 = V1Record(
            pair_id,
            int(changes.get("v1_score_milli", existing.score_milli)),
            int(changes.get("v1_delta_milli", existing.delta_milli)),
        )
        self._replace_v1_line(new_v1)

        v2_rows = self._v2_rows()
        v2_row = next((item for item in v2_rows if item.get("pair_id") == pair_id), None)
        if v2_row is None:
            v2_row = {
                "pair_id": pair_id,
                "reward_score": 0.5,
                "reward_delta": 0.1,
                "reward_label": "neutral",
                "precision_label": "float64",
            }
            v2_rows.append(v2_row)
        v2_row["reward_score"] = float(changes.get("v2_score", v2_row.get("reward_score", 0.5)))
        v2_row["reward_delta"] = float(changes.get("v2_delta", v2_row.get("reward_delta", 0.1)))
        v2_row["reward_label"] = str(changes.get("reward_label", v2_row.get("reward_label", "neutral")))
        v2_row["precision_label"] = str(
            changes.get("precision_label", v2_row.get("precision_label", "float64"))
        )
        v2_rows.sort(key=lambda item: str(item.get("pair_id", "")))
        atomic_write_text(
            self.v2_path,
            json.dumps(v2_rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )

    def delete_pair(self, pair_id: str) -> None:
        raw_rows = [item for item in self._raw_rows() if item.get("pair_id") != pair_id]
        atomic_write_text(
            self.raw_path,
            "".join(
                json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
                for item in raw_rows
            ),
        )
        self._replace_v1_line(None, pair_id=pair_id)
        v2_rows = [item for item in self._v2_rows() if item.get("pair_id") != pair_id]
        atomic_write_text(
            self.v2_path,
            json.dumps(v2_rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )

    def update_config(self, updates: dict[str, Any]) -> None:
        config = self.load_config().with_updates(updates)
        atomic_write_json(self.config_path, config.to_dict())

    def _replace_v1_line(self, replacement: V1Record | None, *, pair_id: str | None = None) -> None:
        target_pair_id = pair_id if pair_id is not None else replacement.pair_id
        lines = self.v1_path.read_text(encoding="utf-8").splitlines()
        output: list[str] = []
        replaced = False
        for line in lines:
            if len(line) == 48 and line[:24].strip() == target_pair_id:
                replaced = True
                if replacement is not None:
                    output.append(encode_v1_record(replacement))
                continue
            output.append(line)
        if replacement is not None and not replaced:
            output.append(encode_v1_record(replacement))
        atomic_write_text(self.v1_path, "".join(line + "\n" for line in output))

    def reset_state(self) -> None:
        if self.state_dir.exists():
            shutil.rmtree(self.state_dir)
        self.snapshot_store = SnapshotStore(self.state_dir)
