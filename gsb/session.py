from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .attribution import attribute_differences
from .errors import GSBError
from .estimation import (
    decide_estimates,
    dominant_deviation_pairs,
    histogram_join_estimate,
    sampling_path_summary,
)
from .filtering import filtered_arrays, predicate_digest
from .generation import TableSample, ensure_scenario_samples, scenario_data_digest
from .histogram import Histogram, build_histogram
from .logging_store import NullLogger, StructuredLogger
from .model import ScenarioSpec, load_scenario
from .sampling import SampleLedger, draw_sample, sampling_join_estimate
from .truth import execute_equi_join, truth_summary
from .util import FIXED_SCALE, json_number, logical_content_digest, sha256_json


REPORT_FORMAT = "gsb-report-v1"


@dataclass(frozen=True)
class AnalysisResult:
    parameter_version: int
    report: dict[str, Any]
    histograms: Mapping[str, Histogram]
    ledgers: Mapping[str, SampleLedger]
    histogram_estimate: dict[str, Any]
    sampling_estimate: dict[str, Any]
    truth: dict[str, Any]
    attribution: dict[str, Any]
    decision: dict[str, Any]
    analysis_digest: str

    def summary(self) -> dict[str, Any]:
        return {
            "parameter_version": self.parameter_version,
            "analysis_digest": self.analysis_digest,
            "histogram_estimate": self.histogram_estimate["estimate_fixed"],
            "sampling_estimate": self.sampling_estimate["estimate_fixed"],
            "truth_rows": self.truth["row_count"],
            "decision": self.decision["outcome"],
            "histogram_digest": self.report["histograms"]["left"]["digest"],
            "sampling_ledger_digest": self.report["sampling"]["left"]["digest"],
            "attribution_digest": self.attribution["digest"],
            "truth_digest": self.truth["digest"],
        }


@dataclass(frozen=True)
class EvolutionResult:
    transition: dict[str, Any]
    result: AnalysisResult


class Session:
    def __init__(
        self,
        spec: ScenarioSpec,
        workdir: str | Path,
        *,
        logger: StructuredLogger | None = None,
        load_samples: bool = True,
    ) -> None:
        self.spec = spec
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or NullLogger()
        self.samples: dict[str, TableSample] = {}
        self.current: AnalysisResult | None = None
        self.stale_versions: list[dict[str, Any]] = []
        self.last_transition: dict[str, Any] | None = None
        self._version_floor = spec.join.parameter_version
        if load_samples:
            self.samples = ensure_scenario_samples(self.spec, self.workdir)

    @classmethod
    def from_spec_file(
        cls,
        spec_path: str | Path,
        workdir: str | Path,
        *,
        logger: StructuredLogger | None = None,
    ) -> "Session":
        return cls(load_scenario(spec_path), workdir, logger=logger)

    @property
    def parameter_version(self) -> int:
        return self.spec.join.parameter_version

    def ensure_samples(self) -> dict[str, TableSample]:
        self.samples = ensure_scenario_samples(self.spec, self.workdir)
        return self.samples

    def _analyze(self, spec: ScenarioSpec, samples: Mapping[str, TableSample]) -> AnalysisResult:
        join = spec.join
        left_table = spec.left
        right_table = spec.right
        left_sample = samples[join.left_table]
        right_sample = samples[join.right_table]
        left_filtered, _ = filtered_arrays(left_table, left_sample.columns, join.left_predicates)
        right_filtered, _ = filtered_arrays(right_table, right_sample.columns, join.right_predicates)
        left_columns = [column.name for column in left_table.columns]
        right_columns = [column.name for column in right_table.columns]
        left_filtered_digest = logical_content_digest(left_filtered, names=left_columns)
        right_filtered_digest = logical_content_digest(right_filtered, names=right_columns)
        left_predicate_digest = predicate_digest(join.left_predicates)
        right_predicate_digest = predicate_digest(join.right_predicates)
        left_histogram = build_histogram(
            left_filtered[join.left_key],
            table=left_table.name,
            column=join.left_key,
            kind=left_table.column(join.left_key).kind,
            bucket_count=join.buckets,
            rule=join.boundary_rules[left_table.column(join.left_key).kind],
            source_digest=left_filtered_digest,
        )
        right_histogram = build_histogram(
            right_filtered[join.right_key],
            table=right_table.name,
            column=join.right_key,
            kind=right_table.column(join.right_key).kind,
            bucket_count=join.buckets,
            rule=join.boundary_rules[right_table.column(join.right_key).kind],
            source_digest=right_filtered_digest,
        )
        left_ledger = draw_sample(
            left_filtered,
            table=left_table.name,
            key_column=join.left_key,
            table_rows=left_table.declared_rows,
            sample_size=join.sample_size,
            sample_seed=join.sample_seed,
            data_digest=left_filtered_digest,
            predicate_digest=left_predicate_digest,
            names=left_columns,
        )
        right_ledger = draw_sample(
            right_filtered,
            table=right_table.name,
            key_column=join.right_key,
            table_rows=right_table.declared_rows,
            sample_size=join.sample_size,
            sample_seed=join.sample_seed,
            data_digest=right_filtered_digest,
            predicate_digest=right_predicate_digest,
            names=right_columns,
        )
        histogram_estimate = histogram_join_estimate(
            left_histogram,
            right_histogram,
            left_filtered_rows=len(left_filtered[join.left_key]),
            right_filtered_rows=len(right_filtered[join.right_key]),
        )
        histogram_estimate["estimate"] = histogram_estimate["estimate_fixed"] / FIXED_SCALE
        sampling_estimate = sampling_join_estimate(
            left_ledger,
            right_ledger,
            left_filtered_rows=len(left_filtered[join.left_key]),
            right_filtered_rows=len(right_filtered[join.right_key]),
        )
        sampling_estimate["estimate"] = sampling_estimate["estimate_from_fixed"]
        truth = execute_equi_join(
            left_filtered,
            right_filtered,
            left_key=join.left_key,
            right_key=join.right_key,
            left_histogram=left_histogram,
            right_histogram=right_histogram,
            left_predicate_digest=left_predicate_digest,
            right_predicate_digest=right_predicate_digest,
        )
        truth["left_null_key_count"] = int(left_histogram.null_count)
        truth["right_null_key_count"] = int(right_histogram.null_count)
        truth["left_filtered_rows_with_null_keys"] = int(len(left_filtered[join.left_key]))
        truth["right_filtered_rows_with_null_keys"] = int(len(right_filtered[join.right_key]))
        decision = decide_estimates(
            histogram_estimate=histogram_estimate,
            sampling_estimate=sampling_estimate,
            truth=truth,
            buckets=join.buckets,
            sample_size=join.sample_size,
        )
        attribution = attribute_differences(
            left_histogram=left_histogram,
            right_histogram=right_histogram,
            histogram_pairs=histogram_estimate["pairs"],
            sampling_pair_contributions=sampling_estimate["pair_contributions"],
            truth_pair_rows=truth["pair_rows"],
            sample_size=join.sample_size,
            buckets=join.buckets,
            top_k=join.top_k,
            left_skewed=left_table.skewed,
            right_skewed=right_table.skewed,
        )
        pair_meta = histogram_estimate["pairs"]
        histogram_pair_map = {
            str(pair["pair_id"]): int(pair["histogram_fixed"]) for pair in pair_meta
        }
        sampling_pair_map = {
            str(pair["pair_id"]): int(pair["sampling_fixed"]) for pair in attribution["pairs"]
        }
        histogram_deviations = dominant_deviation_pairs(
            histogram_pair_map,
            truth["pair_rows"],
            pair_meta,
            top_k=join.top_k,
        )
        sampling_deviations = dominant_deviation_pairs(
            sampling_pair_map,
            truth["pair_rows"],
            pair_meta,
            top_k=join.top_k,
        )
        decision = dict(decision)
        decision["dominant_deviations"] = {
            "histogram": histogram_deviations,
            "sampling": sampling_deviations,
        }
        decision["dominant_deviation_bucket"] = (
            histogram_deviations[0]
            if decision["selected_path"] == "histogram" and histogram_deviations
            else sampling_deviations[0]
            if sampling_deviations
            else None
        )
        decision.pop("digest", None)
        decision["digest"] = sha256_json(decision)
        report = self._build_report(
            spec,
            samples,
            left_filtered,
            right_filtered,
            left_filtered_digest,
            right_filtered_digest,
            left_predicate_digest,
            right_predicate_digest,
            left_histogram,
            right_histogram,
            left_ledger,
            right_ledger,
            histogram_estimate,
            sampling_estimate,
            truth,
            attribution,
            decision,
        )
        analysis_digest = sha256_json(
            {
                "spec_digest": spec.digest,
                "parameters": spec.join.version_parameters,
                "histograms": {
                    "left": left_histogram.digest,
                    "right": right_histogram.digest,
                },
                "ledgers": {"left": left_ledger.digest, "right": right_ledger.digest},
                "histogram": histogram_estimate["path_digest"],
                "sampling": {
                    "estimate_fixed": sampling_estimate["estimate_fixed"],
                    "sample_hit_pairs": sampling_estimate["sample_hit_pairs"],
                },
                "truth": truth["digest"],
                "attribution": attribution["digest"],
                "decision": decision["digest"],
            }
        )
        report["analysis_digest"] = analysis_digest
        return AnalysisResult(
            parameter_version=join.parameter_version,
            report=report,
            histograms={"left": left_histogram, "right": right_histogram},
            ledgers={"left": left_ledger, "right": right_ledger},
            histogram_estimate=histogram_estimate,
            sampling_estimate=sampling_estimate,
            truth=truth,
            attribution=attribution,
            decision=decision,
            analysis_digest=analysis_digest,
        )

    def _build_report(
        self,
        spec: ScenarioSpec,
        samples: Mapping[str, TableSample],
        left_filtered: Mapping[str, np.ndarray],
        right_filtered: Mapping[str, np.ndarray],
        left_filtered_digest: str,
        right_filtered_digest: str,
        left_predicate_digest: str,
        right_predicate_digest: str,
        left_histogram: Histogram,
        right_histogram: Histogram,
        left_ledger: SampleLedger,
        right_ledger: SampleLedger,
        histogram_estimate: Mapping[str, Any],
        sampling_estimate: Mapping[str, Any],
        truth: Mapping[str, Any],
        attribution: Mapping[str, Any],
        decision: Mapping[str, Any],
    ) -> dict[str, Any]:
        join = spec.join
        histogram_public = {
            key: value for key, value in histogram_estimate.items() if key != "pairs"
        }
        histogram_public["pair_count"] = len(histogram_estimate["pairs"])
        sampling_public = sampling_path_summary(sampling_estimate)
        attribution_public = {
            "pair_count": attribution["pair_count"],
            "top_k": attribution["top_k"],
            "digest": attribution["digest"],
            "sparse_threshold": attribution["sparse_threshold"],
            "classification_rule": attribution["classification_rule"],
            "sort_rule": attribution["sort_rule"],
        }
        return {
            "format": REPORT_FORMAT,
            "status": decision["outcome"],
            "scenario": spec.name,
            "spec_digest": spec.digest,
            "parameters": {
                **join.version_parameters,
                "top_k": join.top_k,
                "parameter_version": join.parameter_version,
                "left_table": join.left_table,
                "right_table": join.right_table,
                "left_key": join.left_key,
                "right_key": join.right_key,
                "boundary_rule_declarations": {
                    "int": (
                        "左闭右开；最后一个桶包含右端点"
                        if join.boundary_rules["int"] == "left_closed_right_open"
                        else "左开右闭；第一个桶包含左端点"
                    ),
                    "float": (
                        "左开右闭；第一个桶包含左端点"
                        if join.boundary_rules["float"] == "left_open_right_closed"
                        else "左闭右开；最后一个桶包含右端点"
                    ),
                },
            },
            "data": {
                "scenario_data_digest": scenario_data_digest(samples),
                "tables": {name: sample.to_summary() for name, sample in sorted(samples.items())},
                "filtered": {
                    "left": {
                        "rows": int(len(left_filtered[join.left_key])),
                        "content_digest": left_filtered_digest,
                        "predicate_digest": left_predicate_digest,
                        "predicates": [predicate.to_dict() for predicate in join.left_predicates],
                    },
                    "right": {
                        "rows": int(len(right_filtered[join.right_key])),
                        "content_digest": right_filtered_digest,
                        "predicate_digest": right_predicate_digest,
                        "predicates": [predicate.to_dict() for predicate in join.right_predicates],
                    },
                },
            },
            "histograms": {
                "left": left_histogram.to_dict(),
                "right": right_histogram.to_dict(),
                "scope_declaration": {
                    "quantile_scope": "full_filtered",
                    "boundary_rules": dict(join.boundary_rules),
                    "basis": "联接键先经过声明谓词过滤，再用过滤后全量非空键构建等深分位桶。",
                },
            },
            "sampling": {
                "left": left_ledger.to_dict(),
                "right": right_ledger.to_dict(),
                "declaration": {
                    "method": "canonical-row-order + SHA256-derived PCG64 permutation",
                    "physical_order_independent": True,
                    "sample_seed": join.sample_seed,
                    "sample_size": join.sample_size,
                },
            },
            "estimates": {
                "histogram": histogram_public,
                "sampling": sampling_public,
                "truth": truth_summary(truth),
                "fixed_point_scale": FIXED_SCALE,
                "compatibility": {
                    "histogram_within_bounds": 0 <= histogram_estimate["estimate_fixed"]
                    <= int(round(histogram_estimate["max_possible"] * FIXED_SCALE)),
                    "sampling_within_bounds": 0 <= sampling_estimate["estimate_fixed"]
                    <= int(round(sampling_estimate["max_possible"] * FIXED_SCALE)),
                    "truth_within_bounds": 0 <= truth["row_count"] <= truth["max_possible"],
                },
            },
            "attribution": attribution_public,
            "truth": truth_summary(truth),
            "decision": dict(decision),
            "algorithms": {
                "histogram": (
                    "对每个重叠桶对按 (左桶行数/左桶值域宽度) * "
                    "(右桶行数/右桶值域宽度) * 重叠宽度 估算，逐对定点后求和并封顶。"
                ),
                "sampling": (
                    "样本键频次 fL*fR 乘以 N_L/n_L * N_R/n_R；"
                    "逐共同样本键定点贡献求和。"
                ),
                "attribution": "两条路径均投影到同一显式桶对，按定点贡献差进行稳定排序与分类。",
            },
        }

    def run(self) -> AnalysisResult:
        if not self.samples:
            self.ensure_samples()
        result = self._analyze(self.spec, self.samples)
        self.current = result
        self.logger.log(
            "INFO",
            result.decision["outcome"],
            "analysis_completed",
            {
                "parameter_version": result.parameter_version,
                "analysis_digest": result.analysis_digest,
                "histogram_estimate_fixed": result.histogram_estimate["estimate_fixed"],
                "sampling_estimate_fixed": result.sampling_estimate["estimate_fixed"],
                "truth_rows": result.truth["row_count"],
                "within_interval": result.decision["within_interval"],
            },
        )
        if result.decision["outcome"] == "E_EST":
            self.logger.log(
                "ERROR",
                "E_EST",
                "estimate_interval_miss",
                {
                    "histogram": result.decision["histogram"],
                    "sampling": result.decision["sampling"],
                    "interval": result.decision["interval"],
                    "dominant_deviation_bucket": result.decision["dominant_deviation_bucket"],
                },
            )
        return result

    def current_report(self) -> dict[str, Any]:
        if self.current is None:
            raise GSBError(
                "E_VERSION",
                "当前参数版本尚无有效估算结果",
                {"parameter_version": self.parameter_version},
            )
        report = self.current.report
        if self.last_transition is not None:
            report = dict(report)
            report["last_transition"] = self.last_transition
        return report

    def stale_summary(self) -> list[dict[str, Any]]:
        return list(self.stale_versions)

    def _mark_stale(self, result: AnalysisResult | None, reason: str) -> None:
        if result is None:
            return
        self.stale_versions.append(
            {
                "status": "stale",
                "reason": reason,
                "parameter_version": result.parameter_version,
                "analysis_digest": result.analysis_digest,
                "estimates": {
                    "histogram_fixed": result.histogram_estimate["estimate_fixed"],
                    "sampling_fixed": result.sampling_estimate["estimate_fixed"],
                },
                "truth_rows": result.truth["row_count"],
                "decision": result.decision["outcome"],
            }
        )

    def evolve(
        self,
        *,
        buckets: int | None = None,
        sample_size: int | None = None,
        sample_seed: int | None = None,
        boundary_rules: Mapping[str, str] | None = None,
        top_k: int | None = None,
    ) -> EvolutionResult:
        old = self.current
        old_join = self.spec.join
        new_buckets = old_join.buckets if buckets is None else buckets
        new_sample_size = old_join.sample_size if sample_size is None else sample_size
        new_sample_seed = old_join.sample_seed if sample_seed is None else sample_seed
        new_top_k = old_join.top_k if top_k is None else top_k
        if not isinstance(new_buckets, int) or isinstance(new_buckets, bool) or new_buckets <= 0:
            raise GSBError("E_BUCKET", "分桶数必须为正整数", {"buckets": new_buckets})
        if not isinstance(new_sample_size, int) or isinstance(new_sample_size, bool) or new_sample_size <= 0:
            raise GSBError("E_SPEC", "抽样规模必须为正整数", {"sample_size": new_sample_size})
        if not isinstance(new_sample_seed, int) or isinstance(new_sample_seed, bool):
            raise GSBError("E_SPEC", "抽样种子必须为整数", {"sample_seed": new_sample_seed})
        if not isinstance(new_top_k, int) or isinstance(new_top_k, bool) or new_top_k <= 0:
            raise GSBError("E_SPEC", "top_k 必须为正整数", {"top_k": new_top_k})
        from .model import BOUNDARY_RULES

        rules = dict(old_join.boundary_rules)
        if boundary_rules:
            for kind, rule in boundary_rules.items():
                if kind not in {"int", "float"}:
                    raise GSBError("E_SPEC", "边界规则类型必须为 int 或 float", {"kind": kind})
                normalized = BOUNDARY_RULES.get(rule) if isinstance(rule, str) else None
                if normalized is None:
                    raise GSBError("E_SPEC", "未知边界归属规则", {"rule": rule})
                rules[kind] = normalized
        old_parameters = dict(old_join.version_parameters)
        old_parameters["top_k"] = old_join.top_k
        new_parameters = {
            "buckets": new_buckets,
            "sample_size": new_sample_size,
            "sample_seed": new_sample_seed,
            "boundary_rules": rules,
            "top_k": new_top_k,
        }
        changed = {
            key: {"old": old_parameters[key], "new": new_parameters[key]}
            for key in new_parameters
            if old_parameters[key] != new_parameters[key]
        }
        if not changed:
            if old is None:
                return EvolutionResult({}, self.run())
            return EvolutionResult(
                {
                    "old_parameter_version": old_join.parameter_version,
                    "new_parameter_version": old_join.parameter_version,
                    "changed": {},
                    "no_op": True,
                },
                old,
            )
        new_version = max(old_join.parameter_version + 1, self._version_floor + 1)
        new_join = replace(
            old_join,
            buckets=new_buckets,
            sample_size=new_sample_size,
            sample_seed=new_sample_seed,
            boundary_rules=rules,
            top_k=new_top_k,
            parameter_version=new_version,
        )
        new_spec = replace(self.spec, join=new_join)
        new_samples = ensure_scenario_samples(new_spec, self.workdir)
        new_result = self._analyze(new_spec, new_samples)
        affected = self._affected_pairs(old, new_result)
        transition = {
            "status": "replaced",
            "old_parameter_version": old_join.parameter_version,
            "new_parameter_version": new_version,
            "parameter_diff": changed,
            "affected_overlap_pairs": affected,
            "affected_pair_count": len(affected),
            "estimate_changes": {
                "histogram": {
                    "old_fixed": old.histogram_estimate["estimate_fixed"] if old else None,
                    "new_fixed": new_result.histogram_estimate["estimate_fixed"],
                },
                "sampling": {
                    "old_fixed": old.sampling_estimate["estimate_fixed"] if old else None,
                    "new_fixed": new_result.sampling_estimate["estimate_fixed"],
                },
                "truth": {
                    "old_rows": old.truth["row_count"] if old else None,
                    "new_rows": new_result.truth["row_count"],
                },
                "selected_path": {
                    "old": old.decision["selected_path"] if old else None,
                    "new": new_result.decision["selected_path"],
                },
            },
            "stale_artifacts_excluded": True,
        }
        self._mark_stale(old, f"参数版本迁移至 {new_version}")
        self.spec = new_spec
        self.samples = new_samples
        self.current = new_result
        self.last_transition = transition
        self._version_floor = new_version
        new_report = dict(self.current.report)
        new_report["last_transition"] = transition
        self.current = replace(self.current, report=new_report)
        self.logger.log(
            "INFO",
            "OK",
            "parameter_version_replaced",
            {
                "old_parameter_version": old_join.parameter_version,
                "new_parameter_version": new_version,
                "parameter_diff": changed,
                "affected_pair_count": len(affected),
            },
        )
        return EvolutionResult(transition=transition, result=self.current)

    def _affected_pairs(
        self,
        old: AnalysisResult | None,
        new: AnalysisResult,
    ) -> list[dict[str, Any]]:
        if old is None:
            return []
        old_pairs = {record["pair_id"]: record for record in old.attribution["pairs"]}
        new_pairs = {record["pair_id"]: record for record in new.attribution["pairs"]}
        records: list[dict[str, Any]] = []
        for pair_id in sorted(
            set(old_pairs) | set(new_pairs),
            key=lambda value: (
                int(value.split(":")[0][1:]),
                int(value.split(":")[1][1:]),
            ),
        ):
            old_record = old_pairs.get(pair_id)
            new_record = new_pairs.get(pair_id)
            old_hist = int(old_record["histogram_fixed"]) if old_record else 0
            new_hist = int(new_record["histogram_fixed"]) if new_record else 0
            old_sampling = int(old_record["sampling_fixed"]) if old_record else 0
            new_sampling = int(new_record["sampling_fixed"]) if new_record else 0
            if old_hist == new_hist and old_sampling == new_sampling and old_record and new_record:
                continue
            source = new_record or old_record or {}
            records.append(
                {
                    "pair_id": pair_id,
                    "left_bucket": int(source.get("left_bucket", int(pair_id.split(":")[0][1:]))),
                    "right_bucket": int(source.get("right_bucket", int(pair_id.split(":")[1][1:]))),
                    "left_lower": json_number(source.get("left_lower")),
                    "right_lower": json_number(source.get("right_lower")),
                    "histogram_old_fixed": old_hist,
                    "histogram_new_fixed": new_hist,
                    "sampling_old_fixed": old_sampling,
                    "sampling_new_fixed": new_sampling,
                }
            )
        records.sort(
            key=lambda item: (
                item["left_lower"] if item["left_lower"] is not None else float("-inf"),
                item["left_bucket"],
                item["right_lower"] if item["right_lower"] is not None else float("-inf"),
                item["right_bucket"],
            )
        )
        return records

    def replace_state_from(self, other: "Session", *, reason: str) -> dict[str, Any]:
        old_version = self.parameter_version
        old_analysis = self.current.analysis_digest if self.current else None
        self._mark_stale(self.current, reason)
        self._version_floor = max(
            self._version_floor,
            other._version_floor,
            other.parameter_version,
        )
        self.spec = other.spec
        self.workdir = other.workdir
        self.samples = other.samples
        self.current = other.current
        self.last_transition = {
            "status": "snapshot_replaced",
            "reason": reason,
            "old_parameter_version": old_version,
            "new_parameter_version": other.parameter_version,
            "old_analysis_digest": old_analysis,
            "new_analysis_digest": other.current.analysis_digest if other.current else None,
            "stale_artifacts_excluded": True,
        }
        if self.current is not None:
            report = dict(self.current.report)
            report["last_transition"] = self.last_transition
            self.current = replace(self.current, report=report)
        return self.last_transition

    def save_snapshot(self, path: str | Path) -> dict[str, Any]:
        from .snapshot import save_snapshot

        if self.current is None:
            self.run()
        return save_snapshot(self, path)

    def verify_sample_files(self) -> dict[str, TableSample]:
        return ensure_scenario_samples(self.spec, self.workdir)


def restore_snapshot_into(
    snapshot_path: str | Path,
    workdir: str | Path,
    *,
    current: Session | None = None,
    logger: StructuredLogger | None = None,
) -> tuple[Session, dict[str, Any]]:
    from .snapshot import restore_snapshot

    restored, replay = restore_snapshot(snapshot_path, workdir, logger=logger)
    if current is None:
        return restored, replay
    version_changed = (
        current.spec.digest != restored.spec.digest
        or current.parameter_version != restored.parameter_version
    )
    transition = current.replace_state_from(
        restored,
        reason=f"快照版本 {restored.parameter_version} 恢复",
    )
    transition["version_changed"] = version_changed
    transition["replay"] = replay
    return current, transition
