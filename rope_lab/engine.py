"""Deterministic incremental fixed-point RoPE engine."""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from .codes import E_STAGE_FAILURE, E_VALUE_NONFINITE
from .commands import dimension_dependency_fingerprint, sample_geometry_fingerprint
from .fixedpoint import fixed_config_view, wrap_signed
from .hashing import cache_key, fingerprint, stable_task_key
from .rope import RoPEScheme, quantized_encoding, rotate_single_dimension, rotate_vector_angle_quantized
from .scoring import (
    aggregate_element_errors,
    classify_key,
    element_error,
    float_dot_product,
    ranking_pair_flips,
    stable_softmax,
)
from .scheduler import plan_batch


STAGE_ORDER = {"position": 1, "rotation": 2, "dot_term": 3, "score": 4, "normalization": 5, "stats": 6}
ERROR_STAGES = ("position", "rotation", "dot", "normalization")


class LabEngine:
    def __init__(
        self,
        config: dict[str, Any],
        samples: list[dict[str, Any]],
        *,
        global_version: int = 1,
        parallel: bool | None = None,
    ):
        self.config = copy.deepcopy(config)
        self.samples = sorted(copy.deepcopy(samples), key=lambda item: item["sample_id"])
        self.samples_by_id = {item["sample_id"]: item for item in self.samples}
        self.dimension = int(self.config["dimension"])
        self.fixed = fixed_config_view(self.config)
        self.schemes = {item["name"]: RoPEScheme.from_config(item) for item in self.config["schemes"]}
        self.global_version = int(global_version)
        self.parallel_default = parallel
        self.cache: dict[str, Any] = {}
        self.task_index: dict[str, str] = {}
        self.ledger: list[dict[str, Any]] = []
        self.code_counts: dict[str, int] = {}
        self.pending: list[str] = []
        self.cumulative_consumed = 0
        self.last_plan: dict[str, Any] | None = None
        self.last_report: dict[str, Any] | None = None
        self.history: list[dict[str, Any]] = []
        self._lock = __import__("threading").RLock()

    def config_hash(self) -> str:
        return fingerprint(self.config)

    def workspace_hash(self) -> str:
        return fingerprint({"config": self.config, "samples": self.samples})

    def apply_command(self, command: dict[str, Any]) -> dict[str, Any]:
        from .commands import apply_command

        next_config, next_samples, manifest = apply_command(self.config, self.samples, command)
        self.config = next_config
        self.samples = sorted(next_samples, key=lambda item: item["sample_id"])
        self.samples_by_id = {item["sample_id"]: item for item in self.samples}
        self.fixed = fixed_config_view(self.config)
        self.schemes = {item["name"]: RoPEScheme.from_config(item) for item in self.config["schemes"]}
        self.dimension = int(self.config["dimension"])
        self.global_version += 1
        return manifest

    def _count(self, code: str, amount: int = 1) -> None:
        self.code_counts[code] = int(self.code_counts.get(code, 0)) + amount

    def _fixed_fingerprint(self) -> str:
        return fingerprint(self.fixed)

    def _scheme_fingerprint(self, scheme_name: str) -> str:
        raw = next(item for item in self.config["schemes"] if item["name"] == scheme_name)
        return fingerprint({"rope_formula": "unified-rope-v1", "scheme": raw})

    def position_cache_key(self, sample_id: str, scheme_name: str, dimension: int) -> str:
        sample = self.samples_by_id[sample_id]
        refs = {
            "kind": "position",
            "sample_id": sample_id,
            "scheme": scheme_name,
            "dimension": dimension,
            "scheme_fp": self._scheme_fingerprint(scheme_name),
            "fixed_fp": self._fixed_fingerprint(),
            "positions_fp": fingerprint(
                {"query_position": sample["query_position"], "key_positions": sample["key_positions"]}
            ),
        }
        return cache_key("position", refs)

    def rotation_cache_key(self, sample_id: str, scheme_name: str, dimension: int) -> str:
        sample = self.samples_by_id[sample_id]
        position_key = self.position_cache_key(sample_id, scheme_name, dimension)
        refs = {
            "kind": "rotation",
            "position_key": position_key,
            "dimension_dependency_fp": dimension_dependency_fingerprint(sample, dimension),
            "fixed_fp": self._fixed_fingerprint(),
        }
        return cache_key("rotation", refs)

    def dot_term_cache_key(self, sample_id: str, scheme_name: str, dimension: int) -> str:
        refs = {
            "kind": "dot_term",
            "rotation_key": self.rotation_cache_key(sample_id, scheme_name, dimension),
            "dimension": dimension,
        }
        return cache_key("dot_term", refs)

    def score_cache_key(self, sample_id: str, scheme_name: str) -> str:
        refs = {
            "kind": "score",
            "sample_id": sample_id,
            "scheme": scheme_name,
            "geometry_fp": sample_geometry_fingerprint(self.samples_by_id[sample_id]),
            "dot_term_keys": [self.dot_term_cache_key(sample_id, scheme_name, d) for d in range(self.dimension)],
            "fixed_fp": self._fixed_fingerprint(),
        }
        return cache_key("score", refs)

    def normalization_cache_key(self, sample_id: str, scheme_name: str) -> str:
        return cache_key(
            "normalization",
            {"score_key": self.score_cache_key(sample_id, scheme_name), "fixed_fp": self._fixed_fingerprint()},
        )

    def invalid_sample_ids(self, manifest: dict[str, Any] | None) -> list[str]:
        if manifest is None:
            return sorted(self.samples_by_id)
        affected: set[str] = set()
        if manifest.get("fixed_point_changed"):
            affected.update(self.samples_by_id)
        if manifest.get("added_schemes"):
            affected.update(self.samples_by_id)
        changed_schemes = manifest.get("changed_schemes", {})
        if any(fields for fields in changed_schemes.values()):
            affected.update(self.samples_by_id)
        for sample_id, detail in manifest.get("changed_samples", {}).items():
            if detail.get("changed_dimensions"):
                affected.add(sample_id)
        for sample_id in manifest.get("replaced_samples", []):
            if sample_id in self.samples_by_id:
                affected.add(sample_id)
        return sorted(sid for sid in affected if sid in self.samples_by_id)

    def _position_payload(self, sample_id: str, scheme_name: str, dimension: int) -> dict[str, Any]:
        sample = self.samples_by_id[sample_id]
        scheme = self.schemes[scheme_name]
        encoding = quantized_encoding(
            scheme.angle_for_dimension(int(sample["query_position"]), dimension, self.dimension),
            self.fixed,
        )
        key_encodings = []
        elements = []
        self._append_position_elements(elements, "query", -1, dimension, encoding)
        for key_index, position in enumerate(sample["key_positions"]):
            key_encoding = quantized_encoding(
                scheme.angle_for_dimension(int(position), dimension, self.dimension),
                self.fixed,
            )
            key_encodings.append({"key_index": key_index, "position": int(position), "encoding": key_encoding})
            self._append_position_elements(elements, "key", key_index, dimension, key_encoding)
        return {
            "sample_id": sample_id,
            "scheme": scheme_name,
            "dimension_index": dimension,
            "query_encoding": encoding,
            "key_encodings": key_encodings,
            "error_elements": elements,
        }

    @staticmethod
    def _append_position_elements(elements: list[dict[str, Any]], role: str, key_index: int, dimension: int, enc: dict[str, Any]) -> None:
        for field_name in ("angle", "cos", "sin"):
            truth = float(enc[field_name])
            candidate = float(enc[f"{field_name}_fixed"])
            item = element_error(truth, candidate)
            item.update(
                {
                    "stage": "position",
                    "role": role,
                    "key_index": key_index,
                    "dimension_index": dimension,
                    "field": field_name,
                    "truth": truth,
                    "candidate": candidate,
                }
            )
            elements.append(item)

    def _rotation_payload(self, sample_id: str, scheme_name: str, dimension: int) -> dict[str, Any]:
        sample = self.samples_by_id[sample_id]
        scheme = self.schemes[scheme_name]
        query_record = rotate_single_dimension(
            sample["query_vector"], int(sample["query_position"]), dimension, scheme, self.dimension, self.fixed
        )
        key_records = []
        elements = []
        self._append_rotation_element(elements, "query", -1, dimension, query_record)
        for key_index, position in enumerate(sample["key_positions"]):
            record = rotate_single_dimension(
                sample["key_vectors"][key_index], int(position), dimension, scheme, self.dimension, self.fixed
            )
            key_records.append({"key_index": key_index, "position": int(position), "record": record})
            self._append_rotation_element(elements, "key", key_index, dimension, record)
        return {
            "sample_id": sample_id,
            "scheme": scheme_name,
            "dimension_index": dimension,
            "query": query_record,
            "keys": key_records,
            "error_elements": elements,
        }

    @staticmethod
    def _append_rotation_element(elements, role, key_index, dimension, record):
        truth = float(record["truth_value"])
        candidate = float(record["fixed_value"])
        item = element_error(truth, candidate)
        item.update(
            {
                "stage": "rotation",
                "role": role,
                "key_index": key_index,
                "dimension_index": dimension,
                "truth": truth,
                "candidate": candidate,
            }
        )
        elements.append(item)

    def _dot_term_payload(self, sample_id: str, scheme_name: str, dimension: int, rotation_key: str) -> dict[str, Any]:
        rotation_payload = self.cache[rotation_key]
        scale = 1 << int(self.fixed["fraction_bits"])
        query_rotated = rotation_payload["query"]["q_rotated"]
        terms = []
        for key_entry in rotation_payload["keys"]:
            key_index = key_entry["key_index"]
            key_rotated = key_entry["record"]["q_rotated"]
            q_term = int(query_rotated) * int(key_rotated)
            fixed_contribution = q_term / (scale * scale)
            truth_contribution = (
                float(rotation_payload["query"]["truth_value"]) * float(key_entry["record"]["truth_value"])
            )
            error = element_error(truth_contribution, fixed_contribution)
            terms.append(
                {
                    "key_index": key_index,
                    "q_term": q_term,
                    "fixed_contribution": fixed_contribution,
                    "truth_contribution": truth_contribution,
                    **error,
                }
            )
        return {
            "sample_id": sample_id,
            "scheme": scheme_name,
            "dimension_index": dimension,
            "terms": terms,
        }

    def _score_payload(self, sample_id: str, scheme_name: str, dot_keys: list[str]) -> dict[str, Any]:
        sample = self.samples_by_id[sample_id]
        scheme = self.schemes[scheme_name]
        bits = int(self.fixed["fraction_bits"])
        accumulator_bits = int(self.fixed["accumulator_bits"])
        scale_squared = (1 << bits) ** 2
        key_count = len(sample["key_positions"])
        rows = []

        angle_query = rotate_vector_angle_quantized(
            sample["query_vector"], int(sample["query_position"]), scheme, self.dimension, self.fixed
        )
        angle_keys = [
            rotate_vector_angle_quantized(vector, int(position), scheme, self.dimension, self.fixed)
            for vector, position in zip(sample["key_vectors"], sample["key_positions"])
        ]

        for key_index in range(key_count):
            q_terms = [self.cache[key]["terms"][key_index]["q_term"] for key in dot_keys]
            truth_terms = [self.cache[key]["terms"][key_index]["truth_contribution"] for key in dot_keys]

            wide_accumulator = 0
            truth_score = 0.0
            for dimension_index in range(self.dimension):
                wide_accumulator += int(q_terms[dimension_index])
                truth_score += float(truth_terms[dimension_index])

            wrapped_accumulator = 0
            overflow_count = 0
            overflow_steps = []
            for dimension_index, term in enumerate(q_terms):
                wrapped_accumulator += int(term)
                wrapped_accumulator, overflowed = wrap_signed(wrapped_accumulator, accumulator_bits)
                if overflowed:
                    overflow_count += 1
                    overflow_steps.append(dimension_index)

            angle_score = float_dot_product(angle_query, angle_keys[key_index])
            score_wide = wide_accumulator / scale_squared
            score_main = wrapped_accumulator / scale_squared
            if overflow_count:
                self._count("E_FIXED_OVERFLOW", overflow_count)

            rows.append(
                {
                    "key_index": key_index,
                    "position": int(sample["key_positions"][key_index]),
                    "q_score_wide": wide_accumulator,
                    "score_wide": score_wide,
                    "q_score": wrapped_accumulator,
                    "score": score_main,
                    "truth_score": truth_score,
                    "angle_quantized_score": angle_score,
                    "overflow_count": overflow_count,
                    "overflow_steps": overflow_steps,
                }
            )

        truth_scores = [row["truth_score"] for row in rows]
        main_scores = [row["score"] for row in rows]
        wide_scores = [row["score_wide"] for row in rows]
        angle_scores = [row["angle_quantized_score"] for row in rows]
        all_indexes = list(range(key_count))
        main_flips = ranking_pair_flips(all_indexes, truth_scores, main_scores)
        wide_flips = {
            (item["left_key_index"], item["right_key_index"])
            for item in ranking_pair_flips(all_indexes, truth_scores, wide_scores)
        }
        angle_flips = {
            (item["left_key_index"], item["right_key_index"])
            for item in ranking_pair_flips(all_indexes, truth_scores, angle_scores)
        }
        for flip in main_flips:
            pair = (flip["left_key_index"], flip["right_key_index"])
            if pair in angle_flips:
                flip["source"] = "angle_quantization"
            elif pair in wide_flips:
                flip["source"] = "rotation_rounding"
            else:
                flip["source"] = "accumulator_overflow"
            left = pair[0]
            right = pair[1]
            flip["left_position"] = int(sample["key_positions"][left])
            flip["right_position"] = int(sample["key_positions"][right])
        return {
            "sample_id": sample_id,
            "scheme": scheme_name,
            "rows": rows,
            "ranking_flips": main_flips,
        }

    def _normalization_payload(self, sample_id: str, scheme_name: str, score_key: str) -> dict[str, Any]:
        score_payload = self.cache[score_key]
        fixed_scores = [row["score"] for row in score_payload["rows"]]
        truth_scores = [row["truth_score"] for row in score_payload["rows"]]
        fixed_softmax = stable_softmax(fixed_scores)
        truth_softmax = stable_softmax(truth_scores)
        probabilities = []
        elements = []
        for key_index in range(len(fixed_scores)):
            fixed_probability = fixed_softmax["probabilities"][key_index]
            truth_probability = truth_softmax["probabilities"][key_index]
            error = element_error(truth_probability, fixed_probability)
            elements.append(
                {
                    "stage": "normalization",
                    "key_index": key_index,
                    "truth": truth_probability,
                    "candidate": fixed_probability,
                    **error,
                }
            )
            probabilities.append(
                {
                    "key_index": key_index,
                    "fixed_probability": fixed_probability,
                    "truth_probability": truth_probability,
                }
            )
        return {
            "sample_id": sample_id,
            "scheme": scheme_name,
            "fixed": fixed_softmax,
            "truth": truth_softmax,
            "probabilities": probabilities,
            "error_elements": elements,
        }

    def _spec(
        self,
        stable_key: str,
        cache_key: str,
        compute: Callable[[], dict[str, Any]],
        references: dict[str, Any],
        manifest: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "stable_key": stable_key,
            "cache_key": cache_key,
            "old_cache_key": self.task_index.get(stable_key),
            "compute": compute,
            "references": references,
            "manifest_op": (manifest or {}).get("op"),
        }

    def _resolve_spec(self, spec: dict[str, Any]) -> dict[str, Any]:
        cache_key = spec["cache_key"]
        with self._lock:
            if cache_key in self.cache:
                status = "reused"
            else:
                status = "recomputed"

        failure = None
        if status == "recomputed":
            try:
                payload = spec["compute"]()
            except Exception as exc:  # noqa: BLE001 - isolation records failure
                failure = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "code": getattr(exc, "code", None)
                    or (E_VALUE_NONFINITE if "non-finite" in str(exc) else E_STAGE_FAILURE),
                }
                status = "failed"
                self._count(failure["code"])
            else:
                with self._lock:
                    if cache_key not in self.cache:
                        self.cache[cache_key] = payload
                    else:
                        status = "reused"

        with self._lock:
            self.task_index[spec["stable_key"]] = cache_key
            entry = {
                "stable_key": spec["stable_key"],
                "sort_key": spec["stable_key"],
                "cache_key": cache_key,
                "old_cache_key": spec["old_cache_key"],
                "cache_status": status,
                "failure": failure,
                "key_changed": spec["old_cache_key"] not in (None, cache_key),
                "new_task": spec["old_cache_key"] is None,
                "manifest_op": spec.get("manifest_op"),
                "references": spec["references"],
            }
            if status == "reused":
                entry["reuse_hit"] = cache_key
                entry["invalidation_basis"] = ["content_fingerprint_match"]
            elif status == "failed":
                entry["invalidation_basis"] = ["stage_exception_isolated"]
                entry["error_code"] = failure["code"]
            elif spec["old_cache_key"] is None:
                entry["invalidation_basis"] = ["blank_cache_or_new_stable_task"]
            elif entry["key_changed"]:
                entry["invalidation_basis"] = ["content_fingerprint_changed"]
            else:
                entry["invalidation_basis"] = ["cache_miss_with_same_key"]
            return entry

    def _execute_stage(self, specs: list[dict[str, Any]], parallel: bool) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        use_threads = parallel and int(self.config.get("concurrency", 1)) > 1 and len(specs) > 1
        if use_threads:
            with ThreadPoolExecutor(max_workers=int(self.config["concurrency"])) as executor:
                futures = [executor.submit(self._resolve_spec, spec) for spec in specs]
                for future in as_completed(futures):
                    entries.append(future.result())
        else:
            for spec in specs:
                entries.append(self._resolve_spec(spec))
        return entries

    def run(self, manifest: dict[str, Any] | None = None, *, parallel: bool = True) -> dict[str, Any]:
        self.ledger = []
        invalid_ids = self.invalid_sample_ids(manifest)
        plan = plan_batch(
            invalid_ids,
            self.samples_by_id,
            self.dimension,
            int(self.config["point_budget"]),
            int(self.config["sample_quota"]),
            self.pending,
            self.cumulative_consumed,
        )
        self.last_plan = plan
        self.pending = list(plan["pending"])
        self.cumulative_consumed = int(plan["cumulative_consumed"])
        for event in plan["events"]:
            self._count(event["code"])

        admitted_ids = sorted(plan["admitted_sample_ids"])
        blocked_ids = set(plan["pending"]) | set(plan["quota_exceeded_sample_ids"])
        eligible_ids = [sample_id for sample_id in sorted(self.samples_by_id) if sample_id not in blocked_ids]
        scheme_names = sorted(self.schemes)

        position_specs = []
        rotation_specs = []
        dot_specs = []
        score_specs = []
        normalization_specs = []

        for sample_id in eligible_ids:
            for scheme_name in scheme_names:
                pos_keys = []
                rot_keys = []
                dot_keys = []
                for dimension in range(self.dimension):
                    pkey = self.position_cache_key(sample_id, scheme_name, dimension)
                    rkey = self.rotation_cache_key(sample_id, scheme_name, dimension)
                    dkey = self.dot_term_cache_key(sample_id, scheme_name, dimension)
                    pos_keys.append(pkey)
                    rot_keys.append(rkey)
                    dot_keys.append(dkey)
                    position_specs.append(
                        self._spec(
                            stable_task_key(sample_id, dimension, "position", scheme_name),
                            pkey,
                            lambda sid=sample_id, sn=scheme_name, d=dimension: self._position_payload(sid, sn, d),
                            {"sample_id": sample_id, "scheme": scheme_name, "dimension_index": dimension},
                            manifest,
                        )
                    )
                    rotation_specs.append(
                        self._spec(
                            stable_task_key(sample_id, dimension, "rotation", scheme_name),
                            rkey,
                            lambda sid=sample_id, sn=scheme_name, d=dimension: self._rotation_payload(sid, sn, d),
                            {"sample_id": sample_id, "scheme": scheme_name, "dimension_index": dimension, "position_key": pkey},
                            manifest,
                        )
                    )
                    dot_specs.append(
                        self._spec(
                            stable_task_key(sample_id, dimension, "dot_term", scheme_name),
                            dkey,
                            lambda sid=sample_id, sn=scheme_name, d=dimension, rk=rkey: self._dot_term_payload(sid, sn, d, rk),
                            {"sample_id": sample_id, "scheme": scheme_name, "dimension_index": dimension, "rotation_key": rkey},
                            manifest,
                        )
                    )

                score_key = self.score_cache_key(sample_id, scheme_name)
                norm_key = self.normalization_cache_key(sample_id, scheme_name)
                score_specs.append(
                    self._spec(
                        stable_task_key(sample_id, -1, "score", scheme_name),
                        score_key,
                        lambda sid=sample_id, sn=scheme_name, keys=list(dot_keys): self._score_payload(sid, sn, keys),
                        {"sample_id": sample_id, "scheme": scheme_name, "dot_term_keys": dot_keys},
                        manifest,
                    )
                )
                normalization_specs.append(
                    self._spec(
                        stable_task_key(sample_id, -1, "normalization", scheme_name),
                        norm_key,
                        lambda sid=sample_id, sn=scheme_name, sk=score_key: self._normalization_payload(sid, sn, sk),
                        {"sample_id": sample_id, "scheme": scheme_name, "score_key": score_key},
                        manifest,
                    )
                )

        for specs in (position_specs, rotation_specs, dot_specs, score_specs, normalization_specs):
            self.ledger.extend(self._execute_stage(specs, parallel))

        error_records = self._build_error_records(eligible_ids, scheme_names)
        self._current_error_records = error_records
        stats_records, matrix, extrap_flips = self._build_stats(eligible_ids, scheme_names)
        stats_entries = self._build_stats_cache(eligible_ids, scheme_names, stats_records, manifest, parallel)
        self.ledger.extend(stats_entries)

        self.ledger.sort(key=lambda entry: self._ledger_order(entry["stable_key"]))
        report = {
            "global_version": self.global_version,
            "config_hash": self.config_hash(),
            "workspace_hash": self.workspace_hash(),
            "parallel_requested": bool(parallel),
            "complete": len(self.pending) == 0 and not plan["quota_exceeded_sample_ids"],
            "sample_ids": [sample["sample_id"] for sample in self.samples],
            "processed_sample_ids": eligible_ids,
            "newly_admitted_sample_ids": admitted_ids,
            "pending_sample_ids": list(self.pending),
            "quota_exceeded_sample_ids": plan["quota_exceeded_sample_ids"],
            "config": self.config,
            "scheduler": plan,
            "code_counts": dict(sorted(self.code_counts.items())),
            "ledger": self.ledger,
            "error_records": error_records,
            "scheme_stage_matrix": matrix,
            "stats_records": stats_records,
            "extrapolation_flips": extrap_flips,
            "cache_stats": {
                "cache_entries": len(self.cache),
                "ledger_entries": len(self.ledger),
                "reused": sum(1 for entry in self.ledger if entry["cache_status"] == "reused"),
                "recomputed": sum(1 for entry in self.ledger if entry["cache_status"] == "recomputed"),
            },
        }
        self.last_report = report
        self.history.append(
            {
                "global_version": self.global_version,
                "manifest_op": (manifest or {}).get("op", "initial"),
                "config_hash": report["config_hash"],
                "workspace_hash": report["workspace_hash"],
                "cache_stats": report["cache_stats"],
                "pending_sample_ids": list(self.pending),
            }
        )
        return report

    @staticmethod
    def _ledger_order(stable_key: str) -> tuple[str, str, int, int]:
        parts = stable_key.split("|")
        sample_id = parts[0]
        dimension_part = parts[1] if len(parts) > 1 else "d-001"
        stage_part = parts[2].split(":", 1)[-1] if len(parts) > 2 else ""
        scheme_part = parts[3].split("=", 1)[-1] if len(parts) > 3 else ""
        dimension = int(dimension_part[1:]) if dimension_part.startswith("d") else -1
        return sample_id, scheme_part, STAGE_ORDER.get(stage_part, 99), dimension

    def _tolerance_pair(self, stage: str) -> tuple[float, float]:
        tolerance = self.config["tolerances"][stage]
        if isinstance(tolerance, dict):
            return float(tolerance["abs"]), float(tolerance["rel"])
        value = float(tolerance)
        return value, value

    def _find_ledger_cache_key(self, sample_id: str, scheme_name: str, dimension: int | None, stage: str) -> str | None:
        stable_key = stable_task_key(sample_id, dimension if dimension is not None else -1, stage, scheme_name)
        for entry in self.ledger:
            if entry["stable_key"] == stable_key:
                return entry["cache_key"]
        return self.task_index.get(stable_key)

    def _build_error_records(self, sample_ids: list[str], scheme_names: list[str]) -> list[dict[str, Any]]:
        records = []
        for sample_id in sample_ids:
            for scheme_name in scheme_names:
                first_found = False
                for error_stage in ERROR_STAGES:
                    elements: list[dict[str, Any]] = []
                    if error_stage == "position":
                        for dimension in range(self.dimension):
                            key = self._find_ledger_cache_key(sample_id, scheme_name, dimension, "position")
                            payload = self.cache.get(key) if key else None
                            if payload:
                                elements.extend(payload["error_elements"])
                    elif error_stage == "rotation":
                        for dimension in range(self.dimension):
                            key = self._find_ledger_cache_key(sample_id, scheme_name, dimension, "rotation")
                            payload = self.cache.get(key) if key else None
                            if payload:
                                elements.extend(payload["error_elements"])
                    elif error_stage == "dot":
                        key = self._find_ledger_cache_key(sample_id, scheme_name, None, "score")
                        payload = self.cache.get(key) if key else None
                        if payload:
                            for row in payload["rows"]:
                                item = element_error(row["truth_score"], row["score"])
                                item.update({"key_index": row["key_index"], "truth": row["truth_score"], "candidate": row["score"]})
                                elements.append(item)
                    else:
                        key = self._find_ledger_cache_key(sample_id, scheme_name, None, "normalization")
                        payload = self.cache.get(key) if key else None
                        if payload:
                            elements.extend(payload["error_elements"])

                    abs_tol, rel_tol = self._tolerance_pair(error_stage)
                    record = aggregate_element_errors(elements, abs_tol, rel_tol)
                    record.update(
                        {
                            "sample_id": sample_id,
                            "scheme": scheme_name,
                            "stage": error_stage,
                        }
                    )
                    if record["tolerance_exceeded"] and not first_found:
                        record["first_exceeded"] = True
                        first_found = True
                    records.append(record)
        return records

    def _build_stats(self, sample_ids: list[str], scheme_names: list[str]):
        stats_records = []
        matrix = []
        all_flips = []
        bins = [int(value) for value in self.config["extrapolation_bins"]]
        target_length = int(self.config["target_length"])

        for scheme_name in scheme_names:
            category_observations = {"inside": [], "extrapolated": []}
            bin_observations: dict[str, list[dict[str, Any]]] = {"inside": []}
            boundary = []
            scheme_flips = []

            for sample_id in sample_ids:
                sample = self.samples_by_id[sample_id]
                score_key = self._find_ledger_cache_key(sample_id, scheme_name, None, "score")
                norm_key = self._find_ledger_cache_key(sample_id, scheme_name, None, "normalization")
                score_payload = self.cache.get(score_key) if score_key else None
                norm_payload = self.cache.get(norm_key) if norm_key else None
                if not score_payload or not norm_payload:
                    continue
                probability_by_key = {item["key_index"]: item["fixed_probability"] for item in norm_payload["probabilities"]}
                train_length = int(sample["train_length"])

                for row in score_payload["rows"]:
                    key_index = row["key_index"]
                    classification = classify_key(
                        int(sample["query_position"]),
                        row["position"],
                        train_length,
                        target_length,
                        bins,
                    )
                    observation = {
                        "sample_id": sample_id,
                        "key_index": key_index,
                        "position": row["position"],
                        "score": row["score"],
                        "truth_score": row["truth_score"],
                        "probability": probability_by_key[key_index],
                        **classification,
                    }
                    category_observations[classification["category"]].append(observation)
                    bin_observations.setdefault(classification["bin"], []).append(observation)
                    if classification["overrun"] in (0, 1):
                        boundary.append(observation)

                extrap_indexes = {
                    obs["key_index"]
                    for obs in category_observations["extrapolated"]
                    if obs["sample_id"] == sample_id
                }
                for flip in score_payload["ranking_flips"]:
                    if flip["left_key_index"] in extrap_indexes and flip["right_key_index"] in extrap_indexes:
                        flip_record = {"sample_id": sample_id, "scheme": scheme_name, **flip}
                        scheme_flips.append(flip_record)
                        all_flips.append(flip_record)

            categories = {
                category: self._aggregate_observations(observations)
                for category, observations in sorted(category_observations.items())
            }
            bin_stats = {
                bin_name: self._aggregate_observations(observations)
                for bin_name, observations in sorted(bin_observations.items())
            }
            stats_record = {
                "scheme": scheme_name,
                "train_length": int(self.config["train_length"]),
                "target_length": target_length,
                "categories": categories,
                "bins": bin_stats,
                "boundary_samples": sorted(boundary, key=lambda item: (item["sample_id"], item["key_index"])),
                "ranking_flip_count": len(scheme_flips),
            }
            stats_records.append(stats_record)

            for stage in ERROR_STAGES:
                stage_errors = [
                    record
                    for record in getattr(self, "_current_error_records", [])
                    if record["scheme"] == scheme_name and record["stage"] == stage
                ]
                exceeded_samples = [record["sample_id"] for record in stage_errors if record["tolerance_exceeded"]]
                matrix.append(
                    {
                        "scheme": scheme_name,
                        "stage": stage,
                        "sample_count": len(stage_errors),
                        "exceeded_count": len(exceeded_samples),
                        "first_exceeded_count": sum(1 for record in stage_errors if record["first_exceeded"]),
                        "max_abs_error": max((record["abs_error"] for record in stage_errors), default=0.0),
                        "max_rel_error": max((record["rel_error"] for record in stage_errors), default=0.0),
                        "exceeded_samples": sorted(exceeded_samples),
                    }
                )
        return stats_records, matrix, sorted(all_flips, key=lambda item: (item["scheme"], item["sample_id"], item["left_key_index"], item["right_key_index"]))

    @staticmethod
    def _aggregate_observations(observations: list[dict[str, Any]]) -> dict[str, Any]:
        if not observations:
            return {
                "sample_count": 0,
                "observation_count": 0,
                "score_mean": 0.0,
                "score_max_deviation": 0.0,
                "truth_score_mean": 0.0,
                "score_max_abs_error": 0.0,
                "beyond_target_count": 0,
            }
        sample_ids = {obs["sample_id"] for obs in observations}
        score_total = 0.0
        truth_total = 0.0
        for obs in observations:
            score_total += float(obs["score"])
            truth_total += float(obs["truth_score"])
        score_mean = score_total / len(observations)
        truth_mean = truth_total / len(observations)
        max_deviation = 0.0
        max_abs_error = 0.0
        for obs in observations:
            max_deviation = max(max_deviation, abs(float(obs["score"]) - score_mean))
            max_abs_error = max(max_abs_error, abs(float(obs["score"]) - float(obs["truth_score"])))
        return {
            "sample_count": len(sample_ids),
            "observation_count": len(observations),
            "score_mean": score_mean,
            "score_max_deviation": max_deviation,
            "truth_score_mean": truth_mean,
            "score_max_abs_error": max_abs_error,
            "beyond_target_count": sum(1 for obs in observations if obs.get("beyond_target")),
        }

    def _build_stats_cache(self, sample_ids, scheme_names, stats_records, manifest, parallel):
        entries = []
        stats_by_scheme = {record["scheme"]: record for record in stats_records}
        for scheme_name in scheme_names:
            norm_keys = [
                self._find_ledger_cache_key(sample_id, scheme_name, None, "normalization")
                for sample_id in sample_ids
            ]
            key = cache_key(
                "stats",
                {
                    "scheme": scheme_name,
                    "normalization_keys": sorted(key for key in norm_keys if key),
                    "classification_fp": fingerprint(
                        {
                            "config_train_length": self.config["train_length"],
                            "target_length": self.config["target_length"],
                            "bins": self.config["extrapolation_bins"],
                            "sample_markers": {sid: self.samples_by_id[sid]["train_length"] for sid in sample_ids},
                        }
                    ),
                },
            )
            stable = stable_task_key("*SCHEME*", -1, "stats", scheme_name)
            old_key = self.task_index.get(stable)
            if key not in self.cache:
                self.cache[key] = stats_by_scheme[scheme_name]
                status = "recomputed"
            else:
                status = "reused"
            self.task_index[stable] = key
            entries.append(
                {
                    "stable_key": stable,
                    "sort_key": stable,
                    "cache_key": key,
                    "old_cache_key": old_key,
                    "cache_status": status,
                    "key_changed": old_key not in (None, key),
                    "new_task": old_key is None,
                    "manifest_op": (manifest or {}).get("op"),
                    "references": {"scheme": scheme_name, "sample_ids": sample_ids},
                    "reuse_hit": key if status == "reused" else None,
                    "invalidation_basis": ["classification_or_score_fingerprint"] if status == "recomputed" else ["content_fingerprint_match"],
                }
            )
        return entries

    def semantic_report(self) -> dict[str, Any]:
        report = self.last_report or {}
        return {
            "global_version": report.get("global_version"),
            "config_hash": report.get("config_hash"),
            "workspace_hash": report.get("workspace_hash"),
            "complete": report.get("complete"),
            "sample_ids": report.get("sample_ids"),
            "processed_sample_ids": report.get("processed_sample_ids"),
            "pending_sample_ids": report.get("pending_sample_ids"),
            "config": report.get("config"),
            "error_records": report.get("error_records"),
            "scheme_stage_matrix": report.get("scheme_stage_matrix"),
            "stats_records": report.get("stats_records"),
            "extrapolation_flips": report.get("extrapolation_flips"),
            "dimension_outputs": self.dimension_outputs(),
            "row_outputs": self.row_outputs(),
        }

    def dimension_outputs(self) -> list[dict[str, Any]]:
        outputs = []
        for entry in sorted(self.ledger, key=lambda item: self._ledger_order(item["stable_key"])):
            if not any(stage_token in entry["stable_key"] for stage_token in (":position", ":rotation", ":dot_term")):
                continue
            payload = self.cache.get(entry["cache_key"])
            outputs.append(
                {
                    "stable_key": entry["stable_key"],
                    "cache_key": entry["cache_key"],
                    "payload": payload,
                }
            )
        return outputs

    def row_outputs(self) -> list[dict[str, Any]]:
        outputs = []
        for entry in sorted(self.ledger, key=lambda item: self._ledger_order(item["stable_key"])):
            if not any(stage_token in entry["stable_key"] for stage_token in (":score", ":normalization")):
                continue
            outputs.append({"stable_key": entry["stable_key"], "cache_key": entry["cache_key"], "payload": self.cache.get(entry["cache_key"])})
        return outputs
