"""Content-addressed stage cache and deterministic incremental engine."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from typing import Any

from .codes import Code
from .config import QuantConfig, RuntimeConfig, make_config
from .decoder import viterbi_decode
from .entities import merge_entities
from .features import emission_scores, feature_coordinates, sample_lexicon_refs
from .io_utils import fingerprint
from .labels import LABEL_CODES
from .validation import parse_corpus, parse_lexicon


STAGES = ("features", "emission", "viterbi", "merge")
STAGE_INDEX = {stage: index for index, stage in enumerate(STAGES)}


class BundleLoadError(RuntimeError):
    def __init__(self, failures: list[dict[str, Any]]):
        super().__init__("input bundle validation failed")
        self.failures = failures


class IncrementalEngine:
    def __init__(
        self,
        package: dict[str, Any],
        config: RuntimeConfig | None = None,
        samples: list[dict[str, Any]] | None = None,
        lexicon: list[dict[str, str]] | None = None,
        corpus_failures: list[dict[str, Any]] | None = None,
        lexicon_failures: list[dict[str, Any]] | None = None,
    ) -> None:
        self.config = config or RuntimeConfig()
        self.package = package
        self.samples: dict[str, dict[str, Any]] = {item["id"]: item for item in samples or []}
        self.lexicon: dict[str, dict[str, str]] = {item["id"]: item for item in lexicon or []}
        self.cache: dict[str, dict[str, Any]] = {}
        self.stage_index: dict[str, dict[str, str]] = {sample_id: {} for sample_id in self.samples}
        self.command_cursor = 0
        self.generation = 0
        self.snapshot_version = 0
        self.todo_queue: list[dict[str, str]] = []
        self.corpus_failures = corpus_failures or []
        self.lexicon_failures = lexicon_failures or []
        self.command_failures: list[dict[str, Any]] = []
        self.invalidation_log: list[dict[str, Any]] = []
        self.last_run: dict[str, list[dict[str, Any]]] = {"reused": [], "recomputed": []}
        self.counters = {"commands": 0, "invalidated": 0, "reused": 0, "recomputed": 0}
        self.package_fps = self._package_fingerprints()

    @classmethod
    def from_bundle(
        cls,
        corpus_path: str,
        lexicon_path: str,
        package_result: dict[str, Any],
        config: RuntimeConfig | None = None,
    ) -> "IncrementalEngine":
        corpus = parse_corpus(corpus_path)
        lexicon = parse_lexicon(lexicon_path)
        if not package_result.get("ok"):
            raise BundleLoadError(package_result.get("failures", []))
        return cls(
            package=package_result["package"],
            config=config,
            samples=corpus["samples"],
            lexicon=lexicon["entries"],
            corpus_failures=corpus["failures"],
            lexicon_failures=lexicon["failures"],
        )

    def _package_fingerprints(self) -> dict[str, str]:
        package = self.package
        return {
            "package": fingerprint(package),
            "structural": fingerprint(
                {
                    "feature_dim": package["feature_dim"],
                    "bucket_count": package["bucket_count"],
                    "label_encoding": package.get("label_encoding"),
                    "hash_algorithm": package.get("hash_algorithm"),
                }
            ),
            "hash_table": fingerprint(package["hash_table"]),
            "emission": fingerprint(package["emission_weights"]),
            "transition": fingerprint(package["transition_weights"]),
            "model_version": fingerprint(package.get("model_version")),
        }

    @property
    def sample_ids(self) -> list[str]:
        return sorted(self.samples)

    @property
    def lexicon_list(self) -> list[dict[str, str]]:
        return [self.lexicon[entry_id] for entry_id in sorted(self.lexicon)]

    def config_hash(self) -> str:
        return self.config.fingerprint()

    def business_counts(self, include_command_failures: bool = True) -> dict[str, int]:
        counts: dict[str, int] = {}
        failures = [*self.corpus_failures, *self.lexicon_failures]
        if include_command_failures:
            failures = [*failures, *self.command_failures]
        for failure in failures:
            code = failure["code"]
            counts[code] = counts.get(code, 0) + 1
        return dict(sorted(counts.items()))

    def _lexicon_deps(self, refs: list[str]) -> dict[str, str]:
        return {entry_id: fingerprint(self.lexicon[entry_id]) for entry_id in sorted(refs) if entry_id in self.lexicon}

    def _make_key(self, stage: str, key_body: dict[str, Any]) -> str:
        return f"{stage}:{fingerprint(key_body)}"

    def _store(self, stage: str, sample_id: str, key: str, payload: Any, upstream: str | None, deps: dict[str, Any]) -> dict[str, Any]:
        envelope = {
            "stage": stage,
            "sample_id": sample_id,
            "cache_key": key,
            "upstream_fingerprint": upstream,
            "dependency_fingerprints": deps,
            "payload_fingerprint": fingerprint(payload),
            "payload": payload,
        }
        self.cache[key] = envelope
        self.stage_index.setdefault(sample_id, {})[stage] = key
        return envelope

    def run_incremental(self) -> dict[str, list[dict[str, Any]]]:
        reused: list[dict[str, Any]] = []
        recomputed: list[dict[str, Any]] = []
        live_keys: set[str] = set()

        for sample_id in self.sample_ids:
            sample = self.samples[sample_id]
            self.stage_index.setdefault(sample_id, {})
            refs = sample_lexicon_refs(sample["chars"], self.lexicon_list)
            lex_deps = self._lexicon_deps(refs)

            feature_key = self._make_key(
                "features",
                {
                    "stage": "features",
                    "sample_id": sample_id,
                    "chars": sample["chars"],
                    "lexicon_deps": lex_deps,
                    "structural_fp": self.package_fps["structural"],
                },
            )
            cached = self.cache.get(feature_key)
            if cached is not None:
                feature_envelope = cached
                reused.append({"sample_id": sample_id, "stage": "features", "cache_key": feature_key, "basis": "content_key_hit"})
                self.counters["reused"] += 1
            else:
                feature_payload = feature_coordinates(sample, self.lexicon_list, self.package["feature_dim"], self.package["bucket_count"])
                feature_envelope = self._store("features", sample_id, feature_key, feature_payload, None, {"lexicon": lex_deps, "structure": self.package_fps["structural"]})
                recomputed.append({"sample_id": sample_id, "stage": "features", "cache_key": feature_key, "basis": "cache_miss"})
                self.counters["recomputed"] += 1
            self.stage_index[sample_id]["features"] = feature_key
            live_keys.add(feature_key)

            emission_key = self._make_key(
                "emission",
                {
                    "stage": "emission",
                    "upstream_fp": feature_envelope["payload_fingerprint"],
                    "hash_table_fp": self.package_fps["hash_table"],
                    "emission_fp": self.package_fps["emission"],
                    "quant_fp": self.config.quant.fingerprint(),
                },
            )
            cached = self.cache.get(emission_key)
            if cached is not None:
                emission_envelope = cached
                reused.append({"sample_id": sample_id, "stage": "emission", "cache_key": emission_key, "basis": "content_key_hit"})
                self.counters["reused"] += 1
            else:
                emission_payload = emission_scores(feature_envelope["payload"], self.package, self.config.quant)
                emission_envelope = self._store("emission", sample_id, emission_key, emission_payload, feature_envelope["payload_fingerprint"], {"hash_table": self.package_fps["hash_table"], "emission_weights": self.package_fps["emission"], "quant": self.config.quant.fingerprint()})
                recomputed.append({"sample_id": sample_id, "stage": "emission", "cache_key": emission_key, "basis": "cache_miss"})
                self.counters["recomputed"] += 1
            self.stage_index[sample_id]["emission"] = emission_key
            live_keys.add(emission_key)

            viterbi_key = self._make_key(
                "viterbi",
                {
                    "stage": "viterbi",
                    "upstream_fp": emission_envelope["payload_fingerprint"],
                    "transition_fp": self.package_fps["transition"],
                    "model_version_fp": self.package_fps["model_version"],
                },
            )
            cached = self.cache.get(viterbi_key)
            if cached is not None:
                viterbi_envelope = cached
                reused.append({"sample_id": sample_id, "stage": "viterbi", "cache_key": viterbi_key, "basis": "content_key_hit"})
                self.counters["reused"] += 1
            else:
                viterbi_payload = viterbi_decode(emission_envelope["payload"], self.package)
                viterbi_envelope = self._store("viterbi", sample_id, viterbi_key, viterbi_payload, emission_envelope["payload_fingerprint"], {"transition_weights": self.package_fps["transition"]})
                recomputed.append({"sample_id": sample_id, "stage": "viterbi", "cache_key": viterbi_key, "basis": "cache_miss"})
                self.counters["recomputed"] += 1
            self.stage_index[sample_id]["viterbi"] = viterbi_key
            live_keys.add(viterbi_key)

            merge_key = self._make_key(
                "merge",
                {
                    "stage": "merge",
                    "upstream_fp": viterbi_envelope["payload_fingerprint"],
                    "lexicon_deps": lex_deps,
                },
            )
            cached = self.cache.get(merge_key)
            if cached is not None:
                merge_envelope = cached
                reused.append({"sample_id": sample_id, "stage": "merge", "cache_key": merge_key, "basis": "content_key_hit"})
                self.counters["reused"] += 1
            else:
                merge_payload = merge_entities(sample, viterbi_envelope["payload"], self.lexicon_list)
                merge_envelope = self._store("merge", sample_id, merge_key, merge_payload, viterbi_envelope["payload_fingerprint"], {"lexicon": lex_deps})
                recomputed.append({"sample_id": sample_id, "stage": "merge", "cache_key": merge_key, "basis": "cache_miss"})
                self.counters["recomputed"] += 1
            self.stage_index[sample_id]["merge"] = merge_key
            live_keys.add(merge_key)

        for stale_key in list(self.cache):
            if stale_key not in live_keys:
                del self.cache[stale_key]
        for sample_id in list(self.stage_index):
            if sample_id not in self.samples:
                self.stage_index.pop(sample_id, None)

        self.last_run = {"reused": reused, "recomputed": recomputed}
        self.todo_queue = []
        return self.last_run

    def _samples_containing(self, text: str) -> set[str]:
        if not text:
            return set()
        return {sample_id for sample_id, sample in self.samples.items() if text in "".join(sample["chars"])}

    def _samples_referencing_entry(self, entry_id: str) -> set[str]:
        affected: set[str] = set()
        for sample_id, keys in self.stage_index.items():
            feature_key = keys.get("features")
            envelope = self.cache.get(feature_key) if feature_key else None
            if envelope and entry_id in envelope["payload"].get("lexicon_refs", []):
                affected.add(sample_id)
        return affected

    def invalidate(self, sample_ids: set[str] | list[str], from_stage: str, basis: dict[str, Any]) -> list[dict[str, Any]]:
        from_index = STAGE_INDEX[from_stage]
        records: list[dict[str, Any]] = []
        for sample_id in sorted(sample_ids):
            keys = self.stage_index.get(sample_id, {})
            for stage in STAGES[from_index:]:
                key = keys.pop(stage, None)
                if key and key in self.cache:
                    del self.cache[key]
                    record = {"sample_id": sample_id, "stage": stage, "cache_key": key, "basis": basis}
                    records.append(record)
                    self.invalidation_log.append(record)
                    self.counters["invalidated"] += 1
                self.todo_queue.append({"sample_id": sample_id, "stage": stage})
        return records

    def _command_failure(self, seq: int, op: str, detail: str, code: str = Code.COMMAND_VALIDATION) -> dict[str, Any]:
        failure = {"seq": seq, "op": op, "code": code, "message": detail}
        self.command_failures.append(failure)
        return failure

    def apply_command(self, command: dict[str, Any]) -> dict[str, Any]:
        seq = self.command_cursor + 1
        op = command.get("op") if isinstance(command, dict) else None
        self.generation += 1
        self.counters["commands"] += 1
        if not isinstance(op, str):
            failure = self._command_failure(seq, str(op), "命令必须包含字符串 op", Code.UNKNOWN_COMMAND)
            self.command_cursor = seq
            return {"ok": False, "failure": failure, "affected_samples": [], "invalidations": []}

        affected: set[str] = set()
        from_stage = "features"
        reason: dict[str, Any] = {"command_seq": seq, "op": op}

        if op in {"upsert_lexicon", "add_lexicon", "update_lexicon"}:
            entry = command.get("entry") if "entry" in command else {k: v for k, v in command.items() if k != "op"}
            parsed = parse_lexicon({"entries": [entry]})
            if parsed["failures"]:
                failure = self._command_failure(seq, op, parsed["failures"][0]["message"])
                self.command_cursor = seq
                return {"ok": False, "failure": failure, "affected_samples": [], "invalidations": []}
            normalized = parsed["entries"][0]
            entry_id = normalized["id"]
            old = self.lexicon.get(entry_id)
            affected.update(self._samples_referencing_entry(entry_id))
            if old:
                affected.update(self._samples_containing(old["text"]))
                reason["changed_fields"] = sorted({key for key in set(old) | set(normalized) if old.get(key) != normalized.get(key)})
            affected.update(self._samples_containing(normalized["text"]))
            self.lexicon[entry_id] = normalized
            reason["entry_id"] = entry_id
            reason["kind"] = "lexicon_upsert"

        elif op in {"remove_lexicon", "delete_lexicon"}:
            entry_id = command.get("entry_id") or command.get("id")
            old = self.lexicon.get(entry_id) if isinstance(entry_id, str) else None
            if old is None:
                failure = self._command_failure(seq, op, f"词表条目不存在：{entry_id}")
                self.command_cursor = seq
                return {"ok": False, "failure": failure, "affected_samples": [], "invalidations": []}
            affected.update(self._samples_referencing_entry(entry_id))
            affected.update(self._samples_containing(old["text"]))
            self.lexicon.pop(entry_id, None)
            reason.update({"entry_id": entry_id, "old_text": old["text"], "kind": "lexicon_remove"})

        elif op in {"replace_sample", "upsert_sample", "add_sample"}:
            raw_sample = command.get("sample", command)
            parsed = parse_corpus([json.dumps(raw_sample, ensure_ascii=False)])
            if parsed["failures"] or not parsed["samples"]:
                failure = self._command_failure(seq, op, parsed["failures"][0]["message"] if parsed["failures"] else "样本解析失败")
                self.command_cursor = seq
                return {"ok": False, "failure": failure, "affected_samples": [], "invalidations": []}
            sample = parsed["samples"][0]
            sample_id = sample["id"]
            affected.add(sample_id)
            self.samples[sample_id] = sample
            self.stage_index.setdefault(sample_id, {})
            reason.update({"sample_id": sample_id, "kind": "sample_replace"})

        elif op in {"delete_sample", "remove_sample"}:
            sample_id = command.get("sample_id") or command.get("id")
            if sample_id not in self.samples:
                failure = self._command_failure(seq, op, f"样本不存在：{sample_id}")
                self.command_cursor = seq
                return {"ok": False, "failure": failure, "affected_samples": [], "invalidations": []}
            affected.add(sample_id)
            for stage in STAGES:
                key = self.stage_index.get(sample_id, {}).pop(stage, None)
                if key:
                    self.cache.pop(key, None)
            self.samples.pop(sample_id, None)
            self.stage_index.pop(sample_id, None)
            reason.update({"sample_id": sample_id, "kind": "sample_delete"})

        elif op == "set_emission_weight":
            label = command.get("label")
            feature_index = command.get("feature_index")
            bucket = command.get("bucket")
            value = command.get("value")
            if label not in LABEL_CODES or not isinstance(feature_index, int) or not (0 <= feature_index < self.package["feature_dim"]) or not isinstance(bucket, int) or not (0 <= bucket < self.package["bucket_count"]) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                failure = self._command_failure(seq, op, "发射权重定位或 value 非法/非有限")
                self.command_cursor = seq
                return {"ok": False, "failure": failure, "affected_samples": [], "invalidations": []}
            old_value = self.package["emission_weights"][LABEL_CODES[label]][feature_index][bucket]
            self.package["emission_weights"][LABEL_CODES[label]][feature_index][bucket] = float(value)
            self.package_fps = self._package_fingerprints()
            if old_value != float(value):
                affected.update(self.sample_ids)
            from_stage = "emission"
            reason.update({"label": label, "feature_index": feature_index, "bucket": bucket, "kind": "emission_weight"})

        elif op == "set_transition_weight":
            value = command.get("value")
            section = command.get("section", "matrix")
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                failure = self._command_failure(seq, op, "转移权重 value 必须是有限数")
                self.command_cursor = seq
                return {"ok": False, "failure": failure, "affected_samples": [], "invalidations": []}
            if section == "matrix":
                from_label = command.get("from")
                to_label = command.get("to")
                if from_label not in LABEL_CODES or to_label not in LABEL_CODES:
                    failure = self._command_failure(seq, op, "from/to 标签非法")
                    self.command_cursor = seq
                    return {"ok": False, "failure": failure, "affected_samples": [], "invalidations": []}
                old_value = self.package["transition_weights"]["matrix"][LABEL_CODES[from_label]][LABEL_CODES[to_label]]
                self.package["transition_weights"]["matrix"][LABEL_CODES[from_label]][LABEL_CODES[to_label]] = float(value)
                reason.update({"from": from_label, "to": to_label})
            elif section in {"start", "end"}:
                label = command.get("label")
                if label not in LABEL_CODES:
                    failure = self._command_failure(seq, op, "label 非法")
                    self.command_cursor = seq
                    return {"ok": False, "failure": failure, "affected_samples": [], "invalidations": []}
                old_value = self.package["transition_weights"][section][LABEL_CODES[label]]
                self.package["transition_weights"][section][LABEL_CODES[label]] = float(value)
                reason.update({"section": section, "label": label})
            else:
                failure = self._command_failure(seq, op, "transition section 必须是 matrix/start/end")
                self.command_cursor = seq
                return {"ok": False, "failure": failure, "affected_samples": [], "invalidations": []}
            self.package_fps = self._package_fingerprints()
            if old_value != float(value):
                affected.update(self.sample_ids)
            from_stage = "viterbi"
            reason["kind"] = "transition_weight"

        elif op == "set_quant":
            quant_data = asdict(self.config.quant)
            for field in ("decimal_bits", "quant_level", "total_width"):
                if field in command:
                    quant_data[field] = command[field]
            try:
                new_quant = QuantConfig(**quant_data)
            except (TypeError, ValueError) as exc:
                failure = self._command_failure(seq, op, f"量化配置非法：{exc}")
                self.command_cursor = seq
                return {"ok": False, "failure": failure, "affected_samples": [], "invalidations": []}
            old_fp = self.config.quant.fingerprint()
            self.config = replace(self.config, quant=new_quant)
            if new_quant.fingerprint() != old_fp:
                affected.update(self.sample_ids)
            from_stage = "emission"
            reason.update({"quant": asdict(new_quant), "kind": "quant_config"})

        else:
            failure = self._command_failure(seq, op, "未知命令", Code.UNKNOWN_COMMAND)
            self.command_cursor = seq
            return {"ok": False, "failure": failure, "affected_samples": [], "invalidations": []}

        invalidations = self.invalidate(affected, from_stage, reason) if affected else []
        self.command_cursor = seq
        return {
            "ok": True,
            "seq": seq,
            "op": op,
            "affected_samples": sorted(affected),
            "from_stage": from_stage,
            "invalidations": invalidations,
            "reason": reason,
        }

    def apply_commands(self, commands: list[dict[str, Any]], run: bool = True) -> list[dict[str, Any]]:
        results = []
        while self.command_cursor < len(commands):
            command = commands[self.command_cursor]
            result = self.apply_command(command)
            results.append(result)
            if run:
                self.run_incremental()
        return results

    def authoritative_result(self) -> dict[str, Any]:
        if any(sample_id not in self.stage_index or "merge" not in self.stage_index.get(sample_id, {}) for sample_id in self.sample_ids):
            self.run_incremental()

        sample_results = []
        all_entities = []
        all_suppressions = []
        all_repairs = []
        cache_key_map: dict[str, dict[str, str]] = {}
        blank_plan = []

        for order_index, sample_id in enumerate(self.sample_ids):
            sample = self.samples[sample_id]
            keys = self.stage_index[sample_id]
            cache_key_map[sample_id] = {stage: keys[stage] for stage in STAGES}
            merge_payload = self.cache[keys["merge"]]["payload"]
            viterbi_payload = self.cache[keys["viterbi"]]["payload"]
            entities = merge_payload["entities"]
            suppressions = merge_payload["suppressions"]
            repairs = merge_payload["repairs"]
            all_entities.extend(entities)
            all_suppressions.extend(suppressions)
            all_repairs.extend(repairs)
            sample_results.append(
                {
                    "sample_id": sample_id,
                    "chars": sample["chars"],
                    "gold_labels": sample["labels"],
                    "decoded_labels": viterbi_payload["labels"],
                    "repaired_labels": merge_payload["repaired_labels"],
                    "path_total": viterbi_payload["path_total"],
                    "tie_positions": viterbi_payload["tie_positions"],
                    "entity_count": len(entities),
                    "type_distribution": merge_payload["type_distribution"],
                    "cache_keys": {stage: keys[stage] for stage in STAGES},
                }
            )
            for stage in STAGES:
                blank_plan.append({"order": order_index * len(STAGES) + STAGE_INDEX[stage], "sample_id": sample_id, "stage": stage, "cache_key": keys[stage], "blank_state_action": "recompute"})

        type_distribution = {entity_type: 0 for entity_type in ["LOC", "ORG", "PER", "TIME"]}
        for entity in all_entities:
            type_distribution[entity["type"]] += 1

        return {
            "schema_version": "authoritative-v1",
            "model_version": self.package["model_version"],
            "config_fingerprint": self.config_hash(),
            "package_fingerprint": self.package_fps["package"],
            "sample_count": len(self.samples),
            "samples": sample_results,
            "entities": all_entities,
            "suppressions": all_suppressions,
            "repairs": all_repairs,
            "type_distribution": type_distribution,
            "cache_key_map": cache_key_map,
            "live_cache_keys": sorted(self.cache),
            "execution_manifest": {
                "stage_order": list(STAGES),
                "blank_recompute_plan": blank_plan,
                "content_addressed_keys": sorted(self.cache),
                "note": "实际命中/失效是历史相关账本；blank_recompute_plan 是仅由最终内容派生的规范化全量清单。",
            },
            "business_counts": self.business_counts(include_command_failures=False),
        }

    def summary(self, stale: bool = False, code: str = Code.OK) -> dict[str, Any]:
        result = self.authoritative_result()
        affected = sorted({item["sample_id"] for item in self.last_run.get("recomputed", [])})
        total = len(self.samples)
        completed = total if not self.todo_queue else max(0, total - len({item["sample_id"] for item in self.todo_queue}))
        return {
            "code": code,
            "stale": stale,
            "generation": self.generation,
            "command_cursor": self.command_cursor,
            "completed_samples": completed,
            "total_samples": total,
            "progress": 1.0 if total and not self.todo_queue else (completed / total if total else 1.0),
            "entity_count": len(result["entities"]),
            "type_distribution": result["type_distribution"],
            "affected_samples": affected,
            "recomputed_count": len(self.last_run.get("recomputed", [])),
            "reused_count": len(self.last_run.get("reused", [])),
            "business_counts": self.business_counts(),
        }

    def to_persistent_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "config": asdict(self.config),
            "package": self.package,
            "samples": [self.samples[sample_id] for sample_id in self.sample_ids],
            "lexicon": self.lexicon_list,
            "command_cursor": self.command_cursor,
            "generation": self.generation,
            "snapshot_version": self.snapshot_version,
            "stage_index": self.stage_index,
            "todo_queue": self.todo_queue,
            "corpus_failures": self.corpus_failures,
            "lexicon_failures": self.lexicon_failures,
            "command_failures": self.command_failures,
            "invalidation_log": self.invalidation_log,
            "last_run": self.last_run,
            "counters": self.counters,
        }

    @classmethod
    def from_persistent_dict(cls, state: dict[str, Any], cache_by_stage: dict[str, list[dict[str, Any]]]) -> "IncrementalEngine":
        config = make_config(state["config"])
        engine = cls(
            package=state["package"],
            config=config,
            samples=state["samples"],
            lexicon=state["lexicon"],
            corpus_failures=state.get("corpus_failures", []),
            lexicon_failures=state.get("lexicon_failures", []),
        )
        for envelopes in cache_by_stage.values():
            for envelope in envelopes:
                engine.cache[envelope["cache_key"]] = envelope
        engine.command_cursor = state["command_cursor"]
        engine.generation = state["generation"]
        engine.snapshot_version = state["snapshot_version"]
        engine.stage_index = state["stage_index"]
        engine.todo_queue = state["todo_queue"]
        engine.command_failures = state.get("command_failures", [])
        engine.invalidation_log = state.get("invalidation_log", [])
        engine.last_run = state.get("last_run", {"reused": [], "recomputed": []})
        engine.counters = state.get("counters", engine.counters)
        engine.package_fps = engine._package_fingerprints()
        return engine
