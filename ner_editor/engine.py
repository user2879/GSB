"""Incremental, deterministic NER editor coordinator."""

from __future__ import annotations

import copy
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable

from .constants import (
    E_COMMAND_INVALID,
    E_EMISSION_INVALID,
    E_LATENCY_BUDGET_EXCEEDED,
    E_MODEL_PARAM_INVALID,
    LABELS,
    STAGE_ORDER,
    TYPE_CODES,
)
from .decoder import (
    decode_emissions,
    decode_key,
    decode_output_fingerprint,
    transition_parameter_fingerprint,
)
from .emission import (
    emission_key,
    emission_output_fingerprint,
    emission_parameter_fingerprint,
    score_emissions,
)
from .features import (
    build_features,
    feature_key,
    feature_output_fingerprint,
    refresh_feature_refs,
)
from .loader import (
    config_fingerprint,
    corpus_fingerprint,
    model_fingerprint,
    validate_vocab_entry,
    vocabulary_fingerprint,
)
from .merging import merge_entities, merge_key, merge_output_fingerprint
from .models import (
    CommandSpec,
    CorpusSample,
    Failure,
    InteractionConfig,
    ModelParams,
    StageArtifact,
    VocabEntry,
)
from .stages import StageFailure
from .util import digest, elapsed_ms, value_to_plain


@dataclass
class TaskOutcome:
    sample_id: str
    snapshot_id: str
    ledger: dict[str, str]
    artifacts: dict[str, StageArtifact]
    failure: Failure | None = None
    decoder: dict[str, Any] | None = None
    merge: dict[str, Any] | None = None


@dataclass
class PendingBatch:
    snapshot_id: str
    version: int
    futures: list[Future] = field(default_factory=list)


class LocalNerEngine:
    """Mutable command target with content-addressed stage caches."""

    def __init__(
        self,
        samples: dict[str, CorpusSample],
        model: ModelParams,
        config: InteractionConfig,
        *,
        load_failures: list[Failure] | None = None,
        command_failures: list[Failure] | None = None,
        workspace: str | None = None,
        snapshot_manager: Any = None,
    ):
        self.samples = dict(samples)
        self.model = model
        self.config = config
        self.load_failures = list(load_failures or [])
        self.command_failures = list(command_failures or [])
        self.workspace = workspace
        self.snapshot_manager = snapshot_manager
        self.version = 0
        self.last_command_position = 0
        self.processed_commands = 0
        worker_count = max(1, config.thread_pool_workers)
        self.executor = ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="ner-stage",
        )
        # Pay the thread-creation cost before the first interaction. This keeps
        # the latency-budget response path free of first-use pool startup.
        wait([self.executor.submit(lambda: None) for _ in range(worker_count)])
        self._cache: dict[tuple[str, str], StageArtifact] = {}
        self._cache_lock = threading.Lock()
        self._snapshot_failures: dict[tuple[str, str], Failure] = {}
        self._stage_ledger: dict[str, dict[str, str]] = {}
        self._pending: PendingBatch | None = None
        self._last_ready_summary: dict[str, Any] | None = None
        self._last_settle_ledger: dict[str, Any] | None = None
        self._last_affected: list[str] = []
        self._known_done_by_snapshot: dict[str, set[str]] = {}
        self._closed = False
        self.command_responses: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # World fingerprints and content-addressed cache helpers
    # ------------------------------------------------------------------
    def snapshot_id(self) -> str:
        return digest(
            {
                "corpus": corpus_fingerprint(self.samples),
                "vocabulary": vocabulary_fingerprint(self.model.vocabulary),
                "model": model_fingerprint(self.model),
                "config": config_fingerprint(self.config),
            }
        )

    def _cache_get(self, stage: str, key: str) -> StageArtifact | None:
        with self._cache_lock:
            return self._cache.get((stage, key))

    def _cache_put(self, artifact: StageArtifact) -> StageArtifact:
        with self._cache_lock:
            return self._cache.setdefault((artifact.stage, artifact.cache_key), artifact)

    def _failure_for(self, snapshot_id: str, sample_id: str) -> Failure | None:
        with self._cache_lock:
            return self._snapshot_failures.get((snapshot_id, sample_id))

    # ------------------------------------------------------------------
    # Per-sample pipeline (executed by a worker thread)
    # ------------------------------------------------------------------
    def _execute_sample(
        self,
        sample: CorpusSample,
        vocabulary: dict[str, VocabEntry],
        model: ModelParams,
        config: InteractionConfig,
        snapshot_id: str,
    ) -> TaskOutcome:
        ledger: dict[str, str] = {}
        artifacts: dict[str, StageArtifact] = {}
        try:
            fkey = feature_key(
                sample,
                vocabulary,
                model.feature_dim,
                model.hash_buckets,
            )
            feature_artifact = self._cache_get("feature", fkey)
            if feature_artifact is None:
                payload = build_features(
                    sample,
                    vocabulary,
                    model.feature_dim,
                    model.hash_buckets,
                )
                output_fp = feature_output_fingerprint(payload)
                feature_artifact = self._cache_put(
                    StageArtifact(
                        stage="feature",
                        sample_id=sample.sample_id,
                        cache_key=fkey,
                        input_fingerprint=digest(
                            {
                                "sample_id": sample.sample_id,
                                "chars": list(sample.chars),
                                "dims": [model.feature_dim, model.hash_buckets],
                            }
                        ),
                        output_fingerprint=output_fp,
                        payload=payload,
                        refs=refresh_feature_refs(payload, vocabulary),
                        version=output_fp,
                    )
                )
                ledger["feature"] = "recompute"
            else:
                ledger["feature"] = "reuse"
            artifacts["feature"] = feature_artifact
            feature_payload = feature_artifact.payload

            ekey = emission_key(
                feature_artifact.output_fingerprint,
                feature_payload,
                model,
                fixed_decimals=config.fixed_decimals,
                level=config.quantization_level,
            )
            emission_artifact = self._cache_get("emission", ekey)
            if emission_artifact is None:
                emission_payload = score_emissions(feature_payload, model, config)
                output_fp = emission_output_fingerprint(emission_payload)
                emission_artifact = self._cache_put(
                    StageArtifact(
                        stage="emission",
                        sample_id=sample.sample_id,
                        cache_key=ekey,
                        input_fingerprint=emission_parameter_fingerprint(
                            feature_payload,
                            model,
                            fixed_decimals=config.fixed_decimals,
                            level=config.quantization_level,
                        ),
                        output_fingerprint=output_fp,
                        payload=emission_payload,
                        refs=refresh_feature_refs(feature_payload, vocabulary),
                        version=output_fp,
                    )
                )
                ledger["emission"] = "recompute"
            else:
                ledger["emission"] = "reuse"
            artifacts["emission"] = emission_artifact
            emission_payload = emission_artifact.payload

            dkey = decode_key(
                emission_artifact.output_fingerprint,
                model,
                fixed_decimals=config.fixed_decimals,
                level=config.quantization_level,
                position_count=len(sample.chars),
            )
            decode_artifact = self._cache_get("decode", dkey)
            if decode_artifact is None:
                decoder_payload = decode_emissions(emission_payload, model, config)
                output_fp = decode_output_fingerprint(decoder_payload)
                decode_artifact = self._cache_put(
                    StageArtifact(
                        stage="decode",
                        sample_id=sample.sample_id,
                        cache_key=dkey,
                        input_fingerprint=transition_parameter_fingerprint(
                            model,
                            fixed_decimals=config.fixed_decimals,
                            level=config.quantization_level,
                            position_count=len(sample.chars),
                        ),
                        output_fingerprint=output_fp,
                        payload=decoder_payload,
                        refs=refresh_feature_refs(feature_payload, vocabulary),
                        version=output_fp,
                    )
                )
                ledger["decode"] = "recompute"
            else:
                ledger["decode"] = "reuse"
            artifacts["decode"] = decode_artifact
            decoder_payload = decode_artifact.payload

            mkey = merge_key(
                decode_artifact.output_fingerprint,
                sample,
                vocabulary,
                config,
                model.default_entity_type,
            )
            merge_artifact = self._cache_get("merge", mkey)
            if merge_artifact is None:
                probe_cache: dict[tuple[int, int], dict[str, Any]] = {}

                def probe_runner(decimals: int, level: int) -> dict[str, Any]:
                    probe_key = (decimals, level)
                    if probe_key not in probe_cache:
                        probe_emission = score_emissions(
                            feature_payload,
                            model,
                            config,
                            fixed_decimals=decimals,
                            quantization_level=level,
                        )
                        probe_cache[probe_key] = decode_emissions(
                            probe_emission,
                            model,
                            config,
                            fixed_decimals=decimals,
                            quantization_level=level,
                        )
                    return probe_cache[probe_key]

                merge_payload = merge_entities(
                    sample,
                    vocabulary,
                    decoder_payload,
                    config,
                    model,
                    probe_runner=probe_runner,
                )
                output_fp = merge_output_fingerprint(merge_payload)
                merge_artifact = self._cache_put(
                    StageArtifact(
                        stage="merge",
                        sample_id=sample.sample_id,
                        cache_key=mkey,
                        input_fingerprint=digest(
                            {
                                "decode": decode_artifact.output_fingerprint,
                                "sample_labels": list(sample.labels),
                                "sample_types": list(sample.types),
                                "config": config_fingerprint(config),
                            }
                        ),
                        output_fingerprint=output_fp,
                        payload=merge_payload,
                        refs=refresh_feature_refs(feature_payload, vocabulary),
                        version=output_fp,
                    )
                )
                ledger["merge"] = "recompute"
            else:
                ledger["merge"] = "reuse"
            artifacts["merge"] = merge_artifact
            return TaskOutcome(
                sample_id=sample.sample_id,
                snapshot_id=snapshot_id,
                ledger=ledger,
                artifacts=artifacts,
                decoder=decoder_payload,
                merge=merge_artifact.payload,
            )
        except StageFailure as exc:
            failure = exc.failure
            with self._cache_lock:
                self._snapshot_failures[(snapshot_id, sample.sample_id)] = failure
            return TaskOutcome(
                sample_id=sample.sample_id,
                snapshot_id=snapshot_id,
                ledger=ledger,
                artifacts=artifacts,
                failure=failure,
            )

    # ------------------------------------------------------------------
    # Stage planning / cache introspection
    # ------------------------------------------------------------------
    def _sample_stage_view(self, sample: CorpusSample) -> dict[str, Any]:
        view: dict[str, Any] = {"sample_id": sample.sample_id, "stages": {}}
        fkey = feature_key(
            sample,
            self.model.vocabulary,
            self.model.feature_dim,
            self.model.hash_buckets,
        )
        feature_artifact = self._cache_get("feature", fkey)
        view["stages"]["feature"] = {
            "key": fkey,
            "artifact": feature_artifact,
        }
        if feature_artifact is None:
            return view
        ekey = emission_key(
            feature_artifact.output_fingerprint,
            feature_artifact.payload,
            self.model,
            fixed_decimals=self.config.fixed_decimals,
            level=self.config.quantization_level,
        )
        emission_artifact = self._cache_get("emission", ekey)
        view["stages"]["emission"] = {
            "key": ekey,
            "artifact": emission_artifact,
        }
        if emission_artifact is None:
            return view
        dkey = decode_key(
            emission_artifact.output_fingerprint,
            self.model,
            fixed_decimals=self.config.fixed_decimals,
            level=self.config.quantization_level,
            position_count=len(sample.chars),
        )
        decode_artifact = self._cache_get("decode", dkey)
        view["stages"]["decode"] = {
            "key": dkey,
            "artifact": decode_artifact,
        }
        if decode_artifact is None:
            return view
        mkey = merge_key(
            decode_artifact.output_fingerprint,
            sample,
            self.model.vocabulary,
            self.config,
            self.model.default_entity_type,
        )
        merge_artifact = self._cache_get("merge", mkey)
        view["stages"]["merge"] = {
            "key": mkey,
            "artifact": merge_artifact,
        }
        return view

    def _current_done_sample_ids(self, snapshot_id: str) -> set[str]:
        done: set[str] = set()
        for sample_id in sorted(self.samples):
            sample = self.samples[sample_id]
            view = self._sample_stage_view(sample)
            merge_artifact = view["stages"].get("merge", {}).get("artifact")
            if merge_artifact is not None:
                done.add(sample_id)
            elif self._failure_for(snapshot_id, sample_id) is not None:
                done.add(sample_id)
        return done

    def _missing_sample_ids(self, snapshot_id: str) -> list[str]:
        if not self._cache and not self._snapshot_failures:
            return sorted(self.samples)
        done = self._current_done_sample_ids(snapshot_id)
        self._known_done_by_snapshot.setdefault(snapshot_id, set()).update(done)
        return [sample_id for sample_id in sorted(self.samples) if sample_id not in done]

    # ------------------------------------------------------------------
    # Threaded batch execution and latency response
    # ------------------------------------------------------------------
    def _harvest(self, batch: PendingBatch) -> list[TaskOutcome]:
        outcomes: list[TaskOutcome] = []
        remaining: list[Future] = []
        for future in batch.futures:
            if future.done():
                try:
                    outcomes.append(future.result())
                except StageFailure as exc:
                    sample_id = str(getattr(future, "_sample_id", ""))
                    with self._cache_lock:
                        self._snapshot_failures[(batch.snapshot_id, sample_id)] = exc.failure
                except Exception as exc:  # pragma: no cover - defensive boundary
                    sample_id = str(getattr(future, "_sample_id", ""))
                    failure = Failure(
                        code=E_EMISSION_INVALID,
                        stage="emission",
                        message=f"worker failed: {exc}",
                        sample_id=sample_id or None,
                    )
                    with self._cache_lock:
                        self._snapshot_failures[(batch.snapshot_id, sample_id)] = failure
            else:
                remaining.append(future)
        batch.futures = remaining
        current_snapshot = self.snapshot_id()
        for outcome in outcomes:
            if outcome.snapshot_id == current_snapshot:
                self._stage_ledger.setdefault(outcome.sample_id, {}).update(outcome.ledger)
                self._known_done_by_snapshot.setdefault(current_snapshot, set()).add(
                    outcome.sample_id
                )
        if not batch.futures:
            self._pending = None
        return outcomes

    def _schedule_current(
        self,
        budget_ms: float | None,
    ) -> tuple[bool, list[str], float, set[str]]:
        started = time.monotonic()
        snapshot_id = self.snapshot_id()

        def wait_pending() -> None:
            if self._pending is None or not self._pending.futures:
                return
            if budget_ms is None:
                wait(self._pending.futures)
            else:
                remaining_budget = max(
                    0.0,
                    (budget_ms - elapsed_ms(started)) / 1000.0,
                )
                wait(self._pending.futures, timeout=remaining_budget)
            self._harvest(self._pending)

        wait_pending()
        if self._pending is not None and self._pending.futures:
            return (
                False,
                [],
                elapsed_ms(started),
                set(self._known_done_by_snapshot.get(snapshot_id, set())),
            )

        missing = self._missing_sample_ids(snapshot_id)
        if missing:
            futures: list[Future] = []
            vocabulary = dict(self.model.vocabulary)
            model = self.model.clone()
            config = self.config.clone()
            samples = dict(self.samples)
            for sample_id in missing:
                sample = samples[sample_id]
                future = self.executor.submit(
                    self._execute_sample,
                    sample,
                    vocabulary,
                    model,
                    config,
                    snapshot_id,
                )
                try:
                    setattr(future, "_sample_id", sample_id)
                except Exception:
                    pass
                futures.append(future)
            self._pending = PendingBatch(
                snapshot_id=snapshot_id,
                version=self.version,
                futures=futures,
            )
            wait_pending()
            if self._pending is not None and self._pending.futures:
                return (
                    False,
                    missing,
                    elapsed_ms(started),
                    set(self._known_done_by_snapshot.get(snapshot_id, set())),
                )

        done_ids = self._current_done_sample_ids(snapshot_id)
        self._known_done_by_snapshot.setdefault(snapshot_id, set()).update(done_ids)
        complete = len(done_ids) == len(self.samples)
        return complete, missing, elapsed_ms(started), done_ids

    def _empty_summary(self) -> dict[str, Any]:
        return {
            "summary_version": 0,
            "global_version": 0,
            "model_version": self.model.version,
            "config_hash": config_fingerprint(self.config),
            "corpus_hash": corpus_fingerprint(self.samples),
            "vocabulary_hash": vocabulary_fingerprint(self.model.vocabulary),
            "total_samples": len(self.samples),
            "completed_samples": 0,
            "entity_count": 0,
            "type_distribution": {type_code: 0 for type_code in TYPE_CODES},
            "affected_samples": [],
            "progress": {"completed": 0, "total": len(self.samples), "ratio": 0.0},
            "sample_failures": [],
            "stage_versions": {stage: {} for stage in STAGE_ORDER},
        }

    def _build_ready_summary(self) -> dict[str, Any]:
        type_distribution = {type_code: 0 for type_code in TYPE_CODES}
        entity_count = 0
        completed = 0
        stage_versions = {stage: {} for stage in STAGE_ORDER}
        sample_failures: list[dict[str, Any]] = []
        snapshot_id = self.snapshot_id()
        for sample_id in sorted(self.samples):
            sample = self.samples[sample_id]
            view = self._sample_stage_view(sample)
            for stage in STAGE_ORDER:
                artifact = view["stages"].get(stage, {}).get("artifact")
                if artifact is not None:
                    stage_versions[stage][sample_id] = artifact.output_fingerprint
            merge_artifact = view["stages"].get("merge", {}).get("artifact")
            failure = self._failure_for(snapshot_id, sample_id)
            if merge_artifact is not None:
                completed += 1
                entities = merge_artifact.payload.get("entities", [])
                entity_count += len(entities)
                for entity in entities:
                    type_distribution[entity["type"]] = (
                        type_distribution.get(entity["type"], 0) + 1
                    )
            elif failure is not None:
                completed += 1
                sample_failures.append(failure.as_dict())
        summary = {
            "summary_version": self.version,
            "global_version": self.version,
            "model_version": self.model.version,
            "config_hash": config_fingerprint(self.config),
            "corpus_hash": corpus_fingerprint(self.samples),
            "vocabulary_hash": vocabulary_fingerprint(self.model.vocabulary),
            "total_samples": len(self.samples),
            "completed_samples": completed,
            "entity_count": entity_count,
            "type_distribution": type_distribution,
            "affected_samples": list(self._last_affected),
            "progress": {
                "completed": completed,
                "total": len(self.samples),
                "ratio": (completed / len(self.samples)) if self.samples else 1.0,
            },
            "sample_failures": sample_failures,
            "stage_versions": stage_versions,
        }
        if completed == len(self.samples):
            self._last_ready_summary = summary
        return summary

    def _progress_for(
        self,
        affected: list[str],
        snapshot_id: str,
        *,
        known_done: set[str] | None = None,
    ) -> dict[str, Any]:
        relevant = affected or sorted(self.samples)
        done = (
            known_done
            if known_done is not None
            else self._current_done_sample_ids(snapshot_id)
        )
        completed = sum(1 for sample_id in relevant if sample_id in done)
        return {
            "completed": completed,
            "total": len(relevant),
            "ratio": (completed / len(relevant)) if relevant else 1.0,
            "affected_samples": relevant,
        }

    def _response_summary(
        self,
        *,
        started: float,
        affected: list[str],
        complete: bool,
        known_done: set[str] | None = None,
    ) -> tuple[dict[str, Any], str, list[str]]:
        budget = float(self.config.latency_budget_ms)
        elapsed = elapsed_ms(started)
        snapshot_id = self.snapshot_id()
        codes: list[str] = []
        if complete:
            summary = self._build_ready_summary()
            status = "ready"
            if elapsed > budget:
                codes.append(E_LATENCY_BUDGET_EXCEEDED)
                status = "ready_over_budget"
        else:
            summary = copy.deepcopy(self._last_ready_summary or self._empty_summary())
            summary["latest_requested_version"] = self.version
            summary["stale"] = True
            summary["affected_samples"] = list(affected)
            summary["progress"] = self._progress_for(
                affected,
                snapshot_id,
                known_done=known_done,
            )
            codes.append(E_LATENCY_BUDGET_EXCEEDED)
            status = "stale"
        for failure in summary.get("sample_failures", []):
            code = failure.get("code")
            if isinstance(code, str) and code not in codes:
                codes.append(code)
        summary["codes"] = codes
        summary["latency_budget_ms"] = budget
        summary["response_elapsed_ms"] = round(elapsed, 3)
        summary["status"] = status
        return summary, status, codes

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------
    @staticmethod
    def _sample_contains_any(sample: CorpusSample, words: set[str]) -> bool:
        return any(word and word in sample.text for word in words)

    def _affected_for_vocabulary_change(
        self,
        old_vocabulary: dict[str, VocabEntry],
        new_vocabulary: dict[str, VocabEntry],
    ) -> list[str]:
        changed_words: set[str] = set()
        old_ids = set(old_vocabulary)
        new_ids = set(new_vocabulary)
        for entry_id in old_ids | new_ids:
            old = old_vocabulary.get(entry_id)
            new = new_vocabulary.get(entry_id)
            if old is None or new is None or old.text != new.text or old.type_code != new.type_code:
                if old is not None:
                    changed_words.add(old.text)
                if new is not None:
                    changed_words.add(new.text)
        if not changed_words:
            return []
        return [
            sample_id
            for sample_id in sorted(self.samples)
            if self._sample_contains_any(self.samples[sample_id], changed_words)
        ]

    def _apply_command(self, command: CommandSpec) -> tuple[bool, list[Failure], list[str]]:
        payload = command.payload
        command_type = command.command_type
        failures: list[Failure] = []
        affected: list[str] = []
        old_vocabulary = dict(self.model.vocabulary)

        def invalid(message: str, code: str = E_COMMAND_INVALID) -> None:
            failures.append(
                Failure(
                    code=code,
                    stage="command",
                    message=message,
                    command_position=command.position,
                )
            )

        if command_type == "quit":
            return True, failures, []
        if command_type == "noop":
            return True, failures, []
        if command_type == "vocab_add":
            entry = VocabEntry(
                payload["entry"]["id"],
                payload["entry"]["text"],
                payload["entry"]["type"],
            )
            if entry.entry_id in self.model.vocabulary:
                invalid("vocabulary entry id already exists")
            else:
                new_vocabulary = dict(self.model.vocabulary)
                new_vocabulary[entry.entry_id] = entry
                self.model.vocabulary = new_vocabulary
                affected = self._affected_for_vocabulary_change(
                    old_vocabulary, new_vocabulary
                )
        elif command_type == "vocab_delete":
            entry_id = payload["entry_id"]
            if entry_id not in self.model.vocabulary:
                invalid("vocabulary entry id does not exist")
            else:
                new_vocabulary = dict(self.model.vocabulary)
                del new_vocabulary[entry_id]
                self.model.vocabulary = new_vocabulary
                affected = self._affected_for_vocabulary_change(
                    old_vocabulary, new_vocabulary
                )
        elif command_type == "vocab_replace":
            entry_id = payload["entry_id"]
            if entry_id not in self.model.vocabulary:
                invalid("vocabulary entry id does not exist")
            else:
                replacement = VocabEntry(
                    payload["entry"]["id"],
                    payload["entry"]["text"],
                    payload["entry"]["type"],
                )
                if (
                    replacement.entry_id != entry_id
                    and replacement.entry_id in self.model.vocabulary
                ):
                    invalid("replacement vocabulary id already exists")
                else:
                    new_vocabulary = dict(self.model.vocabulary)
                    del new_vocabulary[entry_id]
                    new_vocabulary[replacement.entry_id] = replacement
                    self.model.vocabulary = new_vocabulary
                    affected = self._affected_for_vocabulary_change(
                        old_vocabulary, new_vocabulary
                    )
        elif command_type == "corpus_replace":
            sample = CorpusSample(
                sample_id=payload["sample"]["id"],
                chars=tuple(payload["sample"]["chars"]),
                labels=tuple(payload["sample"]["labels"]),
                types=tuple(payload["sample"]["types"]),
            )
            if sample.sample_id not in self.samples:
                invalid("corpus replace target sample does not exist")
            else:
                self.samples[sample.sample_id] = sample
                affected = [sample.sample_id]
        elif command_type == "corpus_delete":
            sample_id = payload["sample_id"]
            if sample_id not in self.samples:
                invalid("corpus delete target sample does not exist")
            else:
                del self.samples[sample_id]
                affected = [sample_id]
        elif command_type == "emission_adjust":
            try:
                target = payload.get("target") if isinstance(payload.get("target"), dict) else payload
                dimension = int(target.get("dimension", target.get("row", target.get("bucket"))))
                label = str(target.get("label", target.get("tag", target.get("to_label"))))
                raw_weight = payload.get("weight", payload.get("delta", payload.get("value")))
                if raw_weight is None:
                    raise ValueError("missing weight")
                delta = Decimal(str(raw_weight))
                if not 0 <= dimension < self.model.feature_dim:
                    invalid("emission dimension is outside feature_dim", E_MODEL_PARAM_INVALID)
                elif label not in LABELS:
                    invalid("emission label outside the BIESO enum")
                else:
                    self.model.emission_weights[dimension][LABELS.index(label)] += delta
                    affected = sorted(self.samples)
            except (TypeError, ValueError):
                invalid("emission adjustment target or weight is invalid")
        elif command_type == "transition_adjust":
            try:
                target = payload.get("target") if isinstance(payload.get("target"), dict) else payload
                from_label = target.get("from", target.get("from_label", target.get("source")))
                to_label = target.get("to", target.get("to_label", target.get("destination")))
                raw_weight = payload.get("weight", payload.get("delta", payload.get("value")))
                if raw_weight is None:
                    raise ValueError("missing weight")
                delta = Decimal(str(raw_weight))
                if str(from_label) not in LABELS or str(to_label) not in LABELS:
                    invalid("transition adjustment labels outside the BIESO enum")
                else:
                    from_index = LABELS.index(str(from_label))
                    to_index = LABELS.index(str(to_label))
                    self.model.transition_weights[from_index][to_index] += delta
                    affected = [
                        sample_id
                        for sample_id in sorted(self.samples)
                        if len(self.samples[sample_id].chars) >= 2
                    ]
            except (TypeError, ValueError):
                invalid("transition adjustment target or weight is invalid")
        elif command_type == "fixed_point_set":
            decimals = payload.get("fixed_decimals")
            if not isinstance(decimals, int) or isinstance(decimals, bool) or not 0 <= decimals <= 12:
                invalid("fixed decimals must be an integer in [0, 12]")
            else:
                new_config = self.config.clone()
                new_config.fixed_decimals = decimals
                self.config = new_config
                affected = sorted(self.samples)
        elif command_type == "quantization_set":
            level = payload.get("quantization_level")
            if not isinstance(level, int) or isinstance(level, bool) or level <= 0:
                invalid("quantization level must be a positive integer")
            else:
                new_config = self.config.clone()
                new_config.quantization_level = level
                new_config.quantization_levels = tuple(
                    sorted(set(new_config.quantization_levels) | {level})
                )
                self.config = new_config
                affected = sorted(self.samples)
        else:
            invalid("unsupported command type")

        if failures:
            return False, failures, []
        return True, failures, affected

    def apply_command(self, command: CommandSpec) -> dict[str, Any]:
        started = time.monotonic()
        self.processed_commands += 1
        self.last_command_position = max(self.last_command_position, command.position)
        applied, command_failures, affected = self._apply_command(command)
        self._last_affected = sorted(affected)
        self.command_failures.extend(command_failures)
        if applied:
            self.version += 1
            self._stage_ledger = {}
            complete, missing, scheduling_ms, known_done = self._schedule_current(
                max(0.0, float(self.config.latency_budget_ms) - elapsed_ms(started))
            )
        else:
            current_snapshot = self.snapshot_id()
            known_done = self._current_done_sample_ids(current_snapshot)
            self._known_done_by_snapshot.setdefault(current_snapshot, set()).update(
                known_done
            )
            complete = len(known_done) == len(self.samples)
        summary, status, codes = self._response_summary(
            started=started,
            affected=sorted(affected),
            complete=complete,
            known_done=known_done,
        )
        response = {
            "command_position": command.position,
            "sequence": command.sequence,
            "command_type": command.command_type,
            "accepted": applied,
            "version": self.version,
            "latency_budget_ms": float(self.config.latency_budget_ms),
            "elapsed_ms": round(elapsed_ms(started), 3),
            "status": status,
            "codes": codes,
            "affected_samples": sorted(affected),
            "summary": summary,
        }
        self.command_responses.append(response)
        return response

    # ------------------------------------------------------------------
    # Settling, diagnostics and state export
    # ------------------------------------------------------------------
    def settle(self) -> dict[str, Any]:
        started = time.monotonic()
        if self._pending is not None:
            wait(self._pending.futures)
            self._harvest(self._pending)
        snapshot_id = self.snapshot_id()
        while True:
            missing = self._missing_sample_ids(snapshot_id)
            if not missing:
                break
            self._schedule_current(None)
            snapshot_id = self.snapshot_id()
            if self._pending is not None:
                wait(self._pending.futures)
                self._harvest(self._pending)
        summary = self._build_ready_summary()
        summary["status"] = "ready"
        summary["codes"] = []
        summary["settle_elapsed_ms"] = round(elapsed_ms(started), 3)
        self._last_ready_summary = copy.deepcopy(summary)
        self._last_settle_ledger = self.build_ledger()
        return summary

    def _fresh_refs(self, sample: CorpusSample) -> list[str]:
        refs = []
        for entry in sorted(
            self.model.vocabulary.values(), key=lambda item: (item.text, item.entry_id)
        ):
            if entry.text and entry.text in sample.text:
                refs.append(entry.entry_id)
        return refs

    def build_ledger(self) -> dict[str, Any]:
        ledger: dict[str, Any] = {}
        snapshot_id = self.snapshot_id()
        for sample_id in sorted(self.samples):
            sample = self.samples[sample_id]
            view = self._sample_stage_view(sample)
            stages: list[dict[str, Any]] = []
            failure = self._failure_for(snapshot_id, sample_id)
            for stage in STAGE_ORDER:
                stage_view = view["stages"].get(stage)
                if stage_view is None:
                    break
                artifact: StageArtifact | None = stage_view.get("artifact")
                if artifact is None:
                    break
                status = self._stage_ledger.get(sample_id, {}).get(stage, "reuse")
                stages.append(
                    {
                        "stage": stage,
                        "status": status,
                        "cache_key": artifact.cache_key,
                        "input_fingerprint": artifact.input_fingerprint,
                        "output_fingerprint": artifact.output_fingerprint,
                        "version": artifact.version,
                        "sample_id": sample_id,
                        "task_sort_key": f"{sample_id}:{stage}",
                        "vocabulary_entry_ids": self._fresh_refs(sample),
                    }
                )
            if failure is not None:
                stages.append(
                    {
                        "stage": failure.stage,
                        "status": "failed",
                        "code": failure.code,
                        "sample_id": sample_id,
                        "task_sort_key": f"{sample_id}:{stage}",
                        "vocabulary_entry_ids": self._fresh_refs(sample),
                    }
                )
            ledger[sample_id] = {
                "sample_id": sample_id,
                "vocabulary_entry_ids": self._fresh_refs(sample),
                "stages": stages,
            }
        return {
            "sample_order": sorted(self.samples),
            "samples": ledger,
        }

    def sample_report(self, sample_id: str) -> dict[str, Any] | None:
        sample = self.samples.get(sample_id)
        if sample is None:
            return None
        view = self._sample_stage_view(sample)
        merge_artifact = view["stages"].get("merge", {}).get("artifact")
        failure = self._failure_for(self.snapshot_id(), sample_id)
        if merge_artifact is None:
            if failure is not None:
                return {
                    "sample_id": sample_id,
                    "chars": list(sample.chars),
                    "code": failure.code,
                    "failure": failure.as_dict(),
                    "entities": [],
                }
            return None
        merge_payload = merge_artifact.payload
        emission_artifact = view["stages"].get("emission", {}).get("artifact")
        decode_artifact = view["stages"].get("decode", {}).get("artifact")
        return {
            "sample_id": sample_id,
            "chars": merge_payload["chars"],
            "gold_labels": merge_payload["gold_labels"],
            "gold_types": merge_payload["gold_types"],
            "entities": merge_payload["entities"],
            "path_labels": merge_payload["path_labels"],
            "path_total": merge_payload["path_total"],
            "position_margins": merge_payload["position_margins"],
            "repairs": merge_payload["repairs"],
            "boundary_records": merge_payload["boundary_records"],
            "candidate_scores": (
                {
                    "label_codes": decode_artifact.payload.get("label_codes", {}),
                    "trace": decode_artifact.payload.get("trace", []),
                    "tie_records": decode_artifact.payload.get("tie_records", []),
                }
                if decode_artifact is not None
                else None
            ),
            "emission_quantization": (
                {
                    "fixed_decimals": emission_artifact.payload.get("fixed_decimals"),
                    "quantization_level": emission_artifact.payload.get(
                        "quantization_level"
                    ),
                    "positions": emission_artifact.payload.get("positions", []),
                }
                if emission_artifact is not None
                else None
            ),
            "stage_versions": {
                stage: artifact.output_fingerprint
                for stage in STAGE_ORDER
                if (artifact := view["stages"].get(stage, {}).get("artifact"))
                is not None
            },
            "stage_cache_keys": {
                stage: view["stages"].get(stage, {}).get("key")
                for stage in STAGE_ORDER
                if view["stages"].get(stage, {}).get("key") is not None
            },
        }

    def current_failures(self) -> list[Failure]:
        snapshot_id = self.snapshot_id()
        values: list[Failure] = []
        for sample_id in sorted(self.samples):
            failure = self._failure_for(snapshot_id, sample_id)
            if failure is not None:
                values.append(failure)
        return sorted(values, key=Failure.sort_key)

    def all_failures(self) -> list[Failure]:
        combined: list[Failure] = []
        seen: set[tuple[Any, ...]] = set()
        for failure in (
            list(self.load_failures)
            + list(self.command_failures)
            + self.current_failures()
        ):
            identity = failure.identity()
            if identity not in seen:
                seen.add(identity)
                combined.append(failure)
        return sorted(combined, key=Failure.sort_key)

    def business_projection(self) -> dict[str, Any]:
        """Content-only projection used by incremental/full equivalence checks."""

        samples = []
        for sample_id in sorted(self.samples):
            report = self.sample_report(sample_id)
            if report is None:
                report = {
                    "sample_id": sample_id,
                    "missing_stages": True,
                }
            else:
                report = {
                    key: value
                    for key, value in report.items()
                    if key
                    in {
                        "sample_id",
                        "chars",
                        "gold_labels",
                        "gold_types",
                        "entities",
                        "path_labels",
                        "path_total",
                        "position_margins",
                        "repairs",
                        "boundary_records",
                        "candidate_scores",
                        "emission_quantization",
                        "stage_versions",
                        "stage_cache_keys",
                    }
                }
            samples.append(report)
        stage_failure_codes = [
            {
                "sample_id": failure.sample_id,
                "stage": failure.stage,
                "code": failure.code,
            }
            for failure in self.current_failures()
        ]
        return {
            "sample_order": sorted(self.samples),
            "samples": samples,
            "stage_failure_codes": stage_failure_codes,
            "model_version": self.model.version,
            "config_hash": digest(
                {
                    "fixed_decimals": self.config.fixed_decimals,
                    "quantization_level": self.config.quantization_level,
                    "quantization_levels": list(self.config.quantization_levels),
                    "boundary_score_tolerance": self.config.boundary_score_tolerance,
                    "default_entity_type": self.config.default_entity_type,
                }
            ),
            "corpus_hash": corpus_fingerprint(self.samples),
            "vocabulary_hash": vocabulary_fingerprint(self.model.vocabulary),
        }

    def full_report(self, *, equivalence: dict[str, Any] | None = None) -> dict[str, Any]:
        summary = self.settle()
        samples = [
            report
            for sample_id in sorted(self.samples)
            if (report := self.sample_report(sample_id)) is not None
        ]
        failures = self.all_failures()
        code_counts: dict[str, int] = {}
        for failure in failures:
            code_counts[failure.code] = code_counts.get(failure.code, 0) + 1
        for response in self.command_responses:
            for code in response.get("codes", []):
                if code == E_LATENCY_BUDGET_EXCEEDED:
                    code_counts[code] = code_counts.get(code, 0) + 1
        report: dict[str, Any] = {
            "schema": "ner-editor-report-v1",
            "global_version": self.version,
            "model_version": self.model.version,
            "config_hash": config_fingerprint(self.config),
            "corpus_hash": corpus_fingerprint(self.samples),
            "vocabulary_hash": vocabulary_fingerprint(self.model.vocabulary),
            "summary": summary,
            "samples": samples,
            "failures": [failure.as_dict() for failure in failures],
            "current_stage_failures": [
                failure.as_dict() for failure in self.current_failures()
            ],
            "code_counts": dict(sorted(code_counts.items())),
            "reuse_recompute_ledger": self._last_settle_ledger or self.build_ledger(),
            "command_responses": self.command_responses,
            "initial_load_failures": [
                failure.as_dict() for failure in sorted(self.load_failures, key=Failure.sort_key)
            ],
            "command_failures": [
                failure.as_dict()
                for failure in sorted(self.command_failures, key=Failure.sort_key)
            ],
        }
        if equivalence is not None:
            report["equivalence"] = equivalence
        return value_to_plain(report)

    def restore_artifacts(
        self,
        artifacts: list[StageArtifact],
        runtime_failures: list[Failure] | None = None,
    ) -> None:
        with self._cache_lock:
            for artifact in artifacts:
                self._cache[(artifact.stage, artifact.cache_key)] = artifact
        snapshot_id = self.snapshot_id()
        if runtime_failures:
            with self._cache_lock:
                for failure in runtime_failures:
                    if failure.sample_id:
                        self._snapshot_failures[
                            (snapshot_id, failure.sample_id)
                        ] = failure

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.executor.shutdown(wait=True, cancel_futures=False)
