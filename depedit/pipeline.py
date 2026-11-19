"""主编排：并发解析 -> 指纹缓存 -> 双版本序列化 -> 失效判定 -> 快照提交。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cache import IncrementalCache
from .codecs import (
    load_json,
    load_text,
    v1_parse_document,
    v1_serialize_document,
    v2_parse_document,
    v2_serialize_document,
)
from .concurrency import Task, TaskOutcome, run_tasks
from .config import AppConfig, INJECTION_BUSINESS_CODES
from .fingerprints import model_fingerprint, mapping_fingerprint
from .injection import InjectionEngine, current_inject_id, reset_injection_context
from .models import Corpus, Sentence
from .snapshot import SnapshotStore


@dataclass
class PipelineResult:
    corpus: Corpus
    failures: list[dict[str, Any]] = field(default_factory=list)
    invalidation: list[dict[str, Any]] = field(default_factory=list)
    reusable: list[str] = field(default_factory=list)
    degradation: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, dict[str, str]] = field(default_factory=dict)
    cache: IncrementalCache | None = None


def _failure_record(task: Task, exc: Exception) -> dict[str, Any]:
    code = getattr(exc, "code", "E_UNKNOWN")
    inject_id = getattr(exc, "inject_id", None)
    if inject_id is None and task.payload:
        inject_id = task.payload.get("inject_id")
    if inject_id is None:
        inject_id = current_inject_id()
    return {
        "sent_id": task.sent_id,
        "stage": task.stage,
        "business_code": code,
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "inject_id": inject_id,
    }


class Pipeline:
    def __init__(
        self,
        cfg: AppConfig,
        mapping: dict[str, Any],
        injection_engine: InjectionEngine | None = None,
    ):
        self.cfg = cfg
        self.mapping = mapping
        self.injections = injection_engine or InjectionEngine([])

    # ---------- 解析阶段（并发，按句隔离故障） ----------

    def parse_v1(self, v1_path: str | Path) -> tuple[Corpus, list[dict[str, Any]]]:
        text = load_text(v1_path)
        lines_by_sent = self._group_v1_lines(text)
        tasks = [Task(sent_id=sid, stage="parse", payload={"lines": lines})
                 for sid, lines in sorted(lines_by_sent.items())]

        def worker(task: Task) -> Sentence:
            reset_injection_context()
            sid = task.sent_id
            lines = task.payload["lines"]
            lines = self.injections.corrupt_v1_lines(sid, lines, self.cfg)
            sub_text = "\n".join(lines)
            parsed = v1_parse_document(sub_text, self.cfg, self.mapping["v1_to_v2"])
            self.injections.raise_if_present_faults(
                sid,
                "parse",
                {"fixed_width_missing", "head_out_of_range",
                 "score_over_1000", "relation_not_in_map"},
            )
            return parsed[0]

        outcomes = run_tasks(tasks, worker, _failure_record, self.cfg)
        return self._collect(outcomes)

    def parse_v2(self, v2_path: str | Path) -> tuple[Corpus, list[dict[str, Any]]]:
        doc = load_json(v2_path)
        tasks = [Task(sent_id=sid, stage="parse", payload={"payload": doc[sid]})
                 for sid in sorted(doc.keys())]

        def worker(task: Task) -> Sentence:
            reset_injection_context()
            sid = task.sent_id
            payload = self.injections.corrupt_v2_doc(sid, task.payload["payload"])
            parsed = v2_parse_document(
                {sid: payload}, self.cfg, self.mapping["named_relations"]
            )
            self.injections.raise_if_present_faults(
                sid, "parse", {"v2_non_finite_score", "head_out_of_range"}
            )
            return parsed[0]

        outcomes = run_tasks(tasks, worker, _failure_record, self.cfg)
        return self._collect(outcomes)

    def _group_v1_lines(self, text: str) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        width = self.cfg.fixed_widths["sent_id"]
        for line in text.splitlines():
            if not line.strip():
                continue
            sid = line[:width].strip()
            grouped.setdefault(sid, []).append(line)
        return grouped

    def _collect(self, outcomes: list[TaskOutcome]) -> tuple[Corpus, list[dict[str, Any]]]:
        corpus = Corpus()
        failures: list[dict[str, Any]] = []
        for outcome in outcomes:  # 已按稳定键排序
            if outcome.failure is not None:
                failures.append(outcome.failure)
            elif isinstance(outcome.value, Sentence):
                corpus.sentences[outcome.sent_id] = outcome.value
        return corpus, failures

    # ---------- 指纹 + 序列化（并发，逐句） ----------

    def fingerprint_and_serialize(
        self,
        corpus: Corpus,
        model_fp: str,
        mapping_fp: str,
        version: str,
        cache: IncrementalCache | None = None,
    ) -> PipelineResult:
        if cache is None:
            cache = IncrementalCache(model_fp, mapping_fp, version)
        tasks = [Task(sent_id=sid, stage="serialize") for sid in corpus.sorted_ids()]

        def worker(task: Task) -> dict[str, Any]:
            sid = task.sent_id
            sent = corpus.sentences[sid]
            entry = cache.rebuild_sentence(sent)
            self.injections.convert_stage_guard(sid, "v1")
            v1_text, v1_loss = v1_serialize_document([sent], self.cfg, self.mapping["v2_to_v1"])
            self.injections.convert_stage_guard(sid, "v2")
            v2_doc, sidecar = v2_serialize_document([sent], self.cfg)
            v2_text = json.dumps(v2_doc, ensure_ascii=False, sort_keys=True)
            cache.record_artifact(sid, "v1", v1_text.encode("utf-8"), [entry.sentence_key])
            cache.record_artifact(sid, "v2", v2_text.encode("utf-8"), [entry.sentence_key])
            return {
                "sent_id": sid,
                "v1": v1_text,
                "v2": v2_text,
                "sidecar": sidecar,
                "losses": v1_loss,
            }

        outcomes = run_tasks(tasks, worker, _failure_record, self.cfg)
        result = PipelineResult(corpus=corpus, cache=cache)
        for outcome in outcomes:
            if outcome.failure is not None:
                result.failures.append(outcome.failure)
                continue
            value = outcome.value
            sid = value["sent_id"]
            result.artifacts[sid] = {"v1": value["v1"], "v2": value["v2"]}
            result.degradation.extend(value["losses"])
        result.invalidation = cache.invalidation_listing()
        result.reusable = cache.reusable_sentences()
        return result

    # ---------- 升级 / 降级转换 ----------

    def upgrade_v1_to_v2(self, corpus: Corpus) -> tuple[dict[str, Any], dict[str, Any]]:
        ordered = [corpus.sentences[sid] for sid in corpus.sorted_ids()]
        return v2_serialize_document(ordered, self.cfg)

    def downgrade_v2_to_v1(self, corpus: Corpus) -> tuple[str, list[dict[str, Any]]]:
        ordered = [corpus.sentences[sid] for sid in corpus.sorted_ids()]
        return v1_serialize_document(ordered, self.cfg, self.mapping["v2_to_v1"])


def fingerprints_of_inputs(model_path: str | Path, mapping_path: str | Path) -> tuple[str, str]:
    return model_fingerprint(load_json(model_path)), mapping_fingerprint(load_json(mapping_path))


def commit_snapshot(
    store: SnapshotStore,
    result: PipelineResult,
    *,
    global_version: int,
    model_fp: str,
    mapping_fp: str,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return store.commit(
        result.corpus,
        global_version=global_version,
        model_fp=model_fp,
        mapping_fp=mapping_fp,
        artifacts=result.artifacts,
        degradation=result.degradation,
        diagnostics=diagnostics,
    )