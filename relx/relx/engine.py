"""增量抽取引擎：阶段缓存、精确失效、预算调度与并发一致性。

引擎维护 RuntimeState 与 StageCache。每次 run 先按上轮命令计算精确失效
集合，再按句子 ID 稳定顺序规划实体/关系两波任务，交给线程池并发执行，
最终按确定顺序合并结果。任意命令序列的产物与从空白全量重建逐字段一致。
"""

from concurrent.futures import ThreadPoolExecutor

from .cache import StageCache, make_stage_key, quant_tag
from .commands import Command
from .entities import build_lexical_type_index, entity_task_points, extract_entities
from .errors import FailureRecord
from .fingerprint import content_fingerprint
from .relation import relation_task_points, extract_relations
from .scheduler import BudgetScheduler, ScheduledTask
from .state import RuntimeState, apply_command


class SentenceResult:
    def __init__(self, sentence_id):
        self.sentence_id = sentence_id
        self.entities = []
        self.suppressed = []
        self.trigger_hits = ()
        self.relations = ()
        self.misses = ()
        self.referenced_templates = set()
        self.referenced_entities = set()
        self.stage_keys = {}
        self.manifest = []
        self.fingerprint = ""

    def to_dict(self):
        return {
            "sentence_id": self.sentence_id,
            "entities": [item.to_dict() for item in self.entities],
            "suppressed": [item.to_dict() for item in self.suppressed],
            "trigger_hits": [item.to_dict() for item in self.trigger_hits],
            "relations": [item.to_dict() for item in self.relations],
            "boundary_relations": [item.to_dict() for item in self.relations if item.boundary],
            "misses": [item.to_dict() for item in self.misses],
            "referenced_templates": sorted(self.referenced_templates),
            "referenced_entities": sorted(self.referenced_entities),
            "stage_keys": dict(self.stage_keys),
            "reuse_recompute": list(self.manifest),
            "fingerprint": self.fingerprint,
        }


class RunReport:
    def __init__(self):
        self.results = {}
        self.failures = []
        self.reuse = []
        self.recompute = []
        self.invalidations = []
        self.ledger = []
        self.pending = []
        self.batches = 0
        self.ledger_consistent = True
        self.snapshot_commits = []

    def to_dict(self):
        ordered_ids = sorted(self.results.keys())
        return {
            "sentences": [self.results[sid].to_dict() for sid in ordered_ids],
            "failures": [item.to_dict() for item in self.failures],
            "reuse": list(self.reuse),
            "recompute": list(self.recompute),
            "invalidations": list(self.invalidations),
            "budget": {
                "batches": self.batches,
                "pending": self.pending,
                "ledger": [row.to_dict() for row in self.ledger],
                "ledger_consistent": self.ledger_consistent,
                "snapshot_commits": list(self.snapshot_commits),
            },
        }


class ExtractionEngine:
    def __init__(self, state: RuntimeState):
        self.state = state
        self.cache = StageCache()
        self.lexical_type_index = build_lexical_type_index(state.model)
        self.results = {}
        self._last_command = None
        self._quant_signature = quant_tag(state.config)
        self._template_signature = self._template_version_map()

    def _template_version_map(self):
        return {tpl.template_id: tpl.version for tpl in self.state.model.templates}

    def _config_param_material(self):
        from .fixedpoint import quantize
        config = self.state.config
        threshold_q = quantize(config.score_threshold,
                               config.fixedpoint_width, config.quant_level)
        return {
            "max_term_length": config.max_term_length,
            "entity_types": list(config.entity_types),
            "threshold_q": threshold_q,
            "quant": quant_tag(config),
            "feature_dim": self.state.model.feature_dim,
            "hash_buckets": self.state.model.hash_buckets,
        }

    def apply(self, command: Command):
        failure = apply_command(self.state, command)
        if failure is not None:
            return failure
        invalidated = self._invalidate_for(command)
        self._last_command = command
        return invalidated

    def _invalidate_for(self, command):
        invalidated = []
        kind = command.kind
        if kind in ("edit_tokens", "add_entity", "remove_entity"):
            sid = command.payload.get("sentence_id")
            if sid in self.results:
                invalidated.extend(self.cache.invalidate_sentence(sid))
        elif kind == "set_template_weight":
            template_id = command.payload["template_id"]
            invalidated.extend(self.cache.invalidate_template(template_id))
        elif kind in ("set_threshold",):
            for sid in list(self.results.keys()):
                for stage in ("score", "miss"):
                    row = self.cache.invalidate_stage(stage, sid)
                    if row is not None:
                        invalidated.append(row)
        elif kind == "set_quant":
            invalidated.extend(self.cache.invalidate_quant())
        return invalidated

    def run(self, command=None):
        report = RunReport()
        if command is not None:
            outcome = self.apply(command)
            if isinstance(outcome, FailureRecord):
                report.failures.append(outcome)
            else:
                report.invalidations.extend(outcome)

        current_quant = quant_tag(self.state.config)
        if current_quant != self._quant_signature:
            report.invalidations.extend(self.cache.invalidate_quant())
            self._quant_signature = current_quant

        scheduler = BudgetScheduler(self.state.config.feature_budget_per_batch)
        max_workers = self.state.config.thread_count
        batches_since_snapshot = 0
        snapshot_interval = self.state.config.snapshot_commit_batches

        while True:
            planned = self._plan_tasks()
            if not planned and not scheduler.has_pending():
                break
            runnable, pending_rows, oversize, _remaining = scheduler.plan_wave(planned)
            report.failures.extend(pending_rows)
            report.failures.extend(oversize)
            if not runnable:
                break
            self._execute_wave(runnable, max_workers, report)
            batches_since_snapshot += 1
            if batches_since_snapshot >= snapshot_interval:
                pruned = self.cache.snapshot_prune(self.state.sentence_ids_sorted())
                report.snapshot_commits.append({
                    "batch_no": scheduler.batch_no,
                    "pruned_entries": [
                        {"sentence_id": sid, "stage": stage, "key": key}
                        for sid, stage, key in pruned
                    ],
                })
                batches_since_snapshot = 0

        report.batches = scheduler.batch_no
        report.ledger = list(scheduler.ledger)
        report.pending = scheduler.pending_summary()
        report.ledger_consistent = scheduler.verify_ledger()
        report.failures = self._dedupe_pending(report.failures)
        report.failures.sort(key=FailureRecord.sort_key)
        self._build_manifest(report)
        return report

    @staticmethod
    def _dedupe_pending(failures):
        merged = {}
        others = []
        for failure in failures:
            if failure.code == "E_FEATURE_BUDGET_PENDING":
                key = (failure.sentence_id, failure.stage)
                if key not in merged:
                    failure.detail = {"pending_occurrences": 1}
                    merged[key] = failure
                else:
                    merged[key].detail["pending_occurrences"] += 1
            else:
                others.append(failure)
        return others + list(merged.values())

    def _plan_tasks(self):
        tasks = []
        for sid in self.state.sentence_ids_sorted():
            record = self.state.sentences[sid]
            entity_key = self._entity_cache_key(sid)
            if self.cache.get("entity", sid, entity_key) is None:
                tasks.append(ScheduledTask(sid, "entity",
                                           entity_task_points(record, self.state.config)))
                continue
            trigger_key, score_key, miss_key = self._relation_cache_keys(sid)
            need_relation = (
                self.cache.get("trigger", sid, trigger_key) is None
                or self.cache.get("score", sid, score_key) is None
                or self.cache.get("miss", sid, miss_key) is None
            )
            if need_relation:
                tasks.append(ScheduledTask(
                    sid, "relation",
                    relation_task_points(len(self._live_entities(sid)),
                                         self.state.model,
                                         self.state.model.feature_dim),
                ))
        return tasks

    def _live_entities(self, sid):
        entry = self.cache._stores["entity"].get(sid)
        return [] if entry is None else entry.value[0]

    def _entity_cache_key(self, sid):
        record = self.state.sentences[sid]
        material = {
            "tokens": record.tokens,
            "gold": [mention.to_dict() for mention in record.entities],
            "config": {
                "entity_types": list(self.state.config.entity_types),
                "max_term_length": self.state.config.max_term_length,
            },
            "vocabulary": list(self.state.model.vocabulary),
        }
        return make_stage_key("entity", content_fingerprint(material))

    def _relation_cache_keys(self, sid):
        entry = self.cache._stores["entity"].get(sid)
        entity_fp = entry.value[2] if entry is not None else ""
        param = self._config_param_material()
        trigger_material = self._trigger_material(sid)
        trigger_key = make_stage_key("trigger", sid, entity_fp, param, trigger_material)
        score_material = self._score_material(sid)
        score_key = make_stage_key("score", sid, entity_fp, trigger_key, param, score_material)
        miss_key = make_stage_key("miss", sid, entity_fp, trigger_key, score_key, param)
        return trigger_key, score_key, miss_key

    def _trigger_material(self, sid):
        from .fixedpoint import quantize
        record = self.state.sentences[sid]
        config = self.state.config
        material = []
        for tpl in self.state.model.templates:
            if any(term in " ".join(record.tokens) for term in tpl.trigger_terms):
                material.append({
                    "template_id": tpl.template_id,
                    "weight_q": quantize(tpl.weight, config.fixedpoint_width,
                                         config.quant_level),
                    "trigger_terms": list(tpl.trigger_terms),
                })
        return sorted(material, key=lambda item: item["template_id"])

    def _score_material(self, sid):
        entry = self.cache._stores["entity"].get(sid)
        if entry is None:
            return []
        entities = entry.value[0]
        from .fixedpoint import quantize
        config = self.state.config
        material = []
        for head_index, head in enumerate(entities):
            for tail in entities[head_index + 1:]:
                for tpl in self.state.model.templates:
                    if tpl.head_type == head.entity_type and tpl.tail_type == tail.entity_type:
                        material.append({
                            "template_id": tpl.template_id,
                            "weight_q": quantize(tpl.weight, config.fixedpoint_width,
                                                 config.quant_level),
                        })
        return sorted(material, key=lambda item: item["template_id"])

    def _execute_wave(self, runnable, max_workers, report):
        entity_tasks = [task for task in runnable if task.stage == "entity"]
        relation_tasks = [task for task in runnable if task.stage == "relation"]

        def run_entity(task):
            record = self.state.sentences[task.sentence_id]
            entities, suppressed, fingerprint = extract_entities(
                record, self.state.config, self.lexical_type_index
            )
            return task.sentence_id, entities, suppressed, fingerprint

        def run_relation(task):
            entity_entry = self.cache._stores["entity"][task.sentence_id]
            entities, _suppressed, _fp = entity_entry.value
            record = self.state.sentences[task.sentence_id]
            hits, relations, misses, referenced, trigger_fp = extract_relations(
                entities, record.tokens, self.state.model, self.state.config
            )
            return task.sentence_id, hits, relations, misses, referenced, trigger_fp

        if entity_tasks:
            workers = min(max_workers, len(entity_tasks))
            if workers == 1:
                entity_outputs = [run_entity(task) for task in entity_tasks]
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    entity_outputs = list(pool.map(run_entity, entity_tasks))
            for sid, entities, suppressed, fingerprint in entity_outputs:
                key = self._entity_cache_key(sid)
                self.cache.put("entity", sid, key,
                               (entities, suppressed, fingerprint),
                               referenced_entities=tuple(item.candidate_id for item in entities),
                               quant_tag=quant_tag(self.state.config))
                report.recompute.append((sid, "entity", key))

        relation_planned = {task.sentence_id: task for task in relation_tasks}
        if relation_planned:
            live_relations = [
                relation_planned[sid]
                for sid in self.state.sentence_ids_sorted()
                if sid in relation_planned
            ]
            workers = min(max_workers, len(live_relations))
            if workers == 1:
                relation_outputs = [run_relation(task) for task in live_relations]
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    relation_outputs = list(pool.map(run_relation, live_relations))

            trigger_key, score_key, miss_key = None, None, None
            for sid, hits, relations, misses, referenced, trigger_fp in relation_outputs:
                trigger_key, score_key, miss_key = self._relation_cache_keys(sid)
                versions = {tpl_id: self._template_version_map()[tpl_id]
                            for tpl_id in sorted(referenced)}
                self.cache.put("trigger", sid, trigger_key, (hits, trigger_fp),
                               referenced_templates=tuple(sorted(referenced)),
                               template_versions=versions,
                               quant_tag=quant_tag(self.state.config))
                self.cache.put("score", sid, score_key, relations,
                               referenced_templates=tuple(sorted(referenced)),
                               template_versions=versions,
                               quant_tag=quant_tag(self.state.config))
                self.cache.put("miss", sid, miss_key, misses,
                               referenced_templates=tuple(sorted(referenced)),
                               template_versions=versions,
                               quant_tag=quant_tag(self.state.config))
                report.recompute.append((sid, "trigger", trigger_key))
                report.recompute.append((sid, "score", score_key))
                report.recompute.append((sid, "miss", miss_key))

        self._collect_results(report)

    def _collect_results(self, report):
        for sid in self.state.sentence_ids_sorted():
            entity_entry = self.cache._stores["entity"].get(sid)
            if entity_entry is None:
                continue
            entities, suppressed, entity_fp = entity_entry.value
            trigger_entry = self.cache._stores["trigger"].get(sid)
            score_entry = self.cache._stores["score"].get(sid)
            miss_entry = self.cache._stores["miss"].get(sid)
            if not all((trigger_entry, score_entry, miss_entry)):
                continue
            hits, _trigger_fp = trigger_entry.value
            relations = score_entry.value
            misses = miss_entry.value
            result = SentenceResult(sid)
            result.entities = list(entities)
            result.suppressed = list(suppressed)
            result.trigger_hits = hits
            result.relations = relations
            result.misses = misses
            result.referenced_templates = set(trigger_entry.referenced_templates)
            result.referenced_entities = set(entity_entry.referenced_entities)
            result.stage_keys = {
                "entity": entity_entry.key,
                "trigger": trigger_entry.key,
                "score": score_entry.key,
                "miss": miss_entry.key,
            }
            result.fingerprint = content_fingerprint(result.to_dict())
            self.results[sid] = result
        report.results = self.results

    def _build_manifest(self, report):
        recomputed = {(sid, stage) for sid, stage, _key in report.recompute}
        reuse_rows = []
        for sid in sorted(self.results.keys()):
            result = self.results[sid]
            for stage in ("entity", "trigger", "score", "miss"):
                if (sid, stage) in recomputed:
                    continue
                key = result.stage_keys.get(stage)
                if key is not None:
                    reuse_rows.append((sid, stage, key))
        report.reuse = sorted(reuse_rows)
        report.recompute = sorted(report.recompute)
        for sid in sorted(self.results.keys()):
            result = self.results[sid]
            result.manifest = [
                {"stage": stage,
                 "action": "recompute" if (sid, stage) in recomputed else "reuse",
                 "key": result.stage_keys.get(stage, "")}
                for stage in ("entity", "trigger", "score", "miss")
            ]
