"""End-to-end orchestration: load -> plan -> schedule -> staged snapshots."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import PipelineModel
from .engine import STAGES, Engine
from .errors import Code, IssueLog
from .ledger import Ledger
from .scheduler import Scheduler, Task
from .snapshot import Manifest, SnapshotStore
from .state import EditOutcome, StateEditor
from .synthetic import synthetic_embedding_matrix


def build_embedding_table(model: PipelineModel, seed: int, span: int = 4096) -> dict[int, list[int]]:
    """Deterministic integer embedding rows, reproducible from the fixed seed.

    The matrix is materialized with a fixed-seed NumPy generator; rows are
    converted to plain Python ints for the word-size-independent fixed-point
    kernels.
    """
    matrix = synthetic_embedding_matrix(
        model.config.vocab_size, model.config.dim, seed, span
    )
    table: dict[int, list[int]] = {}
    for token in range(model.config.vocab_size):
        table[token] = [int(value) for value in matrix[token].tolist()]
    return table


@dataclass
class RunDiagnostics:
    run_id: str
    exchange_version: str
    global_version: int
    resumed: bool
    stage_points: dict[str, int]
    invalidated_count: int
    reused_count: int
    pending: list[dict]
    code_counts: dict[str, int]
    completed_ops: list[str]
    consumed_points: int
    elapsed_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class Orchestrator:
    def __init__(self, model: PipelineModel, snapshot_root: str | Path,
                 exchange_version: str = "e2", seed: int = 20261126):
        self.model = model
        self.exchange_version = exchange_version
        self.store = SnapshotStore(snapshot_root)
        self.log = IssueLog()
        self.table = build_embedding_table(model, seed)
        self.engine = Engine(model, exchange_version, self.table)
        self.editor = StateEditor(model, exchange_version, self.log)
        self.marker = 0

    # ------------------------------------------------------------------ #
    def start(self) -> Manifest:
        config_dict = self.model.config.to_dict()
        manifest = self.store.should_resume(config_dict, self.exchange_version)
        if manifest is not None:
            self.engine.ledger = self.store.load_ledger()
            self.marker = len(self.engine.ledger.events)
            return manifest
        self.engine.ledger = Ledger()
        return self.store.new_manifest(config_dict, self.exchange_version,
                                       self.store.next_version(self.store.load_manifest()))

    def apply_edits(self, edits: list[dict]) -> EditOutcome:
        outcome = self.editor.apply(edits)
        if outcome.version_changed:
            self.exchange_version = self.editor.exchange_version
            self.engine.exchange_version = self.exchange_version
        invalidated = self.engine.reconcile(outcome.changed_tokens, outcome.changed_ops)
        outcome.applied.append(f"invalidated:{len(invalidated)}")
        return outcome

    # ------------------------------------------------------------------ #
    def run_batch(self, manifest: Manifest) -> tuple[Manifest, RunDiagnostics]:
        started = time.perf_counter()
        event_boundary = len(self.engine.ledger.events)
        plan = self.engine.plan()

        tasks: list[Task] = []
        unit_by_key: dict[str, dict] = {}
        for unit in plan.to_compute:
            op = self.engine._op_map[unit["node_id"]]
            token_count = (unit["segment"][1] - unit["segment"][0]) if unit["segment"] else len(self.model.tokens)
            cost = max(1, self.model.config.dim * max(1, token_count))
            task = Task(
                chain_id=unit["chain_id"],
                seq=unit["seq"],
                token_index=unit["segment"][0] if unit["segment"] else 0,
                cost=cost,
                payload=unit,
            )
            tasks.append(task)
            unit_by_key[task.key()] = unit

        scheduler = Scheduler(
            max_workers=self.model.config.max_workers,
            point_budget=self.model.config.point_budget,
            operator_quota=self.model.config.operator_quota,
            log=self.log,
        )

        def worker(task: Task):
            return self.engine.compute_unit(task.payload)

        def ready(task: Task) -> bool:
            unit = task.payload
            op = self.engine._op_map[unit["node_id"]]
            for ref in op.inputs:
                ref_op = self.engine._op_map[f"c{op.chain_id}op{ref}"]
                if ref_op.op_type == "segment_pool":
                    scopes = self.model.segments
                    if any(self.engine._upstream_entry(ref_op, seg) is None for seg in scopes):
                        return False
                elif self.engine._upstream_entry(ref_op) is None:
                    return False
            return True

        schedule = scheduler.run_batch(
            tasks, worker,
            consumed_already=manifest.consumed_points,
            quota_already=dict(manifest.quota_spent),
            ready=ready,
        )
        for task, _entry in schedule.results:
            manifest.quota_spent[f"c{task.chain_id}op{task.seq}"] = (
                manifest.quota_spent.get(f"c{task.chain_id}op{task.seq}", 0) + task.cost
            )
        manifest.consumed_points += schedule.consumed_points
        manifest.pending = [
            {"key": task.key(), "cost": task.cost, "node_id": task.payload["node_id"],
             "stage": task.payload["stage"], "segment": task.payload["segment"]}
            for task in schedule.pending
        ]
        manifest.completed_ops = sorted({
            entry.node_id for entry in self.engine.ledger.all_entries()
            if entry.stage != "segment_pool" or entry.entry_key not in {p["key"] for p in manifest.pending}
        })
        for stage in STAGES:
            manifest.stage_versions[stage] = manifest.stage_versions.get(stage, 0) + (
                1 if any(unit["stage"] == stage for unit in plan.to_compute) else 0
            )
        manifest.stage_points = dict(self.engine.stage_points)

        stage_payloads = self._stage_payloads()
        committed = self.store.commit(manifest, self.engine.ledger, stage_payloads)
        self.store.prune_old_versions()

        events = self.engine.ledger.events[event_boundary:]
        invalidated = sum(1 for event in events if event.kind == "invalidated")
        # plan() records one reuse event per reusable entry this batch.
        reused = len(plan.reusable)
        diagnostics = RunDiagnostics(
            run_id=f"v{manifest.global_version}",
            exchange_version=self.exchange_version,
            global_version=manifest.global_version,
            resumed=manifest.global_version > 1 or event_boundary > 0,
            stage_points=dict(self.engine.stage_points),
            invalidated_count=invalidated,
            reused_count=reused,
            pending=manifest.pending,
            code_counts=self.log.count_by_code(),
            completed_ops=list(manifest.completed_ops),
            consumed_points=manifest.consumed_points,
            elapsed_ms={"batch_ms": round((time.perf_counter() - started) * 1000, 3)},
        )
        return manifest, diagnostics

    def run_to_completion(self, manifest: Manifest | None = None) -> tuple[Manifest, RunDiagnostics, list[RunDiagnostics]]:
        """Repeat batches until the pending queue drains."""
        manifest = manifest if manifest is not None else self.start()
        history: list[RunDiagnostics] = []
        while True:
            manifest, diagnostics = self.run_batch(manifest)
            history.append(diagnostics)
            if not manifest.pending:
                break
            # Next batch gets a fresh global version + replenished budget.
            manifest = self._next_manifest(manifest)
        return manifest, history[-1], history

    def _next_manifest(self, previous: Manifest) -> Manifest:
        return Manifest(
            global_version=previous.global_version + 1,
            config_hash=previous.config_hash,
            exchange_version=previous.exchange_version,
            completed_ops=list(previous.completed_ops),
            stage_versions=dict(previous.stage_versions),
            consumed_points=0,
            quota_spent=dict(previous.quota_spent),
            pending=[],
            stage_points=dict(previous.stage_points),
        )

    def _stage_payloads(self) -> dict[str, object]:
        payloads: dict[str, object] = {stage: [] for stage in STAGES}
        for entry in self.engine.ledger.all_entries():
            payloads.setdefault(entry.stage, []).append({
                "entry_key": entry.entry_key,
                "cache_key": entry.cache_key,
                "output_fp": entry.output_fp,
                "token_indices": entry.token_indices,
                "referenced_tokens": entry.referenced_tokens,
                "referenced_ops": entry.referenced_ops,
                "output": entry.output,
            })
        return payloads

    def write_report(self, diagnostics: RunDiagnostics, path: str | Path) -> None:
        report = diagnostics.to_dict()
        report["issue_log"] = self.log.to_dict()
        report["outputs"] = self.engine.all_outputs()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
