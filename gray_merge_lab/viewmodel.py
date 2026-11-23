from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .canonical import atomic_write_json, read_json
from .diagnostics import anomaly_suite, environment_report
from .engine import GatewayEngine, InProcessTransport, ModelBank
from .errors import LabError
from .ledger import Ledger
from .replay import authoritative_recompute, replay_remaining
from .snapshot import SnapshotStore, snapshot_document, verify_snapshot_document
from .specs import LoadedSpec, load_spec


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data"


@dataclass
class SessionPaths:
    backends: Path = DEFAULT_DATA / "backends.json"
    rules: Path = DEFAULT_DATA / "rules.json"
    samples: Path = DEFAULT_DATA / "samples.ndjson"
    workdir: Path = Path("artifacts/app")

    @property
    def ledger(self) -> Path:
        return self.workdir / "ledger.sqlite3"

    @property
    def snapshot(self) -> Path:
        return self.workdir / "snapshot.json"

    @property
    def checkpoint(self) -> Path:
        return self.workdir / "checkpoint.json"


@dataclass
class DiagnosticSession:
    paths: SessionPaths = field(default_factory=SessionPaths)
    scale_exp: int = 3
    auto_seed: bool = True
    spec: LoadedSpec | None = None
    ledger: Ledger | None = None
    engine: GatewayEngine | None = None
    quick_payload: dict[str, Any] = field(default_factory=dict)
    snapshot_error: dict[str, Any] | None = None
    last_error: dict[str, Any] | None = None
    last_authority: dict[str, Any] | None = None
    last_recovery: dict[str, Any] | None = None
    last_anomalies: list[dict[str, Any]] | None = None
    hot_update_count: int = 0

    def initialize(self) -> "DiagnosticSession":
        self.paths.workdir.mkdir(parents=True, exist_ok=True)
        self.spec = load_spec(self.paths.backends, self.paths.rules, self.paths.samples)
        self.ledger = Ledger(self.paths.ledger)
        bank = ModelBank(self.spec)
        transport = InProcessTransport(bank)
        self.engine = GatewayEngine(
            self.spec,
            self.ledger,
            scale_exp=self.scale_exp,
            snapshot_path=self.paths.snapshot,
            transport=transport,
        )
        if self.auto_seed and self.ledger.cursor()["completed_batches"] == 0:
            self._seed_demo_session()
        self._load_snapshot()
        return self

    def _seed_demo_session(self) -> None:
        assert self.engine is not None and self.spec is not None
        first = list(self.spec.samples[:16])
        second = list(self.spec.samples[16:])
        self.engine.process_samples(first)
        atomic_write_json(
            self.paths.checkpoint,
            snapshot_document(self.engine.manifest(), self.engine.payload()),
        )
        self.engine.process_samples(second)
        self.engine.refresh_snapshot()

    def _load_snapshot(self) -> None:
        assert self.engine is not None
        replacement = self.engine.payload()
        try:
            result = SnapshotStore(self.paths.snapshot).load_or_replace(
                self.engine.manifest(), replacement
            )
            self.quick_payload = result.payload
            self.snapshot_error = None
        except LabError as exc:
            if exc.code != "E_SNAPSHOT":
                raise
            self.snapshot_error = exc.to_dict()
            self.quick_payload = replacement

    def close(self) -> None:
        if self.ledger is not None:
            self.ledger.close()
            self.ledger = None

    def _require(self) -> tuple[GatewayEngine, Ledger, LoadedSpec]:
        if self.engine is None or self.ledger is None or self.spec is None:
            raise LabError("E_RUNTIME", "session is not initialized")
        return self.engine, self.ledger, self.spec

    def refresh_quick(self) -> dict[str, Any]:
        engine, _, _ = self._require()
        document = SnapshotStore(self.paths.snapshot).read_valid()
        current = engine.manifest()
        diffs = SnapshotStore(self.paths.snapshot).compare_manifest(current)
        if diffs:
            raise LabError(
                "E_SNAPSHOT",
                "snapshot manifest differs from current session",
                details={"differences": diffs},
            )
        self.quick_payload = document["payload"]
        return self.quick_payload

    def quick_rows(self) -> list[dict[str, Any]]:
        rows = []
        for rule_id, stats in sorted(self.quick_payload.get("rule_stats", {}).items()):
            rows.append(
                {
                    "rule_id": rule_id,
                    "batches": stats["batch_count"],
                    "stable_batches": stats["stable_batches"],
                    "canary_batches": stats["canary_batches"],
                    "stable_samples": stats["stable_samples"],
                    "canary_samples": stats["canary_samples"],
                    "raw_duration_sum_ms": stats["raw_duration_sum_ms"],
                    "avg_duration_q": stats["avg_duration_q"],
                    "avg_samples_q": stats["avg_samples_q"],
                    "sampling_error_affected": stats["sampling_error_affected"],
                    "source": "snapshot",
                }
            )
        return rows

    def preview_batches(self) -> list[dict[str, Any]]:
        return sorted(
            self.quick_payload.get("preview_batches", []),
            key=lambda item: (item.get("batch_index", 0), item.get("batch_id", "")),
        )

    def merge_conclusions(self) -> list[dict[str, Any]]:
        return sorted(
            self.quick_payload.get("conclusions", []),
            key=lambda item: (item.get("batch_id", "")),
        )

    def merge_detail(self, batch_id: str) -> dict[str, Any] | None:
        return next(
            (item for item in self.quick_payload.get("conclusions", []) if item.get("batch_id") == batch_id),
            None,
        )

    def run_remaining_samples(self) -> list[dict[str, Any]]:
        engine, ledger, spec = self._require()
        existing = {
            sample_id
            for record in ledger.batches()
            for sample_id in record.get("sample_ids", [])
        }
        remaining = [sample for sample in spec.samples if sample.sample_id not in existing]
        if not remaining:
            return []
        executions = engine.process_samples(remaining)
        self.quick_payload = engine.payload()
        return [
            {
                "batch_id": item.plan.batch_id,
                "destination": item.plan.destination,
                "samples": len(item.plan.samples),
            }
            for item in executions
        ]

    def hot_update_and_reinfer(self) -> dict[str, Any]:
        engine, ledger, spec = self._require()
        canary = next(
            (
                record
                for record in ledger.batches()
                if record["destination"] == "canary" and record["rule_id"] == "search"
            ),
            None,
        )
        if canary is None:
            raise LabError("E_RUNTIME", "no canary search batch is available for re-inference")
        backend_id = next(item.canary_backend for item in spec.rules if item.rule_id == "search")
        backend = next(item for item in engine.spec.backends if item.backend_id == backend_id)
        changes = {
            "seed": (backend.seed ^ (0x9E3779B97F4A7C15 * (self.hot_update_count + 1))) & 0xFFFFFFFFFFFFFFFF,
            "hidden_dim": backend.hidden_dim + 2,
        }
        summary = engine.hot_update_backend(backend_id, changes)
        reinfer = engine.reinfer_existing_batch(canary["batch_id"])
        self.hot_update_count += 1
        self.quick_payload = engine.payload()
        return {
            "backend_summary": summary,
            "batch_id": reinfer["batch_id"],
            "adopted_merge_seq": reinfer["adopted_merge_seq"],
            "backend_param_version": reinfer["backend_param_version"],
            "receipt_count": reinfer["receipt_count"],
            "rejected_count": reinfer["rejected_count"],
        }

    def recompute_authoritative(self) -> dict[str, Any]:
        engine, ledger, _ = self._require()
        payload = SnapshotStore(self.paths.snapshot).read_valid()["payload"]
        result = authoritative_recompute(ledger, payload)
        self.last_authority = result
        self.quick_payload = payload
        return result

    def recover_from_checkpoint(self) -> dict[str, Any]:
        engine, ledger, _ = self._require()
        if self.paths.checkpoint.exists():
            document = read_json(self.paths.checkpoint)
            verify_snapshot_document(document)
            payload = document["payload"]
            source = "checkpoint"
        else:
            payload = SnapshotStore(self.paths.snapshot).read_valid()["payload"]
            source = "snapshot"
        result = replay_remaining(ledger, payload)
        engine.refresh_snapshot()
        self.quick_payload = engine.payload()
        result["source"] = source
        self.last_recovery = result
        return result

    def inject_anomalies(self) -> list[dict[str, Any]]:
        engine, _, _ = self._require()
        results = anomaly_suite(engine)
        self.last_anomalies = results
        return results

    def environment(self) -> dict[str, Any]:
        return environment_report(self.spec)

    def ledger_meta(self) -> dict[str, Any]:
        _, ledger, _ = self._require()
        return {
            "meta": ledger.meta(),
            "cursor": ledger.cursor(),
            "batches": len(ledger.batches()),
            "receipts": len(ledger.receipt_rows()),
            "integrity": ledger.verify_integrity(),
        }

    def snapshot_meta(self) -> dict[str, Any]:
        engine, _, _ = self._require()
        try:
            document = SnapshotStore(self.paths.snapshot).read_valid()
            return {
                "ok": True,
                "manifest": document["manifest"],
                "content_digest": document["content_digest"],
                "progress": document["payload"].get("progress", {}),
                "current_manifest": engine.manifest(),
            }
        except LabError as exc:
            return {"ok": False, "error": exc.to_dict()}

    def reset(self) -> None:
        if self.ledger is not None:
            self.ledger.close()
            self.ledger = None
        for path in (self.paths.ledger, self.paths.snapshot, self.paths.checkpoint):
            if path.exists():
                path.unlink()
        for suffix in ("-wal", "-shm"):
            extra = Path(str(self.paths.ledger) + suffix)
            if extra.exists():
                extra.unlink()
        self.quick_payload = {}
        self.last_authority = None
        self.last_recovery = None
        self.last_anomalies = None
        self.snapshot_error = None
        self.hot_update_count = 0
        self.initialize()
        return self
