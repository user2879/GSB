"""End-to-end orchestration used by the CLI and acceptance tests."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .arbiter import LeaseArbiter
from .converge import check_convergence
from .errors import AppError
from .ledger import Ledger
from .pipeline import run_pipeline
from .processes import NodeCluster
from .spec import load_spec
from .util import now_ms
from .verify import verify_ledger


class PipelineRun:
    def __init__(
        self,
        spec_path: str | Path,
        work_dir: str | Path,
        *,
        lease_ms: int = 500,
        queue_timeout_ms: int = 15_000,
        wait_for_expiry: bool = True,
        startup_timeout_s: float = 6.0,
    ) -> None:
        self.spec_path = Path(spec_path)
        self.work_dir = Path(work_dir)
        self.lease_ms = lease_ms
        self.queue_timeout_ms = queue_timeout_ms
        self.wait_for_expiry = wait_for_expiry
        self.startup_timeout_s = startup_timeout_s
        self.ledger_path = self.work_dir / "ledger.sqlite3"
        self.nodes_root = self.work_dir / "nodes"
        self.snapshot_path = self.work_dir / "ledger.snapshot.json"

    def execute(self) -> dict[str, Any]:
        spec = load_spec(self.spec_path)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.nodes_root.mkdir(parents=True, exist_ok=True)
        ledger = Ledger.open(self.ledger_path, spec)
        session_id: str | None = None
        cluster: NodeCluster | None = None
        arbiter: LeaseArbiter | None = None
        recovered = ledger.recover(current_session=None)
        session_id = ledger.begin_session(os.getpid(), {"spec": str(self.spec_path)})
        try:
            cluster = NodeCluster(
                spec,
                self.spec_path,
                self.nodes_root,
                startup_timeout_s=self.startup_timeout_s,
            )
            cluster.start_all()
            clients = {node_id: item.client for node_id, item in cluster.nodes.items()}
            arbiter = LeaseArbiter(
                spec,
                ledger,
                clients,
                session_id,
                lease_ms=self.lease_ms,
                queue_timeout_ms=self.queue_timeout_ms,
            ).start()
            pipeline_report = run_pipeline(
                spec,
                ledger,
                arbiter,
                wait_for_expiry=self.wait_for_expiry,
            )
            arbiter.stop()
            arbiter = None
            if not pipeline_report.get("ok"):
                raise AppError(
                    "E_ARB",
                    "pipeline did not complete all lease requests",
                    {"pipeline": pipeline_report},
                )
            ledger.close_session(session_id, exit_report=True, state="completed")
            session_id = None
        except Exception as exc:
            if arbiter is not None:
                arbiter.stop()
            if session_id is not None:
                try:
                    ledger.close_session(session_id, exit_report=False, state="failed")
                except Exception:
                    pass
            raise
        finally:
            if cluster is not None:
                cluster.stop_all()
        ledger.close()

        snapshot = None
        verification = None
        convergence = None
        try:
            with Ledger(self.ledger_path, spec, read_only=True) as read_only_ledger:
                snapshot = read_only_ledger.export_snapshot(self.snapshot_path)
            verification = verify_ledger(self.spec_path, self.ledger_path)
            convergence = check_convergence(self.spec_path, self.ledger_path, self.nodes_root)
        except AppError:
            raise
        return {
            "ok": True,
            "spec": str(self.spec_path),
            "ledger": str(self.ledger_path),
            "work_dir": str(self.work_dir),
            "lease_ms": self.lease_ms,
            "recovered": recovered,
            "pipeline": pipeline_report,
            "snapshot": snapshot,
            "verification": verification,
            "convergence": convergence,
            "completed_at_ms": now_ms(),
        }
