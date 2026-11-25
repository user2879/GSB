from __future__ import annotations

import json
import shutil
from pathlib import Path

from .errors import ReferenceError, SnapshotError, SpecError
from .ledger import Ledger
from .planner import build_plan
from .receipts import ReceiptContext, normalize_receipt
from .compute import param_digest
from .spec import load_bundle, load_workers


def inject_spec_corruption(workdir: str | Path) -> Path:
    target = Path(workdir) / "injection-spec"
    target.mkdir(parents=True, exist_ok=True)
    bad = target / "workers.corrupt.json"
    bad.write_text(
        '{\n  "format": "worker-decl/v1",\n  "workers": [\n    {\n'
        '      "worker_id": "w-broken",\n      "host": "localhost",\n'
        '      "port": 49701,\n      "model_dim": 4,\n'
        '      "hidden_dim": 3,\n      "seed": 1,\n'
        '      "param_version": 1,\n      "quota": 1\n    }\n  ]\n}\n',
        encoding="utf-8",
    )
    try:
        load_workers(bad)
    except SpecError as exc:
        exc.details["injected_file"] = str(bad)
        raise
    raise AssertionError("corrupt spec was unexpectedly accepted")


def inject_dangling_reference(sample_dir: str | Path) -> None:
    bundle = load_bundle(sample_dir)
    plan = build_plan(bundle)
    digests = {worker.worker_id: param_digest(worker) for worker in bundle.workers}
    context = ReceiptContext.from_plan(
        plan,
        {worker.worker_id: worker.param_version for worker in bundle.workers},
        digests,
    )
    batch = plan.batches[-1]
    raw = {
        "format": "r3",
        "batch_id": batch.batch_id,
        "request_id": batch.request_ids[0],
        "output_digest": "a" * 64,
        "param_version": 1,
        "elapsed_ms": 3,
        "worker_id": batch.worker_id,
        "rank": 0,
        "supersedes_ref": "req_dangling_missing",
        "param_digest": digests[batch.worker_id],
    }
    try:
        normalize_receipt(raw, context, line_number=7)
    except ReferenceError as exc:
        exc.details["injected_reference"] = "req_dangling_missing"
        raise
    raise AssertionError("dangling reference was unexpectedly accepted")


def inject_snapshot_digest_mismatch(workdir: str | Path) -> Path:
    target_dir = Path(workdir) / "injection-snapshot"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    db_path = target_dir / "ledger.sqlite3"
    ledger = Ledger(db_path, reset=True)
    try:
        with ledger.conn:
            ledger.conn.execute(
                "INSERT INTO meta(key,value) VALUES('injected_tamper','original') "
                "ON CONFLICT(key) DO UPDATE SET value='tampered-without-digest-refresh'"
            )
        ledger.conn.commit()
        try:
            ledger.verify_digest()
        except SnapshotError as exc:
            exc.details["injected_db"] = str(db_path)
            raise
    finally:
        ledger.close()
    raise AssertionError("digest mismatch was unexpectedly accepted")
