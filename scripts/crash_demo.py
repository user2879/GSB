from __future__ import annotations

import os
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rope_audit.cli import build_fixture_config
from rope_audit.config import parse_config
from rope_audit.engine import AuditEngine
from rope_audit.samples import generate_samples
from rope_audit.snapshot import SnapshotStore


def main() -> None:
    raw_config = build_fixture_config()
    raw_config["budget"] = {"batch_point_budget": 10_000, "sample_quota_points": 10_000}
    config = parse_config(raw_config)
    samples = generate_samples(config, count=2)
    engine = AuditEngine(config, samples)
    engine.run_batch(workers=1)

    store = SnapshotStore(ROOT / "artifacts" / "snapshot-kill-demo")
    store.commit(engine, engine.last_report)
    temporary = store.prepare_transaction(engine, engine.last_report)
    print(str(temporary), flush=True)
    os.kill(os.getpid(), signal.SIGKILL)


if __name__ == "__main__":
    main()
