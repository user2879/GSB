"""Helper process used by the forced-termination recovery test."""

from __future__ import annotations

import sys
from pathlib import Path

from modeldash.nodes import NodeSupervisor, SocketDispatcher
from modeldash.scheduler import Scheduler
from modeldash.spec import load_spec
from modeldash.storage import Ledger
from modeldash.util import StructuredLog


def main() -> int:
    spec_path = Path(sys.argv[1])
    db_path = Path(sys.argv[2])
    data_dir = Path(sys.argv[3])
    spec = load_spec(spec_path)
    logger = StructuredLog(data_dir / "logs" / "runner.jsonl")
    ledger = Ledger(db_path)
    supervisor = NodeSupervisor(spec, data_dir, logger)
    try:
        activation = ledger.activate_spec(spec)
        supervisor.start_all()
        Scheduler(
            spec,
            ledger,
            activation.session_id,
            SocketDispatcher(spec, logger),
            recovered_state=activation.state,
            logger=logger,
        ).run()
        return 0
    finally:
        supervisor.stop_all()
        ledger.close()


if __name__ == "__main__":
    raise SystemExit(main())
