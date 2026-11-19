"""Child process used to verify recovery after a real forced termination."""

from __future__ import annotations

import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_ledger.ledger import Ledger  # noqa: E402
from config_ledger.spec import load_spec  # noqa: E402


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: crash_worker.py SPEC LEDGER", file=sys.stderr)
        return 2
    spec = load_spec(sys.argv[1])
    ledger = Ledger(sys.argv[2])
    ledger.verify_snapshot()
    activation = ledger.activate_parameter(spec)
    spec = spec.with_parameter_version(activation.current_version)
    ledger.begin_attempt(
        parameter_version=spec.parameter_version,
        window_id=spec.observation_window,
    )
    print("IN_PROGRESS_COMMITTED", flush=True)
    time.sleep(120)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
