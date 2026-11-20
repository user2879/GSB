"""Child-process runner that simulates a forced termination mid-run.

Commits ``crash_after`` batches, then os._exit(137) inside the scheduler so
the OS process really dies; the parent demo then performs recovery + replay.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gateway.engine import run
from gateway.ledger import Ledger

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    crash_after = int(sys.argv[1])
    db_path = sys.argv[2]
    ledger = Ledger(db_path)
    run(os.path.join(HERE, "backends.json"),
        os.path.join(HERE, "rules.json"),
        os.path.join(HERE, "requests_large.ndjson"),
        ledger, global_limit=4, forward_repeat=200,
        crash_after_commits=crash_after)
    ledger.close()


if __name__ == "__main__":
    main()
