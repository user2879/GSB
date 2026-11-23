"""Helper process: open a ledger, begin a batch, then die without commit.

Usage: python -m tests.crash_worker <ledger_path>
Simulates a forced kill: no commit_batch, no exit report.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from workbench.ledger import Ledger  # noqa: E402


def main():
    path = sys.argv[1]
    ledger = Ledger(path, scale=3)
    ledger.begin_batch(101, 104)
    ledger.conn.commit()
    os._exit(1)


if __name__ == "__main__":
    main()
