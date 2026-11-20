"""Launch the Tkinter diagnostics console on the small-scale sample run."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gateway.engine import run
from gateway.ledger import Ledger
from gateway.ui import launch

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    ledger = Ledger(os.path.join(HERE, "ui_ledger.db"))
    result = run(os.path.join(HERE, "backends.json"),
                 os.path.join(HERE, "rules.json"),
                 os.path.join(HERE, "requests_large.ndjson"),
                 ledger, global_limit=4, forward_repeat=200)
    launch(result, ledger)
    ledger.close()


if __name__ == "__main__":
    main()
