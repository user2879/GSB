"""Forced-termination recovery from committed ledger records only.

A hard kill may interrupt a batch *inside* its SQLite transaction.  Because the
batch write is atomic, those requests survive only as ``in_flight`` rows in the
pending table; they are reported as *unknown* and must be rerun.  The engine
then re-derives every window from committed request rows and compares SLI,
remaining budget and receipt bytes against the ledger via the replay module
(``E_REPLAY`` on the first non-reproducible field).
"""

from .clock import Clock
from .config import MeshConfig
from .engine import Engine
from .ledger import Ledger


def recover(config: MeshConfig, db_path: str, clock: Clock = None, logger=None):
    ledger = Ledger(db_path)
    ledger.verify_integrity()
    engine = Engine(config, ledger, clock=clock, logger=logger)
    receipts, elapsed_ms, unknown = engine.verify_recovery()
    return {
        "engine": engine,
        "ledger": ledger,
        "receipts": receipts,
        "replay_elapsed_ms": elapsed_ms,
        "unknown_requests": unknown,
        "cursor_sequence": ledger.cursor_sequence(),
        "format_version": ledger.format_version(),
        "head_digest": ledger.head_digest(),
    }


def rerun_unknown(engine: Engine, request_factory):
    """Rebuild and rerun requests that never reached a committed batch."""
    unknown = engine.ledger.unknown_requests()
    if not unknown:
        return []
    requests = [request_factory(row) for row in unknown]
    summary = engine.run_batch(requests)
    return summary
