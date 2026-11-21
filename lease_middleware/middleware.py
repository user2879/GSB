"""High-level facade wiring storage, upgrade, engine, replay and logging."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .clock import Clock
from .engine import LeaseEngine
from .errors import StructuredLogger
from .replay import Replayer
from .spec import Spec, load_spec
from .storage import Storage
from .upgrade import UpgradeManager
from .verifier import verify


class LeaseMiddleware:
    def __init__(self, spec_path: str | Path, db_path: str | Path,
                 log_path: str | Path | None = None,
                 checkpoint_path: str | Path | None = None):
        self.logger = StructuredLogger(log_path)
        try:
            self.spec: Spec = load_spec(spec_path)
        except Exception as exc:
            if hasattr(exc, "code"):
                self.logger.error(exc)
            raise
        self.storage = Storage(db_path, checkpoint_path)
        self.engine = LeaseEngine(self.spec, self.storage, Clock())

    # ------------------------------------------------------------- lifecycle
    def open_and_upgrade(self) -> dict[str, Any]:
        report = UpgradeManager(self.storage, self.spec).run()
        self.logger.info("I_UPGRADE", "lease record upgrade pass complete",
                         **report.to_dict())
        self._bootstrap_view()
        return report.to_dict()

    def _bootstrap_view(self) -> None:
        """Create an initial whole view if none exists yet."""
        if self.storage.list_subscription_view():
            return
        rows: list[tuple[str, str, None, str, int]] = []
        for cid in sorted(self.spec.clients):
            client = self.spec.clients[cid]
            cursor = self.storage.get_cursor(cid)
            for key in sorted(client.subscribed_keys):
                rows.append((cid, key, None, "ungranted", cursor))
        self.storage.replace_subscription_view(rows)

    def replay_check(self) -> dict[str, Any]:
        return Replayer(self.storage, self.spec).run()

    def verify(self) -> dict[str, Any]:
        return verify(self.storage.db_path, self.spec.source_path)

    def close(self) -> None:
        self.engine.stop_sweeper()
        self.storage.close()
