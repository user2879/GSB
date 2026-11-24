from __future__ import annotations

import threading
from typing import Any

from .errors import LeaseError
from .ledger import Ledger
from .util import StructuredLogger, mono_ms


class ExpiryWorker:
    """Real-time monotonic lease reclaimer.

    The worker sleeps until each persisted local-monotonic deadline.  It never
    shortens or extends a lease to manufacture output; the only timing input is
    ``time.monotonic_ns`` in ``mono_ms``.
    """

    def __init__(self, ledger: Ledger, logger: StructuredLogger) -> None:
        self.ledger = ledger
        self.logger = logger
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="lease-expiry", daemon=True)
        self.expirations: list[dict[str, Any]] = []
        self.errors: list[LeaseError] = []

    def start(self) -> None:
        self._thread.start()

    def wake(self) -> None:
        self._wake.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                deadline = self.ledger.next_expiry()
            except Exception as exc:  # pragma: no cover - defensive background boundary
                self.errors.append(
                    LeaseError(
                        "E_LEDGER",
                        f"expiry worker could not read ledger: {exc!r}",
                        field="ledger",
                        expected="readable committed state",
                    )
                )
                return
            now = mono_ms()
            if deadline is None:
                self._wake.wait(0.05)
                self._wake.clear()
                continue
            if deadline > now:
                # Wake early only when a new earlier lease is granted.
                self._wake.wait(min((deadline - now) / 1000.0, 0.25))
                self._wake.clear()
                continue
            active = self.ledger.active_leases()
            due = [lease for lease in active if int(lease["expires_mono_ms"]) <= now]
            for lease in sorted(due, key=lambda item: (item["expires_mono_ms"], item["lease_id"])):
                try:
                    closed = self.ledger.expire_lease(
                        lease["lease_id"], now=mono_ms()
                    )
                    if closed is not None:
                        self.expirations.append(closed)
                        self.logger.info(
                            "lease expired on local monotonic clock",
                            lease_id=closed["lease_id"],
                            node_id=closed["node_id"],
                            event_seq=closed["event_seq"],
                            capacity_after=closed["capacity_after"],
                        )
                except LeaseError as exc:
                    self.errors.append(exc)
                    self.logger.error(exc.code, exc.message, **exc.log_fields())
                except Exception as exc:  # pragma: no cover
                    self.errors.append(
                        LeaseError(
                            "E_LEDGER",
                            f"expiry transaction failed: {exc!r}",
                            field="ledger",
                            expected="committed expiry event",
                        )
                    )

    def wait_until_idle(self, timeout_ms: int) -> bool:
        deadline = mono_ms() + timeout_ms
        while mono_ms() < deadline:
            if not self.ledger.active_leases():
                return True
            self.wake()
            self._stop.wait(0.01)
        return not self.ledger.active_leases()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=2)
