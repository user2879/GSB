from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ledger import Ledger


@dataclass(frozen=True)
class SessionSnapshot:
    format: str
    snapshot_format_version: int
    version: int
    content_digest: str
    state: dict[str, Any]
    replaced: bool
    replacement_reason: str | None

    def wholesale_state(self) -> dict[str, Any]:
        """Return the complete replacement state; callers must not merge it."""

        return self.state


def load_session_snapshot(ledger: Ledger, path: str | Path) -> SessionSnapshot:
    envelope, replaced, reason = ledger.open_or_replace_snapshot(path)
    return SessionSnapshot(
        format=envelope.get("format", "scale-review-snapshot"),
        snapshot_format_version=envelope["snapshot_format_version"],
        version=envelope["version"],
        content_digest=envelope["content_digest"],
        state=envelope["state"],
        replaced=replaced,
        replacement_reason=reason,
    )
