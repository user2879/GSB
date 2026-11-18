"""Non-blocking first-frame preview coordination."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

from .errors import E_PREVIEW_BUDGET_PENDING
from .util import bytes_digest


@dataclass
class PreviewResponse:
    status: str
    business_code: str | None
    preview: dict[str, Any] | None
    progress: dict[str, Any]
    active_parameter_signature: str | None
    stale: bool
    png_bytes: bytes | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "business_code": self.business_code,
            "preview": self.preview,
            "progress": self.progress,
            "active_parameter_signature": self.active_parameter_signature,
            "stale": self.stale,
        }


class PreviewBroker:
    """Publishes the lowest-tier first frame without blocking final rendering.

    A preview request waits at most the configured budget for a preview whose
    parameter signature matches the active run.  If that is not possible it
    immediately returns the most recent completed preview (which may be stale)
    together with ``E_PREVIEW_BUDGET_PENDING``.
    """

    def __init__(self, default_budget_ms: int = 50) -> None:
        self.default_budget_ms = max(0, int(default_budget_ms))
        self._condition = threading.Condition(threading.RLock())
        self._active_signature: str | None = None
        self._latest: dict[str, Any] | None = None
        self._latest_png: bytes | None = None
        self._progress: dict[str, Any] = {}
        self._completed_signature: str | None = None

    def begin_run(self, signature: str, progress: dict[str, Any] | None = None) -> None:
        with self._condition:
            self._active_signature = signature
            self._completed_signature = None
            self._progress = dict(progress or {"phase": "started", "frames_done": 0})
            self._condition.notify_all()

    def update_progress(self, progress: dict[str, Any]) -> None:
        with self._condition:
            self._progress = dict(progress)
            self._condition.notify_all()

    def publish_preview(
        self,
        signature: str,
        png_bytes: bytes,
        *,
        tier: str,
        prompt_id: str,
        seed: int,
        frame_index: int,
        relative_path: str,
        progress: dict[str, Any],
    ) -> None:
        with self._condition:
            self._latest = {
                "parameter_signature": signature,
                "tier": tier,
                "prompt_id": prompt_id,
                "seed": seed,
                "frame_index": frame_index,
                "relative_path": relative_path,
                "png_sha256": bytes_digest(png_bytes),
                "png_size": len(png_bytes),
            }
            self._latest_png = bytes(png_bytes)
            self._progress = dict(progress)
            if signature == self._active_signature:
                self._completed_signature = signature
            self._condition.notify_all()

    def complete_run(self, signature: str, progress: dict[str, Any] | None = None) -> None:
        with self._condition:
            if progress is not None:
                self._progress = dict(progress)
            if self._active_signature == signature:
                self._completed_signature = signature
                self._active_signature = None
            self._condition.notify_all()

    def request_preview(self, budget_ms: int | None = None) -> PreviewResponse:
        budget = self.default_budget_ms if budget_ms is None else max(0, int(budget_ms))
        deadline = time.monotonic() + budget / 1000.0
        with self._condition:
            while True:
                active = self._active_signature
                current_ready = (
                    (active is None and self._latest is not None)
                    or (active is not None and self._completed_signature == active)
                )
                if current_ready:
                    return self._response("ok", None, active)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return self._response("pending", E_PREVIEW_BUDGET_PENDING, active)
                self._condition.wait(timeout=remaining)

    def _response(self, status: str, code: str | None, active: str | None) -> PreviewResponse:
        preview = dict(self._latest) if self._latest is not None else None
        stale = bool(preview is not None and active is not None and preview.get("parameter_signature") != active)
        return PreviewResponse(
            status=status,
            business_code=code,
            preview=preview,
            progress=dict(self._progress),
            active_parameter_signature=active,
            stale=stale,
            png_bytes=self._latest_png if not stale else self._latest_png,
        )

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "active_parameter_signature": self._active_signature,
                "completed_parameter_signature": self._completed_signature,
                "latest": dict(self._latest) if self._latest is not None else None,
                "progress": dict(self._progress),
            }
