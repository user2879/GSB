"""Background reconstruction service and atomic view publication."""

from __future__ import annotations

import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .engine import StudioEngine
from .errors import ErrorCode, StudioError
from .persistence import AtomicStateStore
from .snapshots import SceneSnapshot, SnapshotStore, scene_from_engine


@dataclass(frozen=True)
class ServiceResult:
    operation: dict[str, Any]
    base_snapshot_version: int
    published_snapshot_version: int
    published_scene_sha256: str
    committed_version: int


class ThreadedReconstructionService:
    """Worker-thread reconstruction; view readers only see committed snapshots."""

    def __init__(
        self,
        engine: StudioEngine,
        store: AtomicStateStore,
        snapshots: SnapshotStore,
    ) -> None:
        self.engine = engine
        self.store = store
        self.snapshots = snapshots
        self._queue: queue.Queue[tuple[dict[str, Any], dict[str, Any], Future[ServiceResult]] | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="reconstruction-worker", daemon=True)
        self._started = False
        self._closing = False
        self._events: list[dict[str, Any]] = []
        self._event_lock = threading.Lock()

    def start(self, *, commit_initial: bool = False) -> SceneSnapshot:
        if self._started:
            raise RuntimeError("service is already started")
        if commit_initial:
            self.store.commit(self.engine)
        snapshot = self._publish(self.engine)
        self._thread.start()
        self._started = True
        return snapshot

    def _publish(self, engine: StudioEngine) -> SceneSnapshot:
        snapshot = scene_from_engine(engine)
        self.snapshots.publish(snapshot)
        self._record_event(
            {
                "event": "snapshot_published",
                "version": snapshot.version,
                "content_sha256": snapshot.content_sha256,
                "point_count": snapshot.point_count,
                "mesh_sha256": snapshot.mesh_content_sha256,
                "ledger_sha256": snapshot.ledger_sha256,
            }
        )
        return snapshot

    def submit(
        self,
        operation: Mapping[str, Any],
        *,
        injection: Mapping[str, Any] | None = None,
    ) -> Future[ServiceResult]:
        if not self._started or self._closing:
            raise RuntimeError("service is not running")
        future: Future[ServiceResult] = Future()
        self._queue.put((dict(operation), dict(injection or {}), future))
        return future

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            operation, injection, future = item
            checkpoint = None
            operation_applied = False
            commit_succeeded = False
            try:
                base = self.snapshots.get()
                injected_offset = injection.get("injected_vertex_offset")
                if injected_offset is not None:
                    injected_offset = tuple(float(item) for item in injected_offset)
                checkpoint = self.engine.checkpoint()
                record = self.engine.apply_operation(
                    operation,
                    tolerance_override=injection.get("tolerance_override"),
                    injected_vertex_offset=injected_offset,
                )
                operation_applied = True
                current = self.store.commit(self.engine, operation_seq=int(record["seq"]))
                commit_succeeded = True
                snapshot = self._publish(self.engine)
                result = ServiceResult(
                    operation=record,
                    base_snapshot_version=base.version,
                    published_snapshot_version=snapshot.version,
                    published_scene_sha256=snapshot.content_sha256,
                    committed_version=int(current["version"]),
                )
                self._record_event(
                    {
                        "event": "operation_committed",
                        "seq": record["seq"],
                        "type": record["operation_type"],
                        "base_snapshot_version": base.version,
                        "published_snapshot_version": snapshot.version,
                        "scene_sha256": snapshot.content_sha256,
                    }
                )
                future.set_result(result)
            except StudioError as exc:
                if operation_applied and not commit_succeeded and checkpoint is not None:
                    self.engine.rollback(checkpoint)
                self._record_event(
                    {
                        "event": "operation_rejected",
                        "type": operation.get("type"),
                        "code": exc.code.value,
                        "context": exc.context,
                    }
                )
                future.set_exception(exc)
            except Exception as exc:
                if operation_applied and not commit_succeeded and checkpoint is not None:
                    self.engine.rollback(checkpoint)
                wrapped = StudioError(
                    ErrorCode.E_MESH_MISMATCH,
                    "unexpected reconstruction worker failure",
                    context={"exception_type": type(exc).__name__, "reason": str(exc)},
                )
                self._record_event(
                    {
                        "event": "operation_failed",
                        "type": operation.get("type"),
                        "code": wrapped.code.value,
                        "context": wrapped.context,
                    }
                )
                future.set_exception(wrapped)
            finally:
                self._queue.task_done()

    def _record_event(self, event: dict[str, Any]) -> None:
        with self._event_lock:
            self._events.append(event)

    @property
    def events(self) -> list[dict[str, Any]]:
        with self._event_lock:
            return list(self._events)

    def wait_idle(self, timeout: float | None = None) -> None:
        completed = threading.Event()

        def marker() -> None:
            self._queue.join()
            completed.set()

        thread = threading.Thread(target=marker, daemon=True)
        thread.start()
        if not completed.wait(timeout):
            raise TimeoutError("reconstruction queue did not become idle")

    def close(self, *, clean_shutdown: bool = True) -> dict[str, Any] | None:
        if not self._started:
            return None
        self._closing = True
        self._queue.put(None)
        self._thread.join(timeout=30.0)
        marker = self.store.close(self.engine) if clean_shutdown else None
        self._record_event(
            {
                "event": "service_closed",
                "clean_shutdown": bool(clean_shutdown),
                "closed_marker": marker is not None,
            }
        )
        return marker
