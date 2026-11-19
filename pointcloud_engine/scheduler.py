"""Deterministic bounded tile scheduler with interrupt and fault propagation."""

from __future__ import annotations

import os
import signal
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field

from .errors import EngineError
from .io_utils import atomic_write_json, ensure_dir
from .render import PreparedPoints
from .scene import Scene
from .state import INTERRUPT_NAME, TileStore, compute_input_digest, render_and_persist

INTERRUPT_EXIT_CODE = 130


class InterruptCancelled(Exception):
    """Internal control-flow signal: a worker stopped cooperatively."""


@dataclass
class RunOutcome:
    completed: list[int] = field(default_factory=list)
    reused: list[int] = field(default_factory=list)
    rendered: list[int] = field(default_factory=list)
    invalidated: list[int] = field(default_factory=list)
    rerendered: list[int] = field(default_factory=list)
    archived: list[str] = field(default_factory=list)
    recovery_invalid: list[int] = field(default_factory=list)
    interrupted: bool = False
    failure: EngineError | None = None
    failed_tile: int | None = None
    success_count_at_failure: int = 0


class InterruptFlag:
    """Process-level flag set by SIGINT/SIGTERM, observed at tile boundaries."""

    def __init__(self, state_dir: str) -> None:
        self.state_dir = state_dir
        self.event = threading.Event()
        self._previous: dict[int, object] = {}

    def _handler(self, signum, frame):  # noqa: ARG002 - signal signature
        self.event.set()

    def install(self) -> None:
        if not hasattr(signal, "SIGINT"):
            return
        self._previous[signal.SIGINT] = signal.signal(signal.SIGINT, self._handler)
        if hasattr(signal, "SIGTERM"):
            self._previous[signal.SIGTERM] = signal.signal(signal.SIGTERM, self._handler)

    def restore(self) -> None:
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)
        self._previous.clear()

    def triggered(self) -> bool:
        return self.event.is_set()

    def write_marker(self) -> str:
        ensure_dir(self.state_dir)
        path = os.path.join(self.state_dir, INTERRUPT_NAME)
        atomic_write_json(
            path,
            {
                "interrupted": True,
                "exit_code": INTERRUPT_EXIT_CODE,
                "pid": os.getpid(),
                "timestamp": time.time(),
            },
        )
        return path


def _validate_reusable(scene: Scene, prepared: PreparedPoints, store: TileStore,
                       tile_index: int, outcome: RunOutcome) -> bool:
    """Return True when an existing tile can be safely reused."""
    record = store.load_tile(tile_index)
    if record is None:
        return False
    expected = compute_input_digest(scene, prepared, tile_index)
    ok, reason = store.verify_record(record, expected)
    if not ok:
        outcome.invalidated.append(tile_index)
        outcome.recovery_invalid.append(tile_index)
        archived = store.archive_stale_files([store.path_for(tile_index)])
        outcome.archived.extend(archived)
        return False
    return True


def run_scheduler(
    scene: Scene,
    prepared: PreparedPoints,
    store: TileStore,
    interrupt: InterruptFlag | None = None,
) -> RunOutcome:
    outcome = RunOutcome()
    tile_count = scene.tile_count
    per_tile_bytes = scene.tile_bytes()

    # --- recovery: version meta + stale block archival ------------------------
    ensure_dir(scene.state_dir)
    meta_path = store.meta_path()
    stale_paths: list[str] = []
    if os.path.isfile(meta_path):
        try:
            from .io_utils import read_json

            meta = read_json(meta_path)
            if meta.get("version_id") != store.version_id:
                stale_paths.extend(store.list_tile_files())
                stale_paths.append(meta_path)
        except (OSError, ValueError):
            stale_paths.extend(store.list_tile_files())
    else:
        # No version meta but tile files exist -> unknown/old version, archive.
        stale_paths.extend(store.list_tile_files())
    if stale_paths:
        outcome.archived.extend(store.archive_stale_files(stale_paths))
    store.write_meta()

    # --- decide reusable vs missing/invalid tiles -----------------------------
    reusable: set[int] = set()
    for tile_index in range(tile_count):
        if _validate_reusable(scene, prepared, store, tile_index, outcome):
            reusable.add(tile_index)
            outcome.reused.append(tile_index)
    pending = [idx for idx in range(tile_count) if idx not in reusable]

    budget_tiles = max(1, scene.memory_budget_bytes // per_tile_bytes)
    inflight_cap = min(scene.concurrency, budget_tiles)
    inflight: dict[Future, int] = {}
    next_cursor = 0
    inflight_bytes = 0

    def admit_next() -> bool:
        nonlocal next_cursor, inflight_bytes
        if next_cursor >= len(pending):
            return False
        candidate = pending[next_cursor]
        if len(inflight) >= inflight_cap:
            return False
        if inflight_bytes + per_tile_bytes > scene.memory_budget_bytes:
            # Index-order waiting: do not skip ahead to a later tile.
            return False
        if interrupt is not None and interrupt.triggered():
            return False
        future = executor.submit(_worker, scene, prepared, store, candidate, interrupt)
        inflight[future] = candidate
        inflight_bytes += per_tile_bytes
        next_cursor += 1
        return True

    with ThreadPoolExecutor(max_workers=max(1, inflight_cap)) as executor:
        while next_cursor < len(pending) or inflight:
            if interrupt is not None and interrupt.triggered():
                # Boundary effect: do not admit new work; let in-flight finish.
                for future in inflight:
                    future.cancel()
                break

            while admit_next():
                pass

            if not inflight:
                break

            done, _ = wait(tuple(inflight.keys()), return_when="FIRST_COMPLETED")
            for future in done:
                tile_index = inflight.pop(future)
                inflight_bytes -= per_tile_bytes
                try:
                    future.result()
                except InterruptCancelled:
                    # Cooperative interrupt at a tile boundary: drop the
                    # cancelled item; the outer loop then writes the marker.
                    continue
                except EngineError as exc:
                    outcome.failure = exc
                    outcome.failed_tile = tile_index
                    outcome.success_count_at_failure = len(outcome.reused) + len(outcome.rendered)
                    # Cancel everything still queued/running and stop admitting.
                    for other in inflight:
                        other.cancel()
                    pending.clear()
                    inflight.clear()
                    inflight_bytes = 0
                    return outcome
                else:
                    outcome.rendered.append(tile_index)
                    if tile_index in outcome.recovery_invalid:
                        outcome.rerendered.append(tile_index)

    if interrupt is not None and interrupt.triggered():
        outcome.interrupted = True
        interrupt.write_marker()
        return outcome

    outcome.completed = sorted(set(outcome.reused) | set(outcome.rendered))
    return outcome


def _worker(scene: Scene, prepared: PreparedPoints, store: TileStore,
            tile_index: int, interrupt: InterruptFlag | None) -> dict:
    if interrupt is not None and interrupt.triggered():
        raise InterruptCancelled(tile_index)
    if scene.tile_delay_seconds > 0.0:
        # Cooperative sleep so an interrupt is still observed promptly.
        deadline = time.time() + scene.tile_delay_seconds
        while time.time() < deadline:
            if interrupt is not None and interrupt.triggered():
                raise InterruptCancelled(tile_index)
            time.sleep(min(0.02, max(0.0, deadline - time.time())))
    return render_and_persist(scene, prepared, store, tile_index)
