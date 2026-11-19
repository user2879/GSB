"""Worker-thread reconstruction orchestration around immutable snapshots."""
from __future__ import annotations

import queue
import threading

from .errors import SnapshotStaleError
from .service import ReconstructionService


class WorkerReconstruction:
    """Runs edits on a dedicated thread; the UI never blocks on rebuilds.

    Submissions are non-blocking and carry a base version.  A stale submission
    raises E_SNAPSHOT_STALE inside the worker and is reported back instead of
    corrupting the head snapshot.  Published snapshots are replaced atomically.
    """

    def __init__(self, initial_snapshot):
        self.service = ReconstructionService(initial_snapshot)
        self._jobs: "queue.Queue" = queue.Queue()
        self._results: "queue.Queue" = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="recon-worker", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._jobs.get(timeout=0.05)
            except queue.Empty:
                continue
            if job is None:
                break
            edit, base_version, verify = job
            try:
                result = self.service.apply_edit(edit, base_version=base_version, verify=verify)
                self._results.put(("ok", result))
            except SnapshotStaleError as exc:
                self._results.put(("stale", exc))
            except Exception as exc:  # surfaced to the UI thread, never silent
                self._results.put(("error", exc))

    def submit(self, edit: dict, base_version: int, verify: bool = True) -> None:
        self._jobs.put((edit, base_version, verify))

    def poll_result(self, timeout: float = 0.0):
        try:
            return self._results.get(timeout=timeout)
        except queue.Empty:
            return None

    def shutdown(self) -> None:
        self._stop.set()
        self._jobs.put(None)
        self._thread.join(timeout=2.0)
