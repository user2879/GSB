import os
import signal
import threading
from concurrent.futures import ThreadPoolExecutor

from .canonical import version_digest
from .errors import BenchError
from .fill import fill_missing
from .parser import InputBundle
from .persistence import (
    clear_interrupt,
    interrupted,
    invalidate_chunk,
    mark_interrupted,
    read_chunk,
    read_manifest,
    reset_state,
    verify_state,
    write_chunk,
    write_manifest,
)
from .serialize import frame_for_storage
from .solver import solve_frame

EXIT_SIGINT = 130
EXIT_SIGTERM = 143


class InterruptController:
    def __init__(self):
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._exit_code = 0
        self._state_dir = None
        self._previous = {}

    def attach(self, state_dir):
        os.makedirs(state_dir, exist_ok=True)
        self._state_dir = state_dir
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._previous[sig] = signal.signal(sig, self._handler)
            except (ValueError, OSError):
                pass

    def detach(self):
        for sig, handler in self._previous.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass
        self._previous.clear()

    def _handler(self, signum, _frame):
        state_dir = self._state_dir
        if state_dir:
            try:
                mark_interrupted(state_dir)
            except OSError:
                pass
        with self._lock:
            if self._event.is_set():
                return
            self._event.set()
            self._exit_code = EXIT_SIGINT if signum == signal.SIGINT else EXIT_SIGTERM

    @property
    def requested(self):
        return self._event.is_set()

    @property
    def exit_code(self):
        return self._exit_code


class WeightedSemaphore:
    def __init__(self, budget):
        self._capacity = max(1, int(budget))
        self._remaining = self._capacity
        self._condition = threading.Condition()

    def acquire(self, weight):
        weight = min(weight, self._capacity)
        with self._condition:
            while self._remaining < weight:
                self._condition.wait()
            self._remaining -= weight
            return weight

    def release(self, weight):
        with self._condition:
            self._remaining += weight
            self._condition.notify_all()

    @property
    def remaining(self):
        with self._condition:
            return self._remaining


class SolveEngine:
    def __init__(self, bundle: InputBundle, state_dir, controller=None, progress=None):
        self.bundle = bundle
        self.state_dir = state_dir
        self.controller = controller or InterruptController()
        self.progress = progress
        self.version = version_digest(bundle.raw)
        self.fill_result = fill_missing(bundle)
        self.params = bundle.params
        self.frame_count = bundle.curves.frame_count
        self.chunk_size = self.params.chunk_size
        self.chunk_plans = self._plan_chunks()
        self._lock = threading.Lock()

    def _plan_chunks(self):
        plans = []
        for start in range(0, self.frame_count, self.chunk_size):
            ids = list(range(start, min(start + self.chunk_size, self.frame_count)))
            plans.append((start // self.chunk_size, start, ids))
        return plans

    def _chunk_weight(self, frame_ids):
        bone_count = len(self.bundle.skeleton.order)
        per_frame = bone_count * (16 * 4 + 3 * 8) + 4096
        return max(1, len(frame_ids) * per_frame)

    def prepare_state(self):
        os.makedirs(self.state_dir, exist_ok=True)
        manifest = read_manifest(self.state_dir)
        if manifest is None:
            clear_interrupt(self.state_dir)
            write_manifest(self.state_dir, self.version, self.frame_count, self.chunk_size, [])
            return set()
        if manifest.get("version_digest") != self.version:
            reset_state(self.state_dir)
            write_manifest(self.state_dir, self.version, self.frame_count, self.chunk_size, [])
            return set()
        completed = set(manifest.get("completed_chunks", []))
        for index, _start, ids in self.chunk_plans:
            if index not in completed:
                continue
            try:
                payload = read_chunk(self.state_dir, index)
                if payload["frame_start"] != index * self.chunk_size or payload["frame_count"] != len(ids):
                    raise BenchError("E_DIGEST", "chunk frame range mismatch", {"chunk": index})
                if payload["version_digest"] != self.version:
                    raise BenchError("E_DIGEST", "chunk version mismatch", {"chunk": index})
            except BenchError as exc:
                if exc.code == "E_DIGEST":
                    invalidate_chunk(self.state_dir, index)
                    completed.discard(index)
                else:
                    raise
        write_manifest(self.state_dir, self.version, self.frame_count, self.chunk_size, sorted(completed))
        return completed

    def run(self):
        completed = self.prepare_state()
        if self.progress:
            self.progress({"event": "resumed", "completed": sorted(completed)})
        pending = [plan for plan in self.chunk_plans if plan[0] not in completed]
        if not pending:
            clear_interrupt(self.state_dir)
            verify_state(self.state_dir, self.version, self.frame_count, self.chunk_size)
            return sorted(completed)

        budget = WeightedSemaphore(self.params.memory_budget)
        stop = threading.Event()
        first_error = []
        in_flight = threading.BoundedSemaphore(self.params.max_workers)
        executor = ThreadPoolExecutor(max_workers=self.params.max_workers)
        futures = []
        try:
            for index, start, ids in pending:
                if stop.is_set() or self.controller.requested:
                    break
                in_flight.acquire()
                if stop.is_set() or self.controller.requested:
                    in_flight.release()
                    break
                future = executor.submit(self._run_worker, index, start, ids, budget, stop, in_flight)
                futures.append((index, future))

            for index, future in futures:
                try:
                    future.result()
                except BenchError as exc:
                    stop.set()
                    first_error.append(exc)
                except Exception as exc:
                    stop.set()
                    first_error.append(BenchError("E_SOLVE", f"chunk {index} failed: {exc}", {"chunk": index}))
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

        if first_error:
            raise first_error[0]
        if self.controller.requested:
            raise BenchError(
                "E_INTERRUPTED",
                "stop requested at chunk boundary",
                {"exit_code": self.controller.exit_code},
            )
        clear_interrupt(self.state_dir)
        final_manifest = read_manifest(self.state_dir)
        verify_state(self.state_dir, self.version, self.frame_count, self.chunk_size)
        return sorted(final_manifest.get("completed_chunks", []))

    def _run_worker(self, index, start, frame_ids, budget, stop, in_flight):
        weight = self._chunk_weight(frame_ids)
        try:
            if stop.is_set():
                return
            budget.acquire(weight)
            if stop.is_set():
                return
            self._solve_chunk(index, start, frame_ids)
        finally:
            budget.release(weight)
            in_flight.release()

    def _solve_chunk(self, index, start, frame_ids):
        stored_frames = []
        for frame_index in frame_ids:
            frame_local = {
                name: self.fill_result.rotations[name][frame_index]
                for name in self.bundle.skeleton.order
            }
            result = solve_frame(
                frame_index,
                frame_local,
                self.bundle.curves.root_positions[frame_index],
                self.bundle,
            )
            stored_frames.append(frame_for_storage(result))
        write_chunk(self.state_dir, index, start, self.version, stored_frames)
        with self._lock:
            manifest = read_manifest(self.state_dir)
            completed = set(manifest.get("completed_chunks", []))
            completed.add(index)
            write_manifest(
                self.state_dir,
                self.version,
                self.frame_count,
                self.chunk_size,
                sorted(completed),
            )
        if self.progress:
            self.progress({"event": "chunk_done", "chunk": index, "frames": frame_ids})

    def load_all_frames(self):
        verify_state(self.state_dir, self.version, self.frame_count, self.chunk_size)
        frames = []
        for index, _start, ids in self.chunk_plans:
            payload = read_chunk(self.state_dir, index)
            if payload["frame_count"] != len(ids):
                raise BenchError("E_DIGEST", "chunk frame count mismatch during assembly", {"chunk": index})
            frames.extend(payload["frames"])
        frames.sort(key=lambda item: item["frame"])
        if [item["frame"] for item in frames] != list(range(self.frame_count)):
            raise BenchError("E_DIGEST", "assembled frames do not cover the full timeline")
        return frames
