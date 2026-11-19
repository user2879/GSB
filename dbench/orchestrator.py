"""Orchestration: fingerprint planning, concurrent execution and recovery."""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from .cache import plan_cache
from .config import Config
from .dag import (
    DECODE_STAGE,
    ENCODE_STAGE,
    IMAGE_ENCODE_STAGE,
    INIT_STAGE,
    Node,
    build_graph,
    denoise_step_of,
    topological_order,
)
from .errors import E_SNAPSHOT_VERSION, WorkbenchError
from .executor import compute_stage
from .hashing import array_summary, fingerprint_json
from .injection import Injection
from .stages import load_vocab, vocab_fingerprint
from .store import SnapshotStore


class ShutdownRequested(Exception):
    """Raised internally after a graceful exit signal is observed."""


class Orchestrator:
    def __init__(self, config: Config, store: SnapshotStore,
                 injections: tuple[Injection, ...] = (),
                 output_png: str | None = None,
                 jitter: bool = False):
        self.config = config
        self.store = store
        self.injections = injections
        self.output_png = output_png
        self.jitter = jitter  # test-only scheduling jitter
        self._lock = threading.Lock()
        self._arrays: dict[str, np.ndarray] = {}
        self._shutdown = False
        self._stage_events: dict[str, threading.Event] = {}
        self.report: dict[str, Any] = {
            "schema": "dbench.report.v1",
            "config_fingerprint": config.fingerprint(),
            "effective_config_fingerprint": config.effective_fingerprint(),
            "stages": [],
            "reuse": [],
            "recompute": [],
            "failures": [],
            "blocked": [],
            "injection_hits": [],
            "recovery": None,
            "shutdown": None,
        }

    def request_shutdown(self) -> None:
        self._shutdown = True

    # ------------------------------------------------------------------
    def run(self) -> dict:
        nodes = build_graph(self.config)
        # First reconcile writes interrupted by a forced termination, then
        # enforce the continuity gate across every surviving stage.
        self.report["crash_recovery"] = self.store.recover_pending_writes(
            list(nodes)
        )
        for stage in nodes:
            self.store.check_continuity(stage)

        vocab = load_vocab(self.config.vocab_file)
        vocab_fp = vocab_fingerprint(vocab)
        image_pixel_fp = self._image_pixel_fingerprint()
        plan = plan_cache(self.config, nodes, self.store, vocab_fp, image_pixel_fp)
        self.report["dependency_graph"] = self._graph_dict(nodes)
        self.report["plan"] = {
            stage: {
                "param_fingerprint": entry["param_fingerprint"],
                "content_fingerprint": entry["content_fingerprint"],
                "dependency_fingerprints": entry["dependency_fingerprints"],
                "reuse": entry["reuse"],
                "reuse_version": entry["reuse_version"],
            }
            for stage, entry in plan.items()
        }

        progress = self.store.read_progress()
        self.report["recovery"] = self._recovery_point(progress, nodes)

        # Materialize reused artifacts WITHOUT any recomputation.
        for stage in topological_order(nodes):
            entry = plan[stage]
            if entry["reuse"]:
                version = entry["reuse_version"]
                manifest = self.store.read_manifest(stage, version)
                self._arrays[stage] = self.store.load_array(stage, version)
                self._record(stage, "reuse", version, manifest, entry)

        self._execute(nodes, plan)
        self.report["reuse"] = [s["stage"] for s in self.report["stages"]
                                if s["decision"] == "reuse"]
        self.report["recompute"] = [s["stage"] for s in self.report["stages"]
                                    if s["decision"] == "recompute"]
        # Stable, sort-key ordered lists regardless of thread scheduling.
        self.report["stages"].sort(key=lambda s: nodes[s["stage"]].sort_key)
        self.report["failures"].sort(
            key=lambda f: nodes[f["stage"]].sort_key if f["stage"] in nodes else (9, 9)
        )
        self.report["injection_hits"].sort(
            key=lambda h: (nodes[h["stage"]].sort_key if h["stage"] in nodes else (9, 9),
                           h["injection_id"])
        )
        return self.report

    # ------------------------------------------------------------------
    def _execute(self, nodes: dict[str, Node], plan: dict) -> None:
        """Deadlock-free DAG execution.

        Every stage gets a Future. A scheduler running on its own thread waits
        for dependency futures and then submits the stage to a pool sized to the
        number of nodes, so a worker never blocks while holding a compute slot.
        A bounded semaphore enforces the configured maximum concurrency.
        """
        from concurrent.futures import Future, ThreadPoolExecutor

        failed: set[str] = set()
        blocked: set[str] = set()
        state_lock = threading.Lock()
        slots = threading.BoundedSemaphore(max(1, self.config.max_workers))
        futures: dict[str, Future] = {}

        def run_stage(stage: str) -> None:
            node = nodes[stage]
            with state_lock:
                upstream_bad = any(
                    dep in failed or dep in blocked for dep in node.deps
                )
            if upstream_bad:
                with state_lock:
                    blocked.add(stage)
                    self._blocked(stage, node, failed | blocked)
                return
            if plan[stage]["reuse"]:
                return
            if self.jitter:
                time.sleep(0.003 * (abs(hash(stage)) % 7))
            slots.acquire()
            try:
                entry = plan[stage]
                try:
                    upstream = self._upstream_arrays(node, plan)
                    array, extra, hits, stage_err = compute_stage(
                        stage, self.config, entry, upstream, self.injections,
                        self.output_png if stage == DECODE_STAGE else None,
                    )
                except WorkbenchError as exc:
                    with state_lock:
                        failed.add(stage)
                        self._record_failure(stage, node, exc)
                        for hit in getattr(exc, "injection_hits", []):
                            self.report["injection_hits"].append(hit)
                    return
                if stage_err is not None:
                    with state_lock:
                        failed.add(stage)
                        self._record_failure(stage, node, stage_err)
                        for hit in hits:
                            self.report["injection_hits"].append(hit)
                    return
                with state_lock:
                    if self._shutdown:
                        return
                    manifest = self._build_manifest(stage, node, entry,
                                                    array, extra)
                    committed = self.store.commit(stage, manifest, array)
                    self.store.mark_committed(
                        stage, committed["version"],
                        committed["content_fingerprint"],
                    )
                    self._arrays[stage] = array
                    for hit in hits:
                        self.report["injection_hits"].append(hit)
                    self._record(stage, "recompute", committed["version"],
                                 committed, entry)
            finally:
                slots.release()

        def scheduler() -> None:
            with ThreadPoolExecutor(max_workers=max(1, len(nodes))) as pool:
                for stage in topological_order(nodes):
                    for dep in nodes[stage].deps:
                        futures[dep].result()
                    if self._shutdown:
                        idle: Future = Future()
                        idle.set_result(None)
                        futures[stage] = idle
                        continue
                    futures[stage] = pool.submit(run_stage, stage)
                for future in futures.values():
                    future.result()

        scheduler()
        if self._shutdown:
            self.report["shutdown"] = {"graceful": True, "exit_code": 0}

    def _upstream_arrays(self, node: Node, plan: dict) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for dep in node.deps:
            if dep not in self._arrays:
                self._arrays[dep] = self.store.load_array(
                    dep, plan[dep]["reuse_version"]
                )
            out[dep] = self._arrays[dep]
        return out


    def _build_manifest(self, stage: str, node: Node, entry: dict,
                        array: np.ndarray, extra: dict) -> dict:
        return {
            "stage": stage,
            "step": node.step,
            "sort_key": list(node.sort_key),
            "param_fingerprint": entry["param_fingerprint"],
            "dependency_fingerprints": entry["dependency_fingerprints"],
            "content_fingerprint": entry["content_fingerprint"],
            "array_summary": array_summary(array),
            "extra": extra,
        }

    def _record(self, stage: str, decision: str, version: int,
                manifest: dict, entry: dict) -> None:
        self.report["stages"].append({
            "stage": stage,
            "decision": decision,
            "version": version,
            "content_fingerprint": manifest.get("content_fingerprint"),
            "array_summary": manifest.get("array_summary"),
            "dependency_fingerprints": manifest.get(
                "dependency_fingerprints", entry["dependency_fingerprints"]),
        })

    def _record_failure(self, stage: str, node: Node, exc: WorkbenchError) -> None:
        self.report["failures"].append({
            "stage": stage,
            "step": node.step,
            "code": exc.code,
            "message": exc.message,
            "context": exc.context,
        })

    def _blocked(self, stage: str, node: Node, failed: set[str]) -> None:
        blocking = [dep for dep in node.deps if dep in failed]
        self.report["blocked"].append({
            "stage": stage,
            "step": node.step,
            "reason": "upstream_failure",
            "blocking_stages": blocking,
        })

    def _image_pixel_fingerprint(self) -> str | None:
        if not self.config.initial_image:
            return None
        from PIL import Image

        from .hashing import fingerprint_array
        with Image.open(self.config.initial_image) as image:
            return fingerprint_array(np.asarray(image.convert("RGB"), dtype=np.uint8))

    def _recovery_point(self, progress: dict, nodes: dict[str, Node]) -> dict:
        committed = progress.get("committed", {})
        if not committed:
            return {"recovered": False, "stage": None, "version": None}
        # Highest committed version along the current graph is the recovery point.
        candidates = [(stage, info["version"]) for stage, info in committed.items()
                      if stage in nodes]
        if not candidates:
            return {"recovered": False, "stage": None, "version": None}
        stage, version = max(candidates, key=lambda item: nodes[item[0]].sort_key)
        return {"recovered": True, "stage": stage, "version": version,
                "all_committed": {s: info["version"] for s, info in committed.items()}}

    def _graph_dict(self, nodes: dict[str, Node]) -> dict:
        return {
            stage: {
                "step": node.step,
                "sort_key": list(node.sort_key),
                "deps": list(node.deps),
                "kind": node.extra.get("kind"),
                "replaced": node.extra.get("replaced", False),
            }
            for stage, node in nodes.items()
        }
