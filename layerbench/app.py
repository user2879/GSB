"""构建编排服务：把清单、缓存、账本、调度、报告、快照串成一次构建。"""
from __future__ import annotations

import json
import os
import shutil

from . import digest, report as report_mod, snapshot as snapshot_mod
from .cache import CacheIndex, INDEX_FILENAME
from .ledger import Ledger, LEDGER_FILENAME
from .manifest import load_manifest
from .scheduler import Scheduler, monotonic_ms, wall_ms


class Workbench:
    """围绕一个工作目录的固定路径布局。"""

    def __init__(self, workdir: str):
        self.workdir = os.path.abspath(workdir)
        self.cache_dir = os.path.join(self.workdir, "cache")
        self.snapshot_dir = os.path.join(self.workdir, "snapshots")
        self.report_dir = os.path.join(self.workdir, "reports")
        for directory in (self.workdir, self.cache_dir, self.snapshot_dir,
                          self.report_dir):
            os.makedirs(directory, exist_ok=True)
        self.ledger_path = os.path.join(self.workdir, LEDGER_FILENAME)
        self.active_manifest_path = os.path.join(
            self.workdir, "active_manifest.json"
        )

    def stage_manifest(self, manifest_path: str) -> str:
        """把外部清单复制为工作区托管的活动副本（恢复只替换该副本）。"""
        if os.path.abspath(manifest_path) != os.path.abspath(
            self.active_manifest_path
        ):
            if os.path.exists(self.active_manifest_path):
                os.chmod(self.active_manifest_path, 0o644)
            shutil.copy2(manifest_path, self.active_manifest_path)
        return self.active_manifest_path

    def build(self, manifest_path: str, build_id: str | None = None,
              crash_at: str | None = None,
              source_root: str | None = None) -> dict:
        source_root = os.path.abspath(source_root) if source_root else self.workdir
        active = self.stage_manifest(manifest_path)
        manifest = load_manifest(active, workspace=source_root)
        build_id = build_id or "build-" + digest.sha256_hex(
            manifest.version + ":" + str(wall_ms())
        )[:16]
        ledger = Ledger(self.workdir)
        cache = CacheIndex.load(
            self.cache_dir, manifest.cache_capacity, monotonic_ms, wall_ms
        )
        baseline_precursors = set(cache._by_precursor.keys())
        ledger.start_build(build_id, manifest, wall_ms())
        try:
            scheduler = Scheduler(manifest, cache, ledger, build_id,
                                  crash_at=crash_at)
            records = scheduler.run()
            cache_fp = cache.fingerprint()
            report = report_mod.build_report(
                manifest, records, scheduler, ledger, build_id, cache_fp
            )
            per_build_dir = os.path.join(self.report_dir, build_id)
            canonical_path, full_path, canonical_digest = report_mod.write_reports(
                per_build_dir, report
            )
            ledger.finish_build(build_id, wall_ms(), "finished", full_path)
            completed = ledger.completed_node_uids(build_id)
            events = ledger.get_events(build_id)
            snap_path = snapshot_mod.create_snapshot(
                self.snapshot_dir, build_id, manifest, ledger, cache,
                build_id, completed, progress_cursor=len(events),
                baseline_precursors=baseline_precursors,
            )
        finally:
            ledger.close()

        return {
            "build_id": build_id,
            "report": report,
            "canonical_path": canonical_path,
            "full_path": full_path,
            "canonical_digest": canonical_digest,
            "snapshot_path": snap_path,
            "cache_fingerprint": cache.fingerprint(),
        }


    def resume_build(self, manifest_path: str, build_id: str,
                     source_root: str | None = None) -> dict:
        """强制终止后续建：复用完整提交记录，仅重跑 unknown 及下游。"""
        source_root = os.path.abspath(source_root) if source_root else self.workdir
        active = self.stage_manifest(manifest_path)
        manifest = load_manifest(active, workspace=source_root)
        ledger = Ledger(self.workdir)
        cache = CacheIndex.load(
            self.cache_dir, manifest.cache_capacity, monotonic_ms, wall_ms
        )
        baseline_precursors = set(cache._by_precursor.keys())
        try:
            rows = ledger.get_instructions(build_id)
            if not rows:
                raise ValueError(f"账本中找不到构建 {build_id} 的提交记录")
            scheduler = Scheduler(manifest, cache, ledger, build_id)
            running_uids = {r["node_uid"]
                            for r in ledger.running_nodes(build_id)}
            rerun = scheduler.prime_from_ledger(rows)
            unknown = sorted(uid for uid in rerun if uid in running_uids)
            records = scheduler.resume()
            cache_fp = cache.fingerprint()
            report = report_mod.build_report(
                manifest, records, scheduler, ledger, build_id, cache_fp
            )
            per_build_dir = os.path.join(self.report_dir, build_id + "-resume")
            canonical_path, full_path, canonical_digest = report_mod.write_reports(
                per_build_dir, report
            )
            ledger.finish_build(build_id, wall_ms(), "finished", full_path)
            completed = ledger.completed_node_uids(build_id)
            events = ledger.get_events(build_id)
            snap_path = snapshot_mod.create_snapshot(
                self.snapshot_dir, build_id + "-resumed", manifest, ledger,
                cache, build_id, completed, progress_cursor=len(events),
                baseline_precursors=baseline_precursors,
            )
        finally:
            ledger.close()
        return {
            "build_id": build_id,
            "resumed_unknown": unknown,
            "report": report,
            "canonical_path": canonical_path,
            "full_path": full_path,
            "canonical_digest": canonical_digest,
            "snapshot_path": snap_path,
            "cache_fingerprint": cache.fingerprint(),
        }

    def current_cache_fingerprint(self) -> str:
        if not os.path.isfile(os.path.join(self.cache_dir, INDEX_FILENAME)):
            return digest.stable_digest_of_mapping(
                {"capacity": 0, "entries": []}
            )
        return CacheIndex.fingerprint_file(self.cache_dir)
