"""Three adaptive rollback paths, atomic switch and crash recovery.

Paths
-----
* ``bitmap``  - only blocks differing between live volume and target state
                are written; chosen when the difference ratio is below a
                configurable threshold.
* ``full``    - every block is compared and (re)written; used above threshold.
* ``chain``   - target block state is synthesized purely by replaying the
                snapshot delta chain (multi-snapshot rollback).

All three paths produce byte-identical images.  ``rollback`` can additionally
run every path into separate staging images and cross-check them, raising
``E_PATH`` with the first mismatching block number on divergence.
"""
from __future__ import annotations

import json
import os
import uuid
from typing import Any, Optional

from .errors import DomainError, E_INPUT, E_PATH, E_SHAPE
from .manager import DEFAULT_BITMAP_THRESHOLD, DEFAULT_CHAIN_SPAN, now_iso
from .storage import JOB_COMMITTED, JOB_RUNNING, JOB_UNKNOWN
from .volume import block_digest

PATH_BITMAP = "bitmap"
PATH_FULL = "full"
PATH_CHAIN = "chain"
ALL_PATHS = (PATH_BITMAP, PATH_FULL, PATH_CHAIN)


class RollbackEngine:
    def __init__(self, manager: Any) -> None:
        self.m = manager

    # -- target state helpers ------------------------------------------
    def _snapshot_version(self, volume_name: str, snapshot_id: str) -> int:
        row = self.m.store.get_snapshot(volume_name, snapshot_id)
        if row is None:
            raise DomainError(
                E_INPUT, f"unknown snapshot {snapshot_id}", field="snapshot_id",
                snapshot_id=snapshot_id, manifest_version=self.m._manifest_version(volume_name),
            )
        return int(row["version_no"])

    def target_state(self, volume_name: str, snapshot_id: str) -> dict[int, bytes]:
        """Full block map for a snapshot via chain replay (path chain base)."""
        return self.m.reconstruct_state(volume_name, snapshot_id)

    def live_digest_map(self, volume_name: str) -> dict[int, str]:
        volume = self.m.verify_volume_shape(volume_name)
        return volume.block_digests()

    # -- path planning --------------------------------------------------
    def plan_path(
        self,
        volume_name: str,
        snapshot_id: str,
        *,
        bitmap_threshold: float = DEFAULT_BITMAP_THRESHOLD,
        chain_span: int = DEFAULT_CHAIN_SPAN,
        force_path: Optional[str] = None,
    ) -> dict[str, Any]:
        if not 0.0 <= bitmap_threshold <= 1.0:
            raise DomainError(E_INPUT, "bitmap_threshold must be within [0,1]", field="bitmap_threshold")
        if force_path is not None and force_path not in ALL_PATHS:
            raise DomainError(E_INPUT, f"unknown path {force_path!r}", field="force_path")
        rows = self.m.store.list_snapshots(volume_name)
        latest_version = max((int(r["version_no"]) for r in rows), default=0)
        target_version = self._snapshot_version(volume_name, snapshot_id)
        span = latest_version - target_version
        live = self.live_digest_map(volume_name)
        target = self.target_state(volume_name, snapshot_id)
        differing = sorted(b for b in live if live[b] != block_digest(target[b]))
        ratio = len(differing) / len(live)
        if force_path:
            chosen = force_path
            reason = f"forced path {force_path}"
        elif span >= chain_span:
            chosen = PATH_CHAIN
            reason = f"snapshot span {span} >= chain_span {chain_span}"
        elif ratio < bitmap_threshold:
            chosen = PATH_BITMAP
            reason = f"diff ratio {ratio:.4f} < threshold {bitmap_threshold}"
        else:
            chosen = PATH_FULL
            reason = f"diff ratio {ratio:.4f} >= threshold {bitmap_threshold}"
        return {
            "path": chosen,
            "reason": reason,
            "block_count": len(live),
            "differing_blocks": len(differing),
            "diff_ratio": round(ratio, 6),
            "threshold": bitmap_threshold,
            "chain_span": chain_span,
            "snapshot_span": span,
            "target_version": target_version,
            "latest_version": latest_version,
        }

    # -- staging image builders ----------------------------------------
    def _build_staging(
        self, volume_name: str, snapshot_id: str, target: dict[int, bytes], mode: str
    ) -> tuple[str, int]:
        volume = self.m.open_volume(volume_name)
        staging = os.path.join(self.m.tmp_dir, f"{uuid.uuid4().hex}.staging")
        marker = staging + ".marker"
        live_digests = volume.block_digests()
        target_digests = {block_no: block_digest(data) for block_no, data in target.items()}
        if mode == PATH_BITMAP:
            write_blocks = sorted(b for b in target if live_digests[b] != target_digests[b])
        elif mode == PATH_FULL:
            write_blocks = sorted(target)
        else:  # chain: every block the target snapshot defines is materialized
            write_blocks = sorted(target)
        total = self.m._require_volume_row(volume_name)["block_count"]
        self._write_marker(marker, {"phase": "open", "verified": 0, "total": int(total), "wrote": []})
        with open(volume.path, "rb") as src, open(staging, "wb") as dst:
            for block_no in range(int(total)):
                src.seek(block_no * volume.block_size)
                dst.write(src.read(volume.block_size))
        verified = 0
        wrote: list[int] = []
        with open(staging, "r+b") as handle:
            for block_no in write_blocks:
                handle.seek(block_no * volume.block_size)
                handle.write(target[block_no])
                handle.flush()
                os.fsync(handle.fileno())
                wrote.append(block_no)
                verified += 1
                self._write_marker(
                    marker, {"phase": "writing", "verified": verified, "total": int(total), "wrote": wrote}
                )
        # verify every block of the staging image against target digests
        with open(staging, "rb") as handle:
            for block_no in range(int(total)):
                data = handle.read(volume.block_size)
                if block_digest(data) != target_digests[block_no]:
                    raise DomainError(
                        E_PATH,
                        f"staging verification mismatch at block {block_no} on path {mode}",
                        snapshot_id=snapshot_id,
                        block_no=block_no,
                        manifest_version=self.m._manifest_version(volume_name),
                    )
        self._write_marker(
            marker,
            {"phase": "verified", "verified": int(total), "total": int(total), "wrote": wrote},
        )
        return staging, marker

    @staticmethod
    def _write_marker(path: str, payload: dict[str, Any]) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)

    # -- cross path verification ---------------------------------------
    def cross_check_paths(self, volume_name: str, snapshot_id: str) -> dict[str, Any]:
        """Run all three paths and require byte-identical staging images."""
        target = self.target_state(volume_name, snapshot_id)
        images: dict[str, str] = {}
        digests: dict[str, str] = {}
        try:
            for mode in ALL_PATHS:
                staging, marker = self._build_staging(
                    volume_name, snapshot_id, target, mode
                )
                images[mode] = staging
                import hashlib
                hasher = hashlib.blake2b(digest_size=32)
                with open(staging, "rb") as handle:
                    hasher.update(handle.read())
                digests[mode] = hasher.hexdigest()
            reference = digests[PATH_BITMAP]
            for mode in (PATH_FULL, PATH_CHAIN):
                if digests[mode] != reference:
                    first = self._first_divergent_block(images[PATH_BITMAP], images[mode], volume_name)
                    err = DomainError(
                        E_PATH,
                        f"path {mode} diverges from bitmap path",
                        snapshot_id=snapshot_id,
                        block_no=first,
                        manifest_version=self.m._manifest_version(volume_name),
                        extra={"digests": digests},
                    )
                    self.m.logger.log_error(err, event="path_divergence")
                    raise err
            return {"digests": digests, "identical": True}
        finally:
            for staging in images.values():
                if os.path.exists(staging):
                    os.remove(staging)
                marker = staging + ".marker"
                if os.path.exists(marker):
                    os.remove(marker)

    def _first_divergent_block(self, left: str, right: str, volume_name: str) -> int:
        volume = self.m.open_volume(volume_name)
        with open(left, "rb") as a, open(right, "rb") as b:
            block_no = 0
            while True:
                da, db = a.read(volume.block_size), b.read(volume.block_size)
                if da != db:
                    return block_no
                if not da:
                    break
                block_no += 1
        return -1

    # -- public rollback with atomic switch ----------------------------
    def rollback(
        self,
        volume_name: str,
        snapshot_id: str,
        *,
        bitmap_threshold: float = DEFAULT_BITMAP_THRESHOLD,
        chain_span: int = DEFAULT_CHAIN_SPAN,
        force_path: Optional[str] = None,
        verify_all_paths: bool = False,
        crash_after_blocks: Optional[int] = None,
    ) -> dict[str, Any]:
        volume = self.m.verify_volume_shape(volume_name)
        plan = self.plan_path(
            volume_name, snapshot_id,
            bitmap_threshold=bitmap_threshold, chain_span=chain_span, force_path=force_path,
        )
        self.m.validate_chain(volume_name)
        # integrity check across every snapshot that contributes state
        target_version = plan["target_version"]
        rows = self.m.store.list_snapshots(volume_name)
        for row in rows[:target_version]:
            self.m.verify_block_integrity(volume_name, row["snapshot_id"])
        if verify_all_paths:
            self.cross_check_paths(volume_name, snapshot_id)

        target = self.target_state(volume_name, snapshot_id)
        job_id = f"job-{uuid.uuid4().hex[:12]}"
        staging = os.path.join(self.m.tmp_dir, f"{job_id}.staging")
        marker = staging + ".marker"
        manifest_version = self.m._manifest_version(volume_name)
        job = {
            "job_id": job_id, "volume_name": volume_name, "target_snapshot": snapshot_id,
            "manifest_version": int(manifest_version) if manifest_version is not None else 0,
            "state": JOB_RUNNING, "staging_path": staging, "path_used": plan["path"],
            "verified_count": 0, "total_count": volume.block_count, "reason": plan["reason"],
            "created_at": now_iso(), "updated_at": now_iso(),
        }
        self.m.store.upsert_job(job)
        self.m.store.commit()
        self.m.logger.log("INFO", "rollback_started", snapshot_id=snapshot_id,
                          manifest_version=manifest_version, **plan)

        try:
            self._materialize(
                volume, target, staging, marker, snapshot_id, manifest_version,
                plan["path"], job, crash_after_blocks
            )
        except DomainError:
            raise

        # atomic switch: staging -> volume image
        os.replace(staging, volume.path)
        if os.path.exists(marker):
            os.remove(marker)
        job["state"] = JOB_COMMITTED
        job["verified_count"] = volume.block_count
        job["staging_path"] = None
        job["updated_at"] = now_iso()
        self.m.store.upsert_job(job)
        # live state now equals the target snapshot state
        self.m.store.replace_current_state(
            volume_name, {b: block_digest(data) for b, data in target.items()}
        )
        self.m.store.commit()
        self.m.logger.log("INFO", "rollback_committed", snapshot_id=snapshot_id,
                          manifest_version=manifest_version, **{"job_id": job_id, "path": plan["path"]})
        return {
            "job_id": job_id,
            "state": JOB_COMMITTED,
            "path": plan["path"],
            "plan": plan,
            "volume_digest": volume.whole_digest(),
        }

    def _materialize(
        self, volume: Any, target: dict[int, bytes], staging: str, marker: str,
        snapshot_id: str, manifest_version: Optional[int], mode: str,
        job: dict[str, Any], crash_after_blocks: Optional[int],
    ) -> None:
        live = volume.block_digests()
        if mode == PATH_BITMAP:
            write_blocks = [b for b in sorted(target) if live[b] != block_digest(target[b])]
        else:
            write_blocks = sorted(target)
        self._write_marker(marker, {"phase": "open", "verified": 0, "total": volume.block_count, "wrote": []})
        with open(volume.path, "rb") as src, open(staging, "wb") as dst:
            for block_no in range(volume.block_count):
                src.seek(block_no * volume.block_size)
                dst.write(src.read(volume.block_size))
            dst.flush()
            os.fsync(dst.fileno())

        verified = 0
        wrote: list[int] = []
        with open(staging, "r+b") as handle:
            for block_no in write_blocks:
                handle.seek(block_no * volume.block_size)
                handle.write(target[block_no])
                handle.flush()
                os.fsync(handle.fileno())
                wrote.append(block_no)
                verified += 1
                self._write_marker(
                    marker,
                    {"phase": "writing", "verified": verified, "total": volume.block_count, "wrote": wrote},
                )
                job["verified_count"] = verified
                job["updated_at"] = now_iso()
                self.m.store.upsert_job(job)
                self.m.store.commit()
                if crash_after_blocks is not None and verified >= crash_after_blocks:
                    self.m.logger.log(
                        "WARN", "rollback_crash_injected", snapshot_id=snapshot_id,
                        manifest_version=manifest_version, **{"job_id": job["job_id"], "verified": verified},
                    )
                    raise SystemExit(
                        f"simulated forced termination after {verified} block(s); no exit report promised"
                    )

        with open(staging, "rb") as handle:
            for block_no in range(volume.block_count):
                data = handle.read(volume.block_size)
                if len(data) != volume.block_size:
                    raise DomainError(E_SHAPE, "staging image has wrong shape", field="staging",
                                      block_no=block_no, manifest_version=manifest_version)
                if block_digest(data) != block_digest(target[block_no]):
                    err = DomainError(
                        E_PATH, "staging block verification failed", snapshot_id=snapshot_id,
                        block_no=block_no, manifest_version=manifest_version,
                    )
                    self.m.logger.log_error(err, event="rollback_verify_failed")
                    raise err
        self._write_marker(
            marker,
            {"phase": "verified", "verified": volume.block_count, "total": volume.block_count, "wrote": wrote},
        )

    # -- crash recovery -------------------------------------------------
    def recover(self, volume_name: str, job_id: Optional[str] = None) -> dict[str, Any]:
        """Inspect persisted jobs + staging markers after a forced termination."""
        jobs = self.m.store.list_jobs(volume_name)
        if job_id is not None:
            jobs = [row for row in jobs if row["job_id"] == job_id]
        reports = []
        for row in jobs:
            if row["state"] != JOB_RUNNING:
                reports.append({"job_id": row["job_id"], "state": row["state"], "action": "none"})
                continue
            staging = row["staging_path"]
            marker = staging + ".marker" if staging else None
            if staging and os.path.exists(staging) and marker and os.path.exists(marker):
                with open(marker, encoding="utf-8") as handle:
                    payload = json.load(handle)
                verified = int(payload.get("verified", 0))
                total = int(payload.get("total", row["total_count"]))
                # A RUNNING row means the atomic replace never happened, so
                # the rollback is unfinished: mark UNKNOWN and require a rerun.
                state = JOB_UNKNOWN
                self.m.store.upsert_job({**dict(row), "state": JOB_UNKNOWN, "updated_at": now_iso()})
                self.m.store.commit()
                report = {
                    "job_id": row["job_id"],
                    "state": state,
                    "action": "rerun_required",
                    "verified_blocks": verified,
                    "unverified_blocks": total - verified,
                    "total_blocks": total,
                    "staging_path": staging,
                    "target_snapshot": row["target_snapshot"],
                    "manifest_version": row["manifest_version"],
                }
            else:
                self.m.store.upsert_job({**dict(row), "state": JOB_UNKNOWN, "updated_at": now_iso()})
                self.m.store.commit()
                report = {
                    "job_id": row["job_id"],
                    "state": JOB_UNKNOWN,
                    "action": "rerun_required",
                    "verified_blocks": int(row["verified_count"]),
                    "unverified_blocks": int(row["total_count"]) - int(row["verified_count"]),
                    "total_blocks": int(row["total_count"]),
                    "staging_path": staging,
                    "target_snapshot": row["target_snapshot"],
                    "manifest_version": row["manifest_version"],
                }
            self.m.logger.log("WARN", "rollback_recovered", snapshot_id=report.get("target_snapshot"),
                              manifest_version=report.get("manifest_version"),
                              **{"job_id": report["job_id"], "state": report["state"]})
            reports.append(report)
        return {"volume": volume_name, "reports": reports}

    def completion_token(self, volume_name: str, snapshot_id: str) -> dict[str, Any]:
        """Client-side completion check using snapshot id + manifest version."""
        manifest = self.m.store.get_manifest(volume_name)
        jobs = [
            row for row in self.m.store.list_jobs(volume_name)
            if row["target_snapshot"] == snapshot_id and row["state"] == JOB_COMMITTED
        ]
        return {
            "snapshot_id": snapshot_id,
            "manifest_version": int(manifest["format_version"]) if manifest else None,
            "committed": bool(jobs),
            "last_job_id": jobs[-1]["job_id"] if jobs else None,
        }
