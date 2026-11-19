#!/usr/bin/env python3
"""Repeatable end-to-end demo of the four required scenarios.

Scenarios, in fixed order:
  1. first snapshot + rollback (bitmap vs full path, bit-identical)
  2. multi-snapshot chain rollback (chain path synthesized state)
  3. manifest format-version full replacement (read-only backup)
  4. forced termination mid-rollback + restart recovery report

The script prints a stable, greppable report and exits non-zero on failure.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from block_snap import (  # noqa: E402
    BlockSnapService,
    DomainError,
    JOB_UNKNOWN,
    PATH_BITMAP,
    PATH_CHAIN,
    PATH_FULL,
    block_digest,
)
from block_snap.samples import PROFILES, build_profile, reset_workspace  # noqa: E402

WORKSPACE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workdata")


def heading(index: int, title: str) -> None:
    print("\n" + "=" * 72)
    print(f"SCENARIO {index}: {title}")
    print("=" * 72)


def show(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


def assert_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")
    print(f"  [ok] {label}")


def main() -> int:
    reset_workspace(WORKSPACE)
    with BlockSnapService(WORKSPACE) as service:
        # ------------------------------------------------------------------
        heading(1, "first snapshot & rollback - bitmap vs full identical")
        profile = PROFILES["large_low_change"]
        build_profile(service, profile)
        volume = service.open_volume(profile.name)
        before = volume.block_digests()
        snapshots = service.list_snapshots(profile.name)
        base_id = snapshots[0]["snapshot_id"]
        plan = service.plan_rollback(profile.name, base_id)
        show(plan)
        cross = service.cross_check_paths(profile.name, base_id)
        show(cross)
        assert_equal(cross["identical"], True, "three paths produce identical images")
        result = service.rollback(profile.name, base_id)
        assert_equal(result["state"], "COMMITTED", "rollback committed atomically")
        after = service.open_volume(profile.name).block_digests()
        from block_snap import deterministic_block
        baseline = {
            b: block_digest(deterministic_block(profile.seed, b, profile.block_size))
            for b in range(profile.block_count)
        }
        assert_equal(after, baseline, "live volume equals deterministic baseline blocks")
        token = service.completion_token(profile.name, base_id)
        assert_equal(token["committed"], True, "completion token confirms committed rollback")

        # ------------------------------------------------------------------
        heading(2, "multi-snapshot chain rollback - chain path synthesis")
        high = PROFILES["large_high_change"]
        build_profile(service, high)
        snaps = service.list_snapshots(high.name)
        first_id = snaps[0]["snapshot_id"]
        chain_plan = service.plan_rollback(high.name, first_id)
        show(chain_plan)
        assert_equal(chain_plan["path"], PATH_CHAIN, "deep span selects chain path")
        cross = service.cross_check_paths(high.name, first_id)
        assert_equal(cross["identical"], True, "chain path bit-identical to bitmap/full")
        target_state = service.rollback_engine.target_state(high.name, first_id)
        result = service.rollback(high.name, first_id, force_path=PATH_CHAIN)
        live = service.open_volume(high.name).block_digests()
        assert_equal(live, {b: block_digest(data) for b, data in target_state.items()},
                     "chain rollback materializes synthesized target state")
        for forced in (PATH_BITMAP, PATH_FULL):
            # rebuild and force the other paths to prove identical output
            reset_vol = high.name
            service.rollback(high.name, snaps[-1]["snapshot_id"], force_path=PATH_CHAIN)
            plan_forced = service.rollback(high.name, first_id, force_path=forced)
            digest_forced = service.open_volume(high.name).whole_digest()
            service.rollback(high.name, snaps[-1]["snapshot_id"], force_path=PATH_CHAIN)
            service.rollback(high.name, first_id, force_path=PATH_CHAIN)
            digest_chain = service.open_volume(high.name).whole_digest()
            assert_equal(digest_forced, digest_chain, f"{forced} path image equals chain path image")
        show({"path_reports": "all three paths verified byte-identical"})

        # ------------------------------------------------------------------
        heading(3, "manifest format-version full replacement")
        small = PROFILES["small_many_snapshots"]
        build_profile(service, small)
        before_view = service.manifest_view(small.name)
        entries = []
        for row in service.manager.store.list_snapshots(small.name):
            refs = service.manager.store.ref_map(small.name, row["snapshot_id"])
            entries.append({
                "snapshot_id": row["snapshot_id"],
                "name": row["name"],
                "version_no": row["version_no"],
                "parent_id": row["parent_id"],
                "bitmap_hex": bytes(row["bitmap"]).hex(),
                "refs": {str(b): h for b, h in refs.items()},
            })
        replacement = service.replace_manifest(small.name, before_view["format_version"] + 1, entries)
        show(replacement)
        assert_equal(replacement["version_changed"], True, "format version advanced")
        assert_equal(replacement["replaced_entries"], len(entries), "all entries replaced, not merged")
        assert os.path.exists(replacement["backup_path"]), "read-only backup retained"
        assert_equal(os.stat(replacement["backup_path"]).st_mode & 0o200, 0, "backup is read-only")
        after_view = service.manifest_view(small.name)
        assert_equal(after_view["digest_match"], True, "new manifest digest matches entry set")
        audit = service.audit(small.name)
        assert_equal(audit["ok"], True, "audit passes after manifest replacement")
        # stale verdict replacement on manifest version change
        verdict = service.audit_verdict(small.name)
        assert_equal(verdict["manifest_version"], after_view["format_version"],
                     "audit conclusion versioned to new manifest")

        # ------------------------------------------------------------------
        heading(4, "forced termination & restart recovery")
        target = service.list_snapshots(high.name)[0]["snapshot_id"]
        service.rollback(high.name, service.list_snapshots(high.name)[-1]["snapshot_id"])
        try:
            service.rollback(high.name, target, force_path=PATH_FULL, crash_after_blocks=3)
        except SystemExit as exc:
            print(f"  [injected crash] {exc}")
        # simulate restart: brand new service instance on the same workspace
        service.close()
        restarted = BlockSnapService(WORKSPACE)
        try:
            recovery = restarted.recover(high.name)
            show(recovery)
            unfinished = [r for r in recovery["reports"] if r["state"] == JOB_UNKNOWN]
            assert bool(unfinished), "interrupted rollback marked UNKNOWN"
            report = unfinished[-1]
            assert report["unverified_blocks"] > 0, "report lists unverified block count"
            assert report["verified_blocks"] >= 3, "report lists verified block count from marker"
            assert_equal(report["action"], "rerun_required", "recovery demands rerun")
            rerun = restarted.rollback(high.name, target, verify_all_paths=True)
            assert_equal(rerun["state"], "COMMITTED", "rerun completes with atomic switch")
            final_token = restarted.completion_token(high.name, target)
            assert_equal(final_token["committed"], True, "completion verifiable after rerun")
            audit = restarted.audit(high.name)
            assert_equal(audit["ok"], True, "post-recovery audit clean")
            show({"error_log_codes": sorted({e["code"] for e in restarted.query_logs() if e["code"]})})
        finally:
            restarted.close()

        print("\nALL DEMO SCENARIOS PASSED")
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, DomainError) as exc:
        print(f"\nDEMO FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
