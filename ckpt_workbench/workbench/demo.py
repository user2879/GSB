"""Repeatable end-to-end demo over isolated scenario copies."""

import filecmp
import glob
import json
import os
import shutil

from . import app
from .changeset import commit, full_recompute, load_change_set
from .gui import collect_view, render_text
from .inject import (
    inject_dangling_parent,
    inject_shard_corruption,
    inject_snapshot_mismatch,
    run_validation_expect_failure,
)
from .model import load_declaration
from .recovery import detect_inflight, replay_ledger, rerun_pending, rollback_to_parent
from .snapshot import compare_with_ledger, load_snapshot
from .verifier import verify_checkpoint


def _copy_tree(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _capture_shard_bytes(workdir):
    shard_dir = os.path.join(workdir, "shards")
    return {
        os.path.basename(path): open(path, "rb").read()
        for path in sorted(glob.glob(os.path.join(shard_dir, "*.bin")))
    }


def _capture_ledger_rows(workdir):
    ledger = app.open_ledger(workdir)
    rows = {
        "checkpoints": [ledger.get_checkpoint(cid) for cid in ledger.list_checkpoint_ids()],
        "changes": ledger.list_changes(),
    }
    ledger.close()
    return rows


def scenario_incremental(root, decl_path, change_paths):
    workdir = os.path.join(root, "s1_incremental")
    app.initialize(workdir, decl_path)
    decl = load_declaration(decl_path)
    report = {"unchanged_byte_checks": [], "verifications": []}

    for change_path in change_paths:
        before_bytes = _capture_shard_bytes(workdir)
        before_rows = _capture_ledger_rows(workdir)
        result = app.submit_change(workdir, decl_path, change_path)
        after_bytes = _capture_shard_bytes(workdir)
        change = load_change_set(change_path)

        ledger = app.open_ledger(workdir)
        cid = result["checkpoint_id"]
        verify_checkpoint(ledger, cid, change, decl)

        unchanged = result.get("unchanged", [])
        for shard_id in unchanged:
            parent_id = ledger.get_checkpoint(cid)["parent_id"]
            parent_blob = ledger.get_shard(parent_id, shard_id)["blob_path"]
            new_blob = ledger.get_shard(cid, shard_id)["blob_path"]
            same_path = os.path.abspath(parent_blob) == os.path.abspath(new_blob)
            same_bytes = same_path or filecmp.cmp(parent_blob, new_blob, shallow=False)
            # shard version must stay identical for untouched shards
            version_same = (
                ledger.get_shard(parent_id, shard_id)["shard_version"]
                == ledger.get_shard(cid, shard_id)["shard_version"]
            )
            report["unchanged_byte_checks"].append({
                "checkpoint": cid,
                "shard": shard_id,
                "blob_byte_identical": bool(same_bytes),
                "shard_version_unchanged": bool(version_same),
            })
        ledger.close()
        report["verifications"].append({"checkpoint": cid, "status": "MATCH"})

    # determinism: re-run the whole chain in a fresh workdir and compare digests
    mirror = os.path.join(root, "s1_mirror")
    app.initialize(mirror, decl_path)
    for change_path in change_paths:
        app.submit_change(mirror, decl_path, change_path)
    left = _capture_ledger_rows(workdir)
    right = _capture_ledger_rows(mirror)
    report["deterministic_chain_identical"] = _ledger_digest_signature(left) == _ledger_digest_signature(right)
    report["status"] = "OK"
    return report


def _ledger_digest_signature(rows):
    signature = []
    for cp in rows["checkpoints"]:
        signature.append((
            cp["checkpoint_id"],
            cp["content_digest"],
            tuple(sorted((s["shard_id"], s["shard_digest"], s["shard_version"]) for s in cp["shards"])),
        ))
    return signature


def scenario_snapshot(root, decl_path, change_paths):
    workdir = os.path.join(root, "s2_snapshot")
    app.initialize(workdir, decl_path)
    # snapshot the main chain (baseline + every supplied change set)
    for change_path in change_paths:
        app.submit_change(workdir, decl_path, change_path)
    snapshot_dir = os.path.join(root, "snapshots", "v1")
    app.snapshot_save(workdir, snapshot_dir)

    status_before = app.snapshot_status(workdir, snapshot_dir)

    # commit an extra change so the live ledger diverges from the snapshot
    divergent_change = os.path.join(root, "divergent.json")
    with open(divergent_change, "w", encoding="utf-8") as handle:
        json.dump({"change_id": "cp_divergent",
                   "ops": [{"op": "set_cursor", "value": 9999}]}, handle)
    app.submit_change(workdir, decl_path, divergent_change)
    ledger = app.open_ledger(workdir)
    diffs = compare_with_ledger(load_snapshot(snapshot_dir), ledger)
    ledger.close()

    # whole-set replacement must bring every checkpoint's shard blobs back
    restore = app.snapshot_restore(workdir, snapshot_dir)
    ledger = app.open_ledger(workdir)
    from .model import parse_shard_blob, shard_digest
    blob_ok = True
    for cid in ledger.list_checkpoint_ids():
        for shard in ledger.get_checkpoint(cid)["shards"]:
            tensors, momenta = parse_shard_blob(open(shard["blob_path"], "rb").read())
            blob_ok &= shard_digest(
                shard["tensors"], {"tensors": tensors, "momenta": momenta}
            ) == shard["shard_digest"]
    ledger.validate()
    ledger.close()
    after_restore = app.snapshot_status(workdir, snapshot_dir)
    return {
        "initial_status": status_before,
        "version_diffs": diffs,
        "replaced_checkpoints": restore["replaced_checkpoints"],
        "readonly_backup": restore["backup_dir"],
        "restored_blob_digests_match": blob_ok,
        "post_restore_status": after_restore,
        "status": "OK",
    }


def scenario_crash(root, decl_path, change_paths):
    decl = load_declaration(decl_path)
    pending = change_paths[0]
    pending_record = os.path.join(root, "s3_pending_change.json")
    shutil.copyfile(pending, pending_record)

    def make_crashed(name):
        workdir = os.path.join(root, name)
        app.initialize(workdir, decl_path)
        result = app.submit_change(workdir, decl_path, pending_record,
                                   crash_point="before_commit")
        ledger = app.open_ledger(workdir)
        inflight = detect_inflight(ledger)
        unknown = [item["shard_id"] for item in inflight["unknown_shards"]] if inflight else []
        head = ledger.latest_checkpoint_id()
        ledger.close()
        return workdir, result, inflight, unknown, head

    # Disposition A: rollback to the last committed parent checkpoint.
    wd_rollback, result, inflight, unknown, head = make_crashed("s3_crash_rollback")
    ledger = app.open_ledger(wd_rollback)
    rollback = rollback_to_parent(ledger)
    leftover = detect_inflight(ledger)
    replay_rollback = replay_ledger(ledger, decl, full_recompute)
    ledger.close()

    # Disposition B: clear UNKNOWN blobs and re-run the pending change set.
    wd_rerun, _r2, _i2, unknown_rerun, _h2 = make_crashed("s3_crash_rerun")
    ledger = app.open_ledger(wd_rerun)
    change = load_change_set(pending_record)
    rerun_result = rerun_pending(ledger, change, decl, commit)
    leftover_rerun = detect_inflight(ledger)
    replay_rerun = replay_ledger(ledger, decl, full_recompute)
    view = render_text(collect_view(ledger))
    ledger.close()

    return {
        "simulated_crash_checkpoint": result["checkpoint_id"],
        "unknown_shards_marked": unknown,
        "inflight_detail": inflight,
        "head_before_recovery": head,
        "rollback_disposition": {
            "result": rollback,
            "inflight_cleared": leftover is None,
            "replay": replay_rollback,
        },
        "rerun_disposition": {
            "unknown_shards_seen": unknown_rerun,
            "result_checkpoint": rerun_result["checkpoint_id"],
            "inflight_cleared": leftover_rerun is None,
            "replay": replay_rerun,
        },
        "text_view": view,
        "note": "no forced-kill exit report is fabricated; only committed records are trusted",
        "status": "OK",
    }

def scenario_faults(root, decl_path, change_paths):
    report = {}

    # E_INTEGRITY: shard digest corruption on an isolated copy
    wd = os.path.join(root, "s4_integrity")
    app.initialize(wd, decl_path)
    app.submit_change(wd, decl_path, change_paths[0])
    ledger = app.open_ledger(wd)
    target_cid = ledger.latest_checkpoint_id()
    target_shard = ledger.get_shards(target_cid)[0]["shard_id"]
    ledger.close()
    inject_shard_corruption(os.path.join(wd, "ledger.db"),
                            os.path.join(wd, "shards"), target_cid, target_shard)
    report["E_INTEGRITY"] = run_validation_expect_failure(
        os.path.join(wd, "ledger.db"), os.path.join(wd, "shards"))

    # E_CHAIN: dangling parent reference
    wd = os.path.join(root, "s4_chain")
    app.initialize(wd, decl_path)
    app.submit_change(wd, decl_path, change_paths[0])
    ledger = app.open_ledger(wd)
    target_cid = ledger.latest_checkpoint_id()
    ledger.close()
    inject_dangling_parent(os.path.join(wd, "ledger.db"),
                           os.path.join(wd, "shards"), target_cid)
    report["E_CHAIN"] = run_validation_expect_failure(
        os.path.join(wd, "ledger.db"), os.path.join(wd, "shards"))

    # E_SNAPSHOT: tampered snapshot content digest
    wd = os.path.join(root, "s4_snapshot")
    app.initialize(wd, decl_path)
    snap = os.path.join(root, "snapshots", "fault")
    app.snapshot_save(wd, snap)
    report["E_SNAPSHOT"] = inject_snapshot_mismatch(snap)
    return {"status": "OK", "faults": report}


def scenario_idempotency_concurrency(root, decl_path, change_paths):
    workdir = os.path.join(root, "s5_concurrency")
    app.initialize(workdir, decl_path)

    # idempotent duplicate submit
    idem = app.idempotent_submit(workdir, decl_path, change_paths[0])

    # distinct change sets serialized by ledger_version
    outcomes = app.concurrent_submit(workdir, decl_path, change_paths[1:])
    ledger = app.open_ledger(workdir)
    versions = [cp["ledger_version"] for cp in
                [ledger.get_checkpoint(cid) for cid in ledger.list_checkpoint_ids()]]
    parents = {cp["checkpoint_id"]: cp["parent_id"] for cp in
               [ledger.get_checkpoint(cid) for cid in ledger.list_checkpoint_ids()]}
    ledger.close()
    return {
        "idempotent": {
            "duplicate_skipped": idem["second"].get("idempotent", False),
            "checkpoint_count": idem["checkpoint_count"],
        },
        "concurrent_outcomes": outcomes,
        "ledger_versions_monotonic": versions == sorted(set(versions)),
        "parents": parents,
        "status": "OK",
    }


def run_demo(root, decl_path, change_paths):
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(root, exist_ok=True)
    return {
        "incremental": scenario_incremental(root, decl_path, change_paths),
        "snapshot": scenario_snapshot(root, decl_path, change_paths),
        "crash_recovery": scenario_crash(root, decl_path, change_paths),
        "fault_injection": scenario_faults(root, decl_path, change_paths),
        "idempotency_concurrency": scenario_idempotency_concurrency(
            root, decl_path, change_paths),
    }
