"""Session orchestration: upgrades, crash recovery and snapshot handling."""

import glob
import os
import shutil
import struct

from . import bucket as bucket_mod
from . import checkpoint as cp
from . import shuffle
from .errors import BucketError
from .jsonio import read_json
from .shuffle import bucket_path, generate_generation
from .snapshot import (
    build_snapshot,
    compare_snapshot,
    load_snapshot,
    replace_state_dirs,
    save_snapshot,
)


class Terminate(Exception):
    """Raised by progress callbacks to simulate a forced process termination."""


def make_logger(log_dir, name):
    from .logging_utils import JsonlLogger

    os.makedirs(log_dir, exist_ok=True)
    return JsonlLogger(os.path.join(log_dir, name))


def generate_version(directory, spec, version, num_partitions, fraction_bits,
                     phase=None, logger=None, progress_callback=None):
    phase = phase or cp.PHASE_GENERATE
    return generate_generation(
        directory,
        spec,
        version,
        num_partitions,
        fraction_bits,
        phase=phase,
        logger=logger,
        progress_callback=progress_callback,
    )


def _callback_fire(callback, *args):
    if callback is not None:
        callback(*args)


def upgrade_generation(source_dir, upgrade_dir, spec, num_partitions,
                       fraction_bits, target_version=bucket_mod.CURRENT_VERSION,
                       logger=None, progress_callback=None):
    """Upgrade every bucket with per-bucket checkpoints and read-only backups."""
    os.makedirs(upgrade_dir, exist_ok=True)
    if not os.listdir(upgrade_dir) or not glob.glob(
        os.path.join(source_dir, "*.dhj")
    ):
        for path in glob.glob(os.path.join(source_dir, "*.dhj")):
            shutil.copy2(path, os.path.join(upgrade_dir, os.path.basename(path)))
    checkpoint = cp.load_checkpoint(upgrade_dir, cp.PHASE_UPGRADE)
    report = []
    for side in ("left", "right"):
        for partition in range(num_partitions):
            if cp.is_done(checkpoint, partition, side):
                continue
            _callback_fire(progress_callback, side, partition)
            path = bucket_path(upgrade_dir, side, partition)
            old_bucket = bucket_mod.read_bucket_file(path)
            source_version = old_bucket.version
            source_table = (
                spec.tables[spec.query.left_table]
                if side == "left"
                else spec.tables[spec.query.right_table]
            )
            upgraded, added = bucket_mod.upgrade_bucket(
                old_bucket,
                target_version,
                source_table=source_table,
                join_key=spec.query.join_key,
            )
            file_digest = bucket_mod.write_bucket_file(
                path, upgraded, make_backup=True
            )
            cp.mark_entry(
                checkpoint,
                partition,
                side,
                source_version=source_version,
                target_version=target_version,
                added_fields=added,
                fingerprint=upgraded.fingerprint(),
                file_digest=file_digest,
                row_count=upgraded.row_count(),
            )
            cp.save_checkpoint(upgrade_dir, cp.PHASE_UPGRADE, checkpoint)
            report.append(
                {
                    "side": side,
                    "partition": partition,
                    "source_version": source_version,
                    "target_version": target_version,
                    "added_fields": added,
                }
            )
            if logger:
                logger.event(
                    "bucket_upgraded",
                    side=side,
                    partition=partition,
                    source_version=source_version,
                    target_version=target_version,
                    added_fields=added,
                )
    return report, checkpoint


def scan_bucket_states(directory, num_partitions):
    """Classify each expected bucket as vN or unknown (partial/unreadable)."""
    states = {}
    for side in ("left", "right"):
        for partition in range(num_partitions):
            path = bucket_path(directory, side, partition)
            if not os.path.exists(path):
                states[f"{side}:{partition}"] = "unknown"
                continue
            status, _ = bucket_mod.bucket_status(path)
            states[f"{side}:{partition}"] = status
    return states


def recover_bucket_directory(directory, spec, version, num_partitions,
                             fraction_bits, logger=None):
    """Remove unknown/partial buckets so checkpointed generation resumes them."""
    checkpoint = cp.load_checkpoint(directory, cp.PHASE_GENERATE)
    removed = []
    for side in ("left", "right"):
        for partition in range(num_partitions):
            key = f"{side}:{partition}"
            path = bucket_path(directory, side, partition)
            status, _ = (
                bucket_mod.bucket_status(path) if os.path.exists(path)
                else ("unknown", None)
            )
            if status == "unknown":
                if os.path.exists(path):
                    os.unlink(path)
                checkpoint["entries"].pop(key, None)
                removed.append(key)
            elif not cp.is_done(checkpoint, partition, side):
                os.unlink(path)
                removed.append(key)
    cp.save_checkpoint(directory, cp.PHASE_GENERATE, checkpoint)
    if logger and removed:
        logger.event("unknown_buckets_reset", buckets=removed)
    return removed


def recover_result_directory(result_dir, num_partitions, logger=None):
    checkpoint = cp.load_checkpoint(result_dir, cp.PHASE_JOIN)
    removed = []
    from .join import partition_result_path

    for partition in range(num_partitions):
        path = partition_result_path(result_dir, partition)
        valid = False
        if os.path.exists(path):
            try:
                doc = read_json(path)
                valid = doc.get("result_digest") and cp.is_done(
                    checkpoint, partition, "join"
                )
            except Exception:
                valid = False
        if not valid:
            if os.path.exists(path):
                os.unlink(path)
            checkpoint["entries"].pop(f"join:{partition}", None)
            removed.append(partition)
    cp.save_checkpoint(result_dir, cp.PHASE_JOIN, checkpoint)
    if logger and removed:
        logger.event("unknown_join_partitions_reset", partitions=removed)
    return removed


def collect_current_state(spec, bucket_dir, result_dir, num_partitions,
                          fraction_bits):
    from .join import partition_result_path

    bucket_fingerprints = {}
    versions = set()
    for side in ("left", "right"):
        for partition in range(num_partitions):
            bkt = bucket_mod.read_bucket_file(
                bucket_path(bucket_dir, side, partition)
            )
            key = f"{side}:{partition}"
            bucket_fingerprints[key] = bkt.fingerprint()
            versions.add(bkt.version)
    result_fingerprints = {}
    for partition in range(num_partitions):
        doc = read_json(partition_result_path(result_dir, partition))
        result_fingerprints[str(partition)] = doc["result_digest"]
    return {
        "source_digest": spec.source_digest,
        "partition_params_fingerprint": shuffle.shuffle_params_fingerprint(
            num_partitions, fraction_bits
        ),
        "generation_fingerprint": shuffle.generation_fingerprint(
            bucket_dir, num_partitions
        ),
        "completed_buckets": bucket_fingerprints,
        "join_result_digests": result_fingerprints,
        "bucket_format_versions": sorted(versions),
    }


def persist_snapshot(spec, snapshot_dir, bucket_dir, result_dir,
                     num_partitions, fraction_bits):
    state = collect_current_state(
        spec, bucket_dir, result_dir, num_partitions, fraction_bits
    )
    snapshot = build_snapshot(
        spec,
        bucket_dir,
        result_dir,
        bucket_versions=state["bucket_format_versions"],
        params_fingerprint=state["partition_params_fingerprint"],
        bucket_fingerprints=state["completed_buckets"],
        result_fingerprints=state["join_result_digests"],
        generation_fingerprint=state["generation_fingerprint"],
    )
    save_snapshot(snapshot_dir, snapshot)
    return snapshot, state


def open_or_replace_session(spec, snapshot_dir, bucket_dir, result_dir,
                            num_partitions, fraction_bits, logger=None):
    """Compare snapshot vs disk; mismatch triggers whole-state replacement."""
    snapshot = load_snapshot(snapshot_dir)
    if snapshot is None:
        return {"action": "new_session", "mismatches": []}
    try:
        state = collect_current_state(
            spec, bucket_dir, result_dir, num_partitions, fraction_bits
        )
    except (OSError, BucketError, KeyError, ValueError, struct.error) as exc:
        state = None
        mismatches = ["unreadable_state:" + type(exc).__name__]
    else:
        mismatches = compare_snapshot(snapshot, state)
    if not mismatches:
        return {"action": "resumed", "mismatches": []}
    backups = replace_state_dirs(
        bucket_dir, result_dir, snapshot_dir,
        reason=",".join(mismatches), logger=logger,
    )
    return {
        "action": "whole_state_replaced",
        "mismatches": mismatches,
        "backups": backups,
    }
