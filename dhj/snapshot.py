"""Session snapshots with version identification and whole-state replacement."""

import os
import shutil

from .errors import SnapshotError
from .jsonio import digest_hex, read_json, sha256_bytes, write_json

SNAPSHOT_VERSION = 1
SNAPSHOT_NAME = "snapshot.json"


def snapshot_path(snapshot_dir):
    return os.path.join(snapshot_dir, SNAPSHOT_NAME)


def build_snapshot(
    spec,
    bucket_dir,
    result_dir,
    bucket_versions,
    params_fingerprint,
    bucket_fingerprints,
    result_fingerprints,
    generation_fingerprint,
):
    body = {
        "snapshot_version": SNAPSHOT_VERSION,
        "source_digest": spec.source_digest,
        "table_digests": {
            name: table.digest for name, table in spec.tables.items()
        },
        "bucket_format_versions": sorted(set(bucket_versions)),
        "bucket_version_by_side": dict(bucket_versions)
        if isinstance(bucket_versions, dict)
        else {},
        "partition_params_fingerprint": params_fingerprint,
        "generation_fingerprint": generation_fingerprint,
        "completed_buckets": bucket_fingerprints,
        "join_result_digests": result_fingerprints,
    }
    body["content_digest"] = digest_hex(
        {k: v for k, v in body.items() if k != "content_digest"}
    )
    return body


def save_snapshot(snapshot_dir, snapshot):
    os.makedirs(snapshot_dir, exist_ok=True)
    path = snapshot_path(snapshot_dir)
    if os.path.exists(path):
        backup = os.path.join(
            snapshot_dir,
            "snapshot.backup.json",
        )
        if not os.path.exists(backup):
            shutil.copy2(path, backup)
        os.chmod(backup, 0o444)
    write_json(path, snapshot, atomic=True)


def load_snapshot(snapshot_dir):
    path = snapshot_path(snapshot_dir)
    if not os.path.exists(path):
        return None
    data = read_json(path)
    want = digest_hex(
        {k: v for k, v in data.items() if k != "content_digest"}
    )
    if want != data.get("content_digest"):
        raise SnapshotError(
            "snapshot content digest mismatch",
            {"path": path},
        )
    if data.get("snapshot_version") != SNAPSHOT_VERSION:
        raise SnapshotError(
            f"unsupported snapshot version {data.get('snapshot_version')}",
            {"path": path},
        )
    return data


def compare_snapshot(snapshot, current_state):
    """Return list of mismatched state components (empty means identical)."""
    checks = {
        "source_digest": current_state.get("source_digest"),
        "partition_params_fingerprint": current_state.get(
            "partition_params_fingerprint"
        ),
        "generation_fingerprint": current_state.get("generation_fingerprint"),
    }
    mismatches = []
    for key, current_value in checks.items():
        if snapshot.get(key) != current_value:
            mismatches.append(key)
    if snapshot.get("completed_buckets") != current_state.get(
        "completed_buckets"
    ):
        mismatches.append("completed_buckets")
    if snapshot.get("join_result_digests") != current_state.get(
        "join_result_digests"
    ):
        mismatches.append("join_result_digests")
    if snapshot.get("bucket_format_versions") != current_state.get(
        "bucket_format_versions"
    ):
        mismatches.append("bucket_format_versions")
    return mismatches


def replace_state_dirs(bucket_dir, result_dir, snapshot_dir, reason, logger=None):
    """Whole-state replacement: move current dirs aside as read-only backups."""
    backed_up = []
    for directory in (bucket_dir, result_dir, snapshot_dir):
        if not os.path.isdir(directory):
            continue
        backup = f"{directory}.replaced"
        counter = 0
        unique_backup = backup
        while os.path.exists(unique_backup):
            counter += 1
            unique_backup = f"{backup}-{counter}"
        shutil.move(directory, unique_backup)
        for root, _dirs, files in os.walk(unique_backup):
            for name in files:
                try:
                    os.chmod(os.path.join(root, name), 0o444)
                except OSError:
                    pass
        backed_up.append(unique_backup)
        if logger:
            logger.event(
                "state_replaced",
                reason=reason,
                directory=directory,
                backup=unique_backup,
            )
    os.makedirs(bucket_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(snapshot_dir, exist_ok=True)
    return backed_up


def file_digest(path):
    with open(path, "rb") as handle:
        return sha256_bytes(handle.read())
