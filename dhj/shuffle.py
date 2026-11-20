"""Deterministic key-hash shuffle partition assignment and bucket generation."""

import os

from . import bucket as bucket_mod
from . import checkpoint as cp
from .hashing import key_partition
from .jsonio import digest_hex

HASH_RULE = "FNV1A64(q_int) mod num_partitions"


def shuffle_params_fingerprint(num_partitions, fraction_bits):
    return digest_hex(
        {
            "hash_rule": HASH_RULE,
            "num_partitions": num_partitions,
            "fraction_bits": fraction_bits,
        }
    )


def assign_rows(table, join_key, num_partitions, fraction_bits):
    """Return (assignment dict, null row ids). Assignment is deterministic."""
    assignment = {p: [] for p in range(num_partitions)}
    null_rows = []
    for row_id, row in enumerate(table.rows):
        value = row[join_key]
        if value is None:
            null_rows.append(row_id)
            continue
        from .fixedpoint import quantize

        q = quantize(value, fraction_bits)
        partition = key_partition(q, num_partitions)
        assignment[partition].append(row_id)
    for row_ids in assignment.values():
        row_ids.sort()
    return assignment, null_rows


def bucket_path(directory, side, partition):
    return os.path.join(directory, f"{side}-{bucket_mod.bucket_filename(partition)}")


def generate_generation(
    directory,
    spec,
    version,
    num_partitions,
    fraction_bits,
    phase=cp.PHASE_GENERATE,
    logger=None,
    progress_callback=None,
):
    """Write all buckets of ``version`` for both sides, resumable via checkpoint.

    ``progress_callback(side, partition)`` may raise :class:`Terminate` to
    simulate a forced kill; the caller restarts and resumes.
    """
    os.makedirs(directory, exist_ok=True)
    checkpoint = cp.load_checkpoint(directory, phase)
    query = spec.query
    sides = (
        ("left", spec.tables[query.left_table]),
        ("right", spec.tables[query.right_table]),
    )
    assignments = {}
    nulls = {}
    for side, table in sides:
        assignments[side], nulls[side] = assign_rows(
            table, query.join_key, num_partitions, fraction_bits
        )

    written = []
    for side, table in sides:
        for partition in range(num_partitions):
            if cp.is_done(checkpoint, partition, side):
                continue
            if progress_callback is not None:
                progress_callback(side, partition)
            bkt = bucket_mod.make_bucket(
                side=side,
                table=table,
                join_key=query.join_key,
                partition=partition,
                num_partitions=num_partitions,
                fraction_bits=fraction_bits,
                row_ids=assignments[side][partition],
                version=version,
            )
            path = bucket_path(directory, side, partition)
            file_digest = bucket_mod.write_bucket_file(path, bkt)
            cp.mark_entry(
                checkpoint,
                partition,
                side,
                version=version,
                fingerprint=bkt.fingerprint(),
                file_digest=file_digest,
                row_count=bkt.row_count(),
            )
            cp.save_checkpoint(directory, phase, checkpoint)
            written.append(path)
            if logger:
                logger.event(
                    "bucket_written",
                    side=side,
                    partition=partition,
                    version=version,
                    rows=bkt.row_count(),
                )

    manifest = {
        "hash_rule": HASH_RULE,
        "num_partitions": num_partitions,
        "fraction_bits": fraction_bits,
        "version": version,
        "params_fingerprint": shuffle_params_fingerprint(
            num_partitions, fraction_bits
        ),
        "null_rows": nulls,
        "buckets": checkpoint["entries"],
    }
    return manifest, checkpoint


def compare_buckets(bucket_a, bucket_b):
    """Byte-identity equivalence for assignment, order table and summaries.

    Returns ``None`` when equivalent, otherwise a dict locating the first
    inconsistent row identity.
    """
    if bucket_a.partition != bucket_b.partition:
        return {
            "first_index": 0,
            "row_id": None,
            "detail": "partition assignment differs",
        }
    rows_a = bucket_a.ordered_rows()
    rows_b = bucket_b.ordered_rows()
    mismatch = bucket_mod.first_digest_mismatch(rows_a, rows_b)
    if mismatch is None:
        return None
    index, row_id = mismatch
    return {
        "first_index": index,
        "row_id": row_id,
        "detail": "first row summary or row identity mismatch",
    }


def read_all_buckets(directory, num_partitions):
    out = {"left": {}, "right": {}}
    for side in ("left", "right"):
        for partition in range(num_partitions):
            path = bucket_path(directory, side, partition)
            out[side][partition] = bucket_mod.read_bucket_file(path)
    return out


def generation_fingerprint(directory, num_partitions):
    parts = []
    for side in ("left", "right"):
        for partition in range(num_partitions):
            bkt = bucket_mod.read_bucket_file(
                bucket_path(directory, side, partition)
            )
            parts.append(bkt.fingerprint())
    return digest_hex({"bucket_fingerprints": parts})
