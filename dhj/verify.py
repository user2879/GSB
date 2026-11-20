"""Independent verifier: recompute shuffle/join and compare against ledgers."""

import os

from . import bucket as bucket_mod
from . import fixedpoint as fp
from .errors import VerifyError
from .hashing import key_partition
from .jsonio import read_json
from .join import (
    build_ledger,
    join_partition,
    make_result_document,
    output_row_id,
)
from .shuffle import assign_rows, bucket_path


def verify_bucket_assignment(spec, directory, num_partitions, fraction_bits):
    """Recompute every partition assignment independently from source tables."""
    query = spec.query
    for side, table_name in (
        ("left", query.left_table),
        ("right", query.right_table),
    ):
        table = spec.tables[table_name]
        assignment, null_rows = assign_rows(
            table, query.join_key, num_partitions, fraction_bits
        )
        for partition, expected_ids in assignment.items():
            bkt = bucket_mod.read_bucket_file(
                bucket_path(directory, side, partition)
            )
            actual_ids = sorted(r.row_id for r in bkt.ordered_rows())
            expected_sorted = sorted(expected_ids)
            if actual_ids != expected_sorted:
                first = _first_id_diff(actual_ids, expected_sorted)
                raise VerifyError(
                    "bucket partition assignment mismatch",
                    partition=partition,
                    row_id=first,
                    detail=f"side={side}",
                )
            for row in bkt.ordered_rows():
                q = fp.quantize(
                    table.rows[row.row_id][query.join_key], fraction_bits
                )
                if key_partition(q, num_partitions) != partition:
                    raise VerifyError(
                        "row hash maps to a different partition",
                        partition=partition,
                        row_id=row.row_id,
                        detail=f"side={side}",
                    )


def _first_id_diff(actual, expected):
    for index, (a, b) in enumerate(zip(actual, expected)):
        if a != b:
            return a
    if len(actual) != len(expected):
        idx = min(len(actual), len(expected))
        return actual[idx] if idx < len(actual) else expected[idx]
    return None


def verify_join(spec, directory, result_dir, num_partitions, fraction_bits,
                stats_calculator):
    """Recompute joins independently and compare each partition ledger entry."""
    from .join import partition_result_path

    recomputed = {}
    for partition in range(num_partitions):
        left = bucket_mod.read_bucket_file(
            bucket_path(directory, "left", partition)
        )
        right = bucket_mod.read_bucket_file(
            bucket_path(directory, "right", partition)
        )
        rows, stats = join_partition(
            left, right, spec.query, spec.tables, fraction_bits,
            stats_calculator,
        )
        doc = make_result_document(partition, rows, stats, left, right, spec)
        recomputed[partition] = doc
        stored_path = partition_result_path(result_dir, partition)
        stored = read_json(stored_path)
        if stored["result_digest"] != doc["result_digest"]:
            row_id = _first_row_diff(stored["rows"], doc["rows"])
            raise VerifyError(
                "join result mismatch against independent recomputation",
                partition=partition,
                row_id=row_id,
            )
    ledger_path = os.path.join(result_dir, "ledger.json")
    if os.path.exists(ledger_path):
        stored_ledger = read_json(ledger_path)
        expected_ledger = build_ledger(
            spec, recomputed, num_partitions, fraction_bits
        )
        for entry_a, entry_b in zip(
            stored_ledger["partitions"], expected_ledger["partitions"]
        ):
            if entry_a != entry_b:
                row_id = None
                for a, b in zip(
                    entry_a.get("row_ids", []), entry_b.get("row_ids", [])
                ):
                    if a != b:
                        row_id = a
                        break
                raise VerifyError(
                    "ledger entry mismatch",
                    partition=entry_a.get("partition"),
                    row_id=row_id,
                )
    return recomputed


def _first_row_diff(stored_rows, expected_rows):
    stored_ids = [output_row_id(r) for r in stored_rows]
    expected_ids = [output_row_id(r) for r in expected_rows]
    for index, (a, b) in enumerate(zip(stored_ids, expected_ids)):
        if a != b:
            return a
    if len(stored_ids) != len(expected_ids):
        idx = min(len(stored_ids), len(expected_ids))
        return (
            stored_ids[idx] if idx < len(stored_ids) else expected_ids[idx]
        )
    return None


def verify_all(spec, directory, result_dir, num_partitions, fraction_bits,
               stats_calculator):
    verify_bucket_assignment(
        spec, directory, num_partitions, fraction_bits
    )
    recomputed = verify_join(
        spec, directory, result_dir, num_partitions, fraction_bits,
        stats_calculator,
    )
    return {
        "status": "ok",
        "partitions": num_partitions,
        "total_rows": sum(len(d["rows"]) for d in recomputed.values()),
    }
