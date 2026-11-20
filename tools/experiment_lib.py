"""Shared helpers for the repeatable local experiment and CLI tools."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dhj import bucket as bucket_mod
from dhj import join as join_mod
from dhj import stats as stats_mod
from dhj.jsonio import read_json, write_json
from dhj.schema import load_spec
from dhj.shuffle import bucket_path, compare_buckets


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def paths(root=ROOT):
    return {
        "data": os.path.join(root, "data"),
        "buckets": os.path.join(root, "buckets"),
        "snapshots": os.path.join(root, "snapshots"),
        "logs": os.path.join(root, "logs"),
        "artifacts": os.path.join(root, "artifacts"),
    }


def load_experiment_spec(sample_path, query_path=None):
    doc = read_json(sample_path)
    if query_path:
        doc["query"] = read_json(query_path)
    return load_spec(doc)


def stats_calculator(spec, fraction_bits):
    def calculate(rows):
        return stats_mod.compute_stats(
            rows, spec.query.projections, spec.tables, fraction_bits
        )

    return calculate


def write_ledger(result_dir, join_result, spec, num_partitions, fraction_bits):
    ledger = join_mod.build_ledger(
        spec,
        join_result["documents"],
        num_partitions,
        fraction_bits,
    )
    write_json(os.path.join(result_dir, "ledger.json"), ledger, atomic=True)
    return ledger


def fingerprints_for_directories(upgrade_dir, direct_dir, num_partitions):
    rows = []
    for side in ("left", "right"):
        for partition in range(num_partitions):
            a = bucket_mod.read_bucket_file(
                bucket_path(upgrade_dir, side, partition)
            )
            b = bucket_mod.read_bucket_file(
                bucket_path(direct_dir, side, partition)
            )
            rows.append((side, partition, a, b))
    return rows


def assert_byte_identical(upgrade_dir, direct_dir, num_partitions):
    mismatches = []
    for side, partition, a, b in fingerprints_for_directories(
        upgrade_dir, direct_dir, num_partitions
    ):
        if a.fingerprint() != b.fingerprint():
            diff = compare_buckets(a, b)
            mismatches.append({"side": side, **(diff or {})})
    return mismatches


