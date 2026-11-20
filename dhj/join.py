"""Partition-local hash join with resumable per-partition results."""

import os

from . import bucket as bucket_mod
from . import checkpoint as cp
from . import fixedpoint as fp
from .errors import ResumeError
from .jsonio import digest_hex, write_json

ERROR_SOURCES = (
    "none",
    "column_quantization",
    "join_key_quantization",
)


def eval_predicate(value, predicate):
    if value is None:
        return False
    op = predicate.op
    rhs = predicate.value
    if op == "=":
        return value == rhs
    if op == "!=":
        return value != rhs
    if op == "<":
        return value < rhs
    if op == "<=":
        return value <= rhs
    if op == ">":
        return value > rhs
    if op == ">=":
        return value >= rhs
    raise ValueError(f"unsupported operator {op}")


def row_passes(row, predicates):
    return all(eval_predicate(row[p.column], p) for p in predicates)


def _float_error_source(ref, join_key_ref, abs_error, join_key_error):
    if abs_error <= 0.0:
        return "none"
    if join_key_error and ref == join_key_ref:
        return "join_key_quantization"
    return "column_quantization"


def build_output_row(left_bucket_row, right_bucket_row, query, tables,
                     fraction_bits, partition):
    left_table = query.left_table
    join_ref_left = f"{left_table}.{query.join_key}"
    projection = {}
    source_errors = []
    for ref in query.projections:
        table_name, col_name = ref.split(".")
        source_row = (
            left_bucket_row if table_name == left_table else right_bucket_row
        )
        value = source_row.values[col_name]
        dtype = tables[table_name].dtype_map[col_name]
        if dtype == "float":
            q = fp.quantize(value, fraction_bits)
            fixed = fp.dequantize(q, fraction_bits)
            abs_error = abs(fixed - float(value)) if value is not None else 0.0
            source = _float_error_source(
                ref, join_ref_left, abs_error, True
            )
            projection[ref] = {
                "float_ref": float(value) if value is not None else None,
                "fixed_point": q,
                "fixed_point_value": fixed if value is not None else None,
                "abs_error": abs_error,
                "error_source": source,
            }
            if value is not None and abs_error > 0.0:
                source_errors.append((abs_error, source, ref))
        else:
            projection[ref] = {
                "int_value": value,
                "fixed_point": fp.quantize(value, fraction_bits),
            }
    if source_errors:
        source_errors.sort(key=lambda item: (-item[0], item[2]))
        primary_source = source_errors[0][1]
    else:
        primary_source = "none"
    left_q = left_bucket_row.q_key
    right_q = right_bucket_row.q_key
    key_quantization_match = left_q == right_q
    return {
        "partition": partition,
        "left_row_id": left_bucket_row.row_id,
        "right_row_id": right_bucket_row.row_id,
        "left_quantized_key": left_q,
        "right_quantized_key": right_q,
        "key_quantization_match": key_quantization_match,
        "primary_error_source": primary_source,
        "projection": projection,
    }


def output_row_id(row):
    return f"{row['partition']}:{row['left_row_id']}-{row['right_row_id']}"


class HashTable:
    def __init__(self):
        self._slots = {}

    def add(self, row):
        if row.q_key == fp.NULL_Q:
            return
        self._slots.setdefault(row.q_key, []).append(row)

    def lookup(self, q_key):
        return self._slots.get(q_key, ())


def join_partition(left_bucket, right_bucket, query, tables, fraction_bits,
                   stats_calculator):
    """Build on the left (build side), probe with right rows in order."""
    table = HashTable()
    for row in left_bucket.ordered_rows():
        if row_passes(row.values, query.predicates["left"]):
            table.add(row)
    output_rows = []
    for probe in right_bucket.ordered_rows():
        if probe.q_key == fp.NULL_Q:
            continue
        if not row_passes(probe.values, query.predicates["right"]):
            continue
        for build_row in table.lookup(probe.q_key):
            output_rows.append(
                build_output_row(
                    build_row, probe, query, tables, fraction_bits,
                    left_bucket.partition,
                )
            )
    output_rows.sort(
        key=lambda r: (
            r["left_quantized_key"],
            r["left_row_id"],
            r["right_row_id"],
        )
    )
    stats = stats_calculator(output_rows)
    return output_rows, stats


def partition_result_path(result_dir, partition):
    return os.path.join(result_dir, f"join-p{partition:05d}.json")


def run_join(
    bucket_directory,
    result_directory,
    spec,
    num_partitions,
    fraction_bits,
    stats_calculator,
    reuse=None,
    logger=None,
    progress_callback=None,
):
    """Run every partition join, resuming completed partitions.

    ``reuse`` is an optional mapping ``partition -> result document`` produced
    by an earlier run whose buckets are still byte-identical. Reused results
    are re-validated by a fresh join in :func:`cross_check_resume`.
    """
    os.makedirs(result_directory, exist_ok=True)
    checkpoint = cp.load_checkpoint(result_directory, cp.PHASE_JOIN)
    reuse = reuse or {}
    documents = {}
    reused_partitions = []
    recomputed_partitions = []
    for partition in range(num_partitions):
        if cp.is_done(checkpoint, partition, "join"):
            doc = _load_existing(result_directory, partition)
            documents[partition] = doc
            continue
        if partition in reuse:
            doc = reuse[partition]
            write_json(
                partition_result_path(result_directory, partition),
                doc,
                atomic=True,
            )
            cp.mark_entry(
                checkpoint,
                partition,
                "join",
                result_digest=doc["result_digest"],
                row_count=len(doc["rows"]),
                reused=True,
            )
            cp.save_checkpoint(result_directory, cp.PHASE_JOIN, checkpoint)
            documents[partition] = doc
            reused_partitions.append(partition)
            if logger:
                logger.event("join_reused", partition=partition)
            continue
        if progress_callback is not None:
            progress_callback(partition)
        left_bucket = bucket_mod.read_bucket_file(
            bucket_mod_path(bucket_directory, "left", partition)
        )
        right_bucket = bucket_mod.read_bucket_file(
            bucket_mod_path(bucket_directory, "right", partition)
        )
        rows, stats = join_partition(
            left_bucket,
            right_bucket,
            spec.query,
            spec.tables,
            fraction_bits,
            stats_calculator,
        )
        doc = make_result_document(
            partition, rows, stats, left_bucket, right_bucket, spec
        )
        write_json(
            partition_result_path(result_directory, partition), doc, atomic=True
        )
        cp.mark_entry(
            checkpoint,
            partition,
            "join",
            result_digest=doc["result_digest"],
            row_count=len(rows),
            reused=False,
        )
        cp.save_checkpoint(result_directory, cp.PHASE_JOIN, checkpoint)
        documents[partition] = doc
        recomputed_partitions.append(partition)
        if logger:
            logger.event(
                "join_partition_done",
                partition=partition,
                rows=len(rows),
            )
    ledger = build_ledger(spec, documents, num_partitions, fraction_bits)
    return {
        "documents": documents,
        "ledger": ledger,
        "reused_partitions": reused_partitions,
        "recomputed_partitions": recomputed_partitions,
    }


def bucket_mod_path(directory, side, partition):
    from .shuffle import bucket_path

    return bucket_path(directory, side, partition)


def _load_existing(result_directory, partition):
    from .jsonio import read_json

    return read_json(partition_result_path(result_directory, partition))


def make_result_document(partition, rows, stats, left_bucket, right_bucket, spec):
    payload = {
        "partition": partition,
        "num_partitions": left_bucket.num_partitions,
        "fraction_bits": left_bucket.fraction_bits,
        "left_bucket_fingerprint": left_bucket.fingerprint(),
        "right_bucket_fingerprint": right_bucket.fingerprint(),
        "rows": rows,
        "stats": stats,
    }
    payload["result_digest"] = digest_hex(
        {k: v for k, v in payload.items() if k != "result_digest"}
    )
    return payload


def build_ledger(spec, documents, num_partitions, fraction_bits):
    entries = []
    total_rows = 0
    for partition in sorted(documents):
        doc = documents[partition]
        total_rows += len(doc["rows"])
        entries.append(
            {
                "partition": partition,
                "row_count": len(doc["rows"]),
                "result_digest": doc["result_digest"],
                "left_bucket_fingerprint": doc["left_bucket_fingerprint"],
                "right_bucket_fingerprint": doc["right_bucket_fingerprint"],
                "row_ids": [output_row_id(r) for r in doc["rows"]],
            }
        )
    return {
        "query_id": spec.query.query_id,
        "num_partitions": num_partitions,
        "fraction_bits": fraction_bits,
        "total_row_count": total_rows,
        "partitions": entries,
    }


def cross_check_resume(full_documents, resume_documents):
    """Raise E_RESUME at the first shard/row whose resumed result differs."""
    if set(full_documents) != set(resume_documents):
        first = sorted(set(full_documents) ^ set(resume_documents))[0]
        raise ResumeError(
            "resume partition set differs from full recomputation",
            partition=first,
        )
    for partition in sorted(full_documents):
        full = full_documents[partition]
        resumed = resume_documents[partition]
        if full["result_digest"] != resumed["result_digest"]:
            full_ids = [output_row_id(r) for r in full["rows"]]
            resume_ids = [output_row_id(r) for r in resumed["rows"]]
            row_id = None
            for idx, (a, b) in enumerate(zip(full_ids, resume_ids)):
                if a != b:
                    row_id = a
                    break
            if row_id is None and len(full_ids) != len(resume_ids):
                row_id = full_ids[min(len(full_ids), len(resume_ids))]
            if row_id is None and full["rows"] != resumed["rows"]:
                row_id = output_row_id(full["rows"][0]) if full["rows"] else None
            raise ResumeError(
                "resume result differs from full recomputation",
                partition=partition,
                row_id=row_id,
            )
