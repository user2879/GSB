"""Content fingerprints for rows, blocks, layouts, configs and results."""

import hashlib
import json


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("utf-8")


def sha256_hex(obj):
    return hashlib.sha256(canonical(obj)).hexdigest()


def row_fingerprint(dim, quant_level, frac_bits, qvec):
    return sha256_hex({
        "kind": "row",
        "dim": dim,
        "quant_level": quant_level,
        "frac_bits": frac_bits,
        "qvec": list(qvec),
    })


def block_fingerprint(block_index, rows_per_block, row_fps):
    return sha256_hex({
        "kind": "block",
        "block_index": block_index,
        "rows_per_block": rows_per_block,
        "row_fps": list(row_fps),
    })


def layout_fingerprint(block_layout):
    """block_layout: ordered list of [block_index, row_count] pairs."""
    return sha256_hex({
        "kind": "layout",
        "blocks": [list(pair) for pair in block_layout],
    })


def config_hash(structural):
    return sha256_hex({"kind": "config", "structural": structural})


def gather_fingerprint(mode, quant_level, frac_bits, ids, acc):
    return sha256_hex({
        "kind": "gather",
        "mode": mode,
        "quant_level": quant_level,
        "frac_bits": frac_bits,
        "ids": list(ids),
        "acc": list(acc),
    })
