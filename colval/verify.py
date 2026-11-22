"""Byte-identity verification between upgraded blocks and directly-written
current-format blocks of the same logical data.

compare_blocks raises DivergeError (E_DIVERGE) locating the first differing
block_id, column name and row number.
"""

import struct

from .errors import DivergeError
from .formats import bitmap_null_rows, content_digest


def _values_differ(dtype, a, b):
    if dtype == "int64":
        return int(a) != int(b)
    return struct.pack("<d", float(a)) != struct.pack("<d", float(b))


def compare_blocks(expected, actual):
    bid = expected.block_id
    if actual.block_id != bid:
        raise DivergeError("block id mismatch", block_id=bid,
                           field="block_id", expected=bid, actual=actual.block_id)
    for f in ("row_count", "quant_scale_exp", "batch_max_rows"):
        if getattr(expected, f) != getattr(actual, f):
            raise DivergeError("block header mismatch", block_id=bid, field=f,
                               expected=getattr(expected, f),
                               actual=getattr(actual, f))
    if len(expected.columns) != len(actual.columns):
        raise DivergeError("column count mismatch", block_id=bid, field="columns",
                           expected=len(expected.columns),
                           actual=len(actual.columns))
    for ec, ac in zip(expected.columns, actual.columns):
        if ec.name != ac.name:
            raise DivergeError("column order/name mismatch", block_id=bid,
                               column=ec.name, field="columns.name",
                               expected=ec.name, actual=ac.name)
        if ec.dtype != ac.dtype:
            raise DivergeError("column type mismatch", block_id=bid,
                               column=ec.name, field="type",
                               expected=ec.dtype, actual=ac.dtype)
        if ec.null_bitmap != ac.null_bitmap:
            en = set(bitmap_null_rows(ec.null_bitmap))
            an = set(bitmap_null_rows(ac.null_bitmap))
            row = next((r for r in range(expected.row_count)
                        if (r in en) != (r in an)), None)
            raise DivergeError("null bitmap mismatch", block_id=bid,
                               column=ec.name, row=row, field="null_bitmap")
        for r in range(expected.row_count):
            if _values_differ(ec.dtype, ec.values[r], ac.values[r]):
                raise DivergeError("column value mismatch", block_id=bid,
                                   column=ec.name, row=r, field="values",
                                   expected=ec.values[r], actual=ac.values[r])
    if content_digest(expected) != content_digest(actual):
        raise DivergeError("content digest mismatch", block_id=bid,
                           field="content_digest",
                           expected=content_digest(expected),
                           actual=content_digest(actual))


def verify_store_against_canonical(store, canonical_blocks):
    """canonical_blocks: dict block_id -> Block at CURRENT_FORMAT.

    Returns the list of verified block_ids (sorted, stable order).
    """
    verified = []
    for block_id in sorted(canonical_blocks):
        compare_blocks(canonical_blocks[block_id], store.get_block(block_id))
        verified.append(block_id)
    return verified
