"""Integer encodings (raw / dictionary / run-length), deterministic selection
and decode-memory / compression-ratio estimation.

Design points required by the domain:

* every estimate is a deterministic integer computation with explicit
  intermediate terms and a documented quantization error bound;
* automatic encoding selection is a stable total order with explicit
  tie-breaking rules and a recorded reason;
* decoders validate their payloads and raise domain errors (never return
  partially decoded data).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from .errors import EncodingError
from .model import ColumnType, Encoding

# ---------------------------------------------------------------------------
# Payload binary formats (little-endian; all length prefixes unsigned 32-bit)
#
# RAW:          int32 count | int32 width | count * width value bytes
# DICTIONARY:   int32 count | int32 width | int32 card | card*width entries
#               | int32 index_width | count * index_width index bytes
# RLE:          int32 count | int32 width | int32 num_runs
#               | num_runs * (width value bytes | int32 run length)
# ---------------------------------------------------------------------------

_U32 = 4


def _np():
    import numpy as np  # local import: headless / install time friendly

    return np


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_block(values: Sequence[int], dtype: ColumnType, encoding: Encoding) -> bytes:
    np = _np()
    arr = np.asarray(list(values), dtype=dtype.numpy_dtype)
    width = dtype.width
    if encoding is Encoding.RAW:
        body = arr.tobytes()
        return _u32(len(arr)) + _u32(width) + body
    if encoding is Encoding.DICTIONARY:
        entries, inverse = np.unique(arr, return_inverse=True)
        card = int(entries.size)
        index_width = _uint_width_for(card)
        header = _u32(len(arr)) + _u32(width) + _u32(card)
        header += entries.astype(dtype.numpy_dtype).tobytes()
        header += _u32(index_width)
        header += inverse.astype(_index_numpy_dtype(index_width)).tobytes()
        return header
    if encoding is Encoding.RLE:
        run_values, run_lengths = _runs(arr, np)
        header = _u32(len(arr)) + _u32(width) + _u32(len(run_lengths))
        body = bytearray()
        for value, length in zip(run_values, run_lengths):
            body += int(value).to_bytes(width, "little", signed=True)
            body += _u32(int(length))
        return header + bytes(body)
    raise EncodingError(
        f"unsupported encoding marker: {encoding!r}", {"encoding": str(encoding)}
    )


def decode_block(payload: bytes, dtype: ColumnType, encoding: Encoding) -> List[int]:
    """Decode and fully validate one block payload.

    Raises :class:`EncodingError` when the declared encoding cannot consume the
    exact payload, or run/count invariants are violated.
    """
    np = _np()
    width = dtype.width
    if len(payload) < 2 * _U32:
        raise EncodingError(
            "payload shorter than minimal encoding header",
            {"payload_len": len(payload), "min_len": 2 * _U32},
        )
    count, declared_width = _read_u32(payload, 0), _read_u32(payload, 4)
    if declared_width != width:
        raise EncodingError(
            "declared column width does not match schema",
            {"declared_width": declared_width, "schema_width": width},
        )

    if encoding is Encoding.RAW:
        expected = 2 * _U32 + count * width
        if len(payload) != expected:
            raise EncodingError(
                "raw payload length does not match header row count",
                {"expected": expected, "actual": len(payload), "rows": count},
            )
        arr = np.frombuffer(payload, dtype=dtype.numpy_dtype, count=count, offset=2 * _U32)
        return arr.tolist()

    if encoding is Encoding.DICTIONARY:
        offset = 2 * _U32
        if len(payload) < offset + _U32:
            raise EncodingError("dictionary payload missing cardinality", {"offset": offset})
        card = _read_u32(payload, offset)
        offset += _U32
        entries_end = offset + card * width
        if len(payload) < entries_end + _U32:
            raise EncodingError("dictionary payload truncated in entry table", {"card": card})
        entries = np.frombuffer(payload, dtype=dtype.numpy_dtype, count=card, offset=offset)
        offset = entries_end
        index_width = _read_u32(payload, offset)
        offset += _U32
        if index_width not in (1, 2, 4):
            raise EncodingError("unsupported dictionary index width", {"index_width": index_width})
        expected = offset + count * index_width
        if len(payload) != expected:
            raise EncodingError(
                "dictionary payload length does not match index count",
                {"expected": expected, "actual": len(payload), "rows": count},
            )
        idx_dtype = _index_numpy_dtype(index_width)
        indexes = np.frombuffer(payload, dtype=idx_dtype, count=count, offset=offset)
        if int(indexes.size) and int(indexes.max(initial=0)) >= card:
            raise EncodingError(
                "dictionary index references missing entry",
                {"card": card, "max_index": int(indexes.max())},
            )
        return entries[indexes].tolist()

    if encoding is Encoding.RLE:
        offset = 2 * _U32
        if len(payload) < offset + _U32:
            raise EncodingError("rle payload missing run count", {"offset": offset})
        num_runs = _read_u32(payload, offset)
        offset += _U32
        expected = offset + num_runs * (width + _U32)
        if len(payload) != expected:
            raise EncodingError(
                "rle payload length does not match run count",
                {"expected": expected, "actual": len(payload), "runs": num_runs},
            )
        out: List[int] = []
        run_total = 0
        for _ in range(num_runs):
            value = int.from_bytes(payload[offset:offset + width], "little", signed=True)
            offset += width
            length = _read_u32(payload, offset)
            offset += _U32
            if length <= 0:
                raise EncodingError("rle run length must be positive", {"length": length})
            run_total += length
            out.extend([value] * length)
        if run_total != count:
            raise EncodingError(
                "rle run lengths do not sum to header row count",
                {"run_total": run_total, "rows": count},
            )
        return out

    raise EncodingError(
        f"unsupported encoding marker: {encoding!r}", {"encoding": str(encoding)}
    )


def _runs(arr, np) -> Tuple[List[int], List[int]]:
    if arr.size == 0:
        return [], []
    change = np.flatnonzero(arr[1:] != arr[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [arr.size]))
    values = arr[starts].tolist()
    lengths = (ends - starts).tolist()
    return values, [int(x) for x in lengths]


def _u32(value: int) -> bytes:
    return int(value).to_bytes(_U32, "little", signed=False)


def _read_u32(buf: bytes, offset: int) -> int:
    return int.from_bytes(buf[offset:offset + _U32], "little", signed=False)


def _uint_width_for(cardinality: int) -> int:
    if cardinality <= 0xFF:
        return 1
    if cardinality <= 0xFFFF:
        return 2
    return 4


def _index_numpy_dtype(width: int) -> str:
    return {1: "<u1", 2: "<u2", 4: "<u4"}[width]


# ---------------------------------------------------------------------------
# Deterministic estimation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EncodingEstimate:
    encoding: Encoding
    rows: int
    width: int
    cardinality: int
    num_runs: int
    encoded_payload_bytes: int
    decoded_peak_bytes: int
    compression_ratio: float
    quant_error_bound: float
    intermediate: Dict[str, Any]
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "encoding": self.encoding.value,
            "rows": self.rows,
            "width": self.width,
            "cardinality": self.cardinality,
            "num_runs": self.num_runs,
            "encoded_payload_bytes": self.encoded_payload_bytes,
            "decoded_peak_bytes": self.decoded_peak_bytes,
            "compression_ratio": self.compression_ratio,
            "quant_error_bound": self.quant_error_bound,
            "intermediate": self.intermediate,
            "reason": self.reason,
        }
        return data


def profile_values(values: Sequence[int]) -> Tuple[int, int]:
    """Return ``(cardinality, num_runs)`` in a single deterministic pass."""
    cardinality = 0
    num_runs = 0
    seen: Dict[int, None] = {}
    previous: Any = None
    for value in values:
        if value not in seen:
            seen[value] = None
            cardinality += 1
        if num_runs == 0 or value != previous:
            num_runs += 1
        previous = value
    return cardinality, num_runs


def estimate_encoding(
    values: Sequence[int], dtype: ColumnType, encoding: Encoding
) -> EncodingEstimate:
    rows = len(values)
    width = dtype.width
    cardinality, num_runs = profile_values(values)
    raw_bytes = rows * width

    if encoding is Encoding.RAW:
        encoded = 2 * _U32 + raw_bytes
        peak = raw_bytes
        reason = "raw: one slot per row, no dictionary or run bookkeeping"
        intermediate = {
            "raw_value_bytes": raw_bytes,
            "header_bytes": 2 * _U32,
            "formula": "2*4 + rows*width",
        }
    elif encoding is Encoding.DICTIONARY:
        index_width = _uint_width_for(cardinality)
        entry_bytes = cardinality * width
        index_bytes = rows * index_width
        encoded = 4 * _U32 + entry_bytes + index_bytes
        # Decoding materializes the value column and the index vector.
        peak = raw_bytes + rows * index_width
        reason = (
            f"dictionary: {cardinality} distinct entries, "
            f"{index_width}-byte indexes over {rows} rows"
        )
        intermediate = {
            "cardinality": cardinality,
            "index_width": index_width,
            "entry_bytes": entry_bytes,
            "index_bytes": index_bytes,
            "header_bytes": 4 * _U32,
            "formula": "4*4 + card*width + rows*index_width",
        }
    elif encoding is Encoding.RLE:
        run_bytes = num_runs * (width + _U32)
        encoded = 3 * _U32 + run_bytes
        # Decoding expands runs back to the full-width value column.
        peak = raw_bytes
        repeat = (rows / num_runs) if num_runs else 0.0
        reason = (
            f"rle: {num_runs} runs over {rows} rows, "
            f"mean repeat {repeat:.4f}"
        )
        intermediate = {
            "num_runs": num_runs,
            "run_bytes": run_bytes,
            "header_bytes": 3 * _U32,
            "mean_run_repeat": repeat,
            "formula": "3*4 + runs*(width+4)",
        }
    else:
        raise EncodingError(
            f"unsupported encoding marker: {encoding!r}", {"encoding": str(encoding)}
        )

    ratio, quant_bound = _quantized_ratio(raw_bytes, encoded)
    return EncodingEstimate(
        encoding=encoding,
        rows=rows,
        width=width,
        cardinality=cardinality,
        num_runs=num_runs,
        encoded_payload_bytes=encoded,
        decoded_peak_bytes=peak,
        compression_ratio=ratio,
        quant_error_bound=quant_bound,
        intermediate=intermediate,
        reason=reason,
    )


def _quantized_ratio(raw_bytes: int, encoded_bytes: int) -> Tuple[float, float]:
    """Ratio rounded to 4 decimals; bound is half of the last decimal place."""
    if raw_bytes == 0:
        return 0.0, 0.0
    exact = encoded_bytes / raw_bytes
    ratio = round(exact, 4)
    return ratio, 0.5e-4


# Explicit, documented total order for automatic selection.
_SELECTION_ORDER = (Encoding.RLE, Encoding.DICTIONARY, Encoding.RAW)


def select_encoding(values: Sequence[int], dtype: ColumnType) -> Tuple[Encoding, Dict[str, Any]]:
    """Deterministic automatic selection.

    Rule: pick the candidate with the smallest encoded payload. Ties are
    resolved by the fixed priority RLE > DICTIONARY > RAW (run-length wins when
    repetition is as compact as a dictionary; raw is only kept when neither
    compression helps). Every estimate is exposed so the decision is auditable.
    """
    estimates = {
        enc: estimate_encoding(values, dtype, enc) for enc in _SELECTION_ORDER
    }
    ranked = sorted(
        _SELECTION_ORDER,
        key=lambda enc: (estimates[enc].encoded_payload_bytes, _SELECTION_ORDER.index(enc)),
    )
    chosen = ranked[0]
    best_size = estimates[chosen].encoded_payload_bytes
    tied = [
        enc.value
        for enc in _SELECTION_ORDER
        if estimates[enc].encoded_payload_bytes == best_size
    ]
    if len(tied) > 1:
        reason = (
            "tie on encoded payload bytes; resolved by fixed priority "
            "rle > dictionary > raw"
        )
        tie_broken = True
    else:
        reason = f"smallest encoded payload ({best_size} bytes) among candidates"
        tie_broken = False
    decision = {
        "chosen": chosen.value,
        "tie_broken": tie_broken,
        "tied_candidates": tied,
        "reason": reason,
        "estimates": {enc.value: estimates[enc].to_dict() for enc in _SELECTION_ORDER},
    }
    return chosen, decision
