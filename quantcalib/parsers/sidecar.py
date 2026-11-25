"""Version downgrade/upgrade with an explicit sidecar.

Downgrading a q3 receipt to q1 cannot express the group scale digest or the
covered duplicate calibration references. Those fields are therefore written
to a JSON sidecar keyed by layer name; upgrading back to q3 restores them from
the sidecar instead of fabricating defaults.
"""

from __future__ import annotations

import json
from pathlib import Path

from quantcalib.errors import E_RECEIPT_FORMAT
from quantcalib.models import UnifiedReceipt
from quantcalib.parsers.receipts import parse_q1, parse_q3, write_q1, write_q3

_SIDECAR_FIELDS = (
    "group_scale_digest", "duplicate_refs", "weight_snapshot",
    "param_version", "time_ms",
)


def downgrade_q3_to_q1(q3_path: str | Path, q1_path: str | Path,
                       sidecar_path: str | Path) -> list[UnifiedReceipt]:
    receipts = parse_q3(q3_path)
    sidecar: dict[str, dict] = {}
    for receipt in receipts:
        receipt.source_version = "q1"
        receipt.missing_fields = [
            "param_version", "time_ms", "group_scale_digest",
            "duplicate_refs", "weight_snapshot",
        ]
        sidecar[receipt.layer] = {
            field: getattr(receipt, field) for field in _SIDECAR_FIELDS
        }
        # q1 cannot represent q2 numeric fields either; preserve them verbatim.
        receipt.param_version = None
        receipt.time_ms = None
        receipt.group_scale_digest = None
        receipt.duplicate_refs = []
        receipt.weight_snapshot = None
        receipt.param_version = None
        receipt.time_ms = None
        receipt.sidecar = dict(sidecar[receipt.layer])
    write_q1(q1_path, receipts)
    with open(sidecar_path, "w", encoding="utf-8") as handle:
        json.dump({"format": "q3-sidecar", "entries": sidecar},
                  handle, sort_keys=True, indent=2)
    return receipts


def upgrade_q1_to_q3(q1_path: str | Path, sidecar_path: str | Path,
                     q3_path: str | Path) -> list[UnifiedReceipt]:
    receipts = parse_q1(q1_path)
    with open(sidecar_path, "r", encoding="utf-8") as handle:
        sidecar = json.load(handle)
    entries = sidecar.get("entries", {})

    for receipt in receipts:
        if receipt.layer not in entries:
            raise ValueError(
                f"{E_RECEIPT_FORMAT}: missing sidecar entry for {receipt.layer!r}; "
                "refusing to synthesize q3-only defaults"
            )
        saved = entries[receipt.layer]
        receipt.source_version = "q3"
        receipt.missing_fields = []
        for field in _SIDECAR_FIELDS:
            setattr(receipt, field, saved[field])
        receipt.sidecar = {}
    write_q3(q3_path, receipts)
    return parse_q3(q3_path)
