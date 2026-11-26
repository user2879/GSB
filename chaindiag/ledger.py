from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .canonical import sha256_hex


class CacheStore:
    """Content-addressed stage records and stable node-name index."""

    def __init__(self, records: dict[str, dict[str, Any]] | None = None,
                 node_index: dict[str, dict[str, Any]] | None = None):
        self.records: dict[str, dict[str, Any]] = records or {}
        self.node_index: dict[str, dict[str, Any]] = node_index or {}

    @classmethod
    def load(cls, path: str | Path) -> "CacheStore":
        path = Path(path)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data.get("records", {}), data.get("node_index", {}))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps({"records": self.records, "node_index": self.node_index}, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp.replace(path)

    def get(self, key: str) -> dict[str, Any] | None:
        return self.records.get(key)

    def put(self, key: str, record: dict[str, Any]) -> None:
        self.records[key] = record

    def set_node(self, node_id: str, key: str, payload: dict[str, Any]) -> None:
        self.node_index[node_id] = {"cache_key": key, "payload": payload}


def operator_fingerprint(op: dict[str, Any]) -> str:
    payload = {
        "schema": "chaindiag-op-v1",
        "seq": op.get("seq"),
        "kind": op.get("kind"),
        "name": op.get("name"),
        "inputs": op.get("inputs", []),
        "scale_rational": op.get("scale_rational") or f"{op.get('scale_permille', 1000)}/1000",
        "int_bits": op.get("int_bits", 16),
        "quant_scheme": op.get("quant_scheme"),
        "params": op.get("params", {}),
        "attention_snapshot": op.get("attention_snapshot"),
    }
    return sha256_hex(payload)


def token_fingerprint(uid: str, token: int) -> str:
    return sha256_hex({"uid": uid, "token": token})


def classify_payload_change(old: dict[str, Any] | None, new: dict[str, Any]) -> list[str]:
    if not old:
        return ["initial_run"]
    reasons: list[str] = []
    old_identity = old.get("identity", {})
    new_identity = new.get("identity", {})
    if old_identity.get("token_fingerprint") != new_identity.get("token_fingerprint"):
        reasons.append("token-content-changed")
    if old.get("operator_fingerprint") != new.get("operator_fingerprint"):
        reasons.append("operator-parameter-changed")
    old_fp = old.get("fixed_point", {})
    new_fp = new.get("fixed_point", {})
    if old_fp.get("quant_tier") != new_fp.get("quant_tier"):
        reasons.append("quant-tier-changed")
    if old_fp.get("frac_bits") != new_fp.get("frac_bits"):
        reasons.append("frac-bits-changed")
    if old.get("exchange_version") != new.get("exchange_version"):
        reasons.append("exchange-version-changed")
    old_up = old.get("upstream")
    new_up = new.get("upstream")
    if old_up != new_up:
        # Give a more specific reason when segment membership is the only
        # changed structured section.
        old_segment = (old_up or {}).get("segment") if isinstance(old_up, dict) else None
        new_segment = (new_up or {}).get("segment") if isinstance(new_up, dict) else None
        if old_segment != new_segment and old_segment is not None and new_segment is not None:
            reasons.append("segment-boundary-changed")
        else:
            reasons.append("upstream-content-changed")
    if not reasons and old.get("stage") != new.get("stage"):
        reasons.append("stage-definition-changed")
    if not reasons and old.get("cache_key") != new.get("cache_key"):
        reasons.append("content-fingerprint-mismatch")
    return reasons or ["content-fingerprint-mismatch"]
