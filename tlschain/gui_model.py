from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from .canonical import digest_hex
from .errors import DomainError
from .models import Spec
from .session import SessionResult


def _read_ledger_view(ledger_path: str | Path | None, spec: Spec) -> dict[str, Any]:
    if ledger_path is None or not Path(ledger_path).exists():
        return {
            "available": False,
            "format_version": None,
            "content_digest": None,
            "progress_cursor": 0,
            "batches": [],
            "unknown_sequences": [message.seq for message in spec.messages],
        }
    path = Path(ledger_path).resolve()
    uri = f"{path.as_uri()}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        info = connection.execute(
            "SELECT format_version, content_digest FROM ledger_info WHERE id = 1"
        ).fetchone()
        batches = connection.execute(
            """
            SELECT batch_index, through_seq, progress_cursor, spec_digest, payload_digest
            FROM batches ORDER BY batch_index
            """
        ).fetchall()
        cursor = max((int(row["progress_cursor"]) for row in batches), default=0)
        connection.close()
    except sqlite3.DatabaseError as exc:
        raise DomainError("E_REPLAY", f"cannot read ledger for diagnostics: {exc}", field_path="ledger") from exc
    return {
        "available": True,
        "format_version": int(info["format_version"]) if info is not None else None,
        "content_digest": info["content_digest"] if info is not None else None,
        "progress_cursor": cursor,
        "batches": [dict(row) for row in batches],
        "unknown_sequences": [message.seq for message in spec.messages if message.seq > cursor],
    }


def build_diagnostics_model(
    spec: Spec,
    session: SessionResult,
    ledger_path: str | Path | None = None,
) -> dict[str, Any]:
    ledger_view = _read_ledger_view(ledger_path, spec)
    certificates = [
        {
            "id": cert.cert_id,
            "issuer": cert.issuer,
            "serial": cert.serial,
            "not_before_ms": cert.not_before_ms,
            "not_after_ms": cert.not_after_ms,
            "modulus": cert.modulus,
            "algorithm": cert.algorithm,
            "children": sorted(
                child.cert_id
                for child in spec.certificates
                if child.issuer == cert.cert_id and child.cert_id != cert.cert_id
            ),
        }
        for cert in sorted(spec.certificates, key=lambda item: item.cert_id)
    ]
    verification_rows: list[dict[str, Any]] = []
    indeterminate_rows: list[dict[str, Any]] = []
    chain_error_bounds: list[dict[str, Any]] = []
    for message in spec.messages:
        record = session.message_records[message.seq]
        selected = record["selected_chain"]
        chain_error_bounds.append(
            {
                "seq": message.seq,
                "modulus_quantization": selected["error_bound"]["modulus_quantization"],
                "time_window": selected["error_bound"]["time_window"],
                "path_length": selected["error_bound"]["path_length"],
                "total": selected["error_bound"]["total"],
                "pairwise_threshold": record["pairwise_indistinguishability_bound"],
            }
        )
        for step, cert_record in enumerate(selected["certificates"], start=1):
            row = {
                "seq": message.seq,
                "sender": message.sender,
                "message_state": record["decision"],
                "step": step,
                "cert_id": cert_record["cert_id"],
                "verification_cert_id": cert_record["verification_cert_id"],
                "verification_modulus": cert_record["verification_modulus"],
                "signature_q": cert_record["signature_q"],
                "digest_q": cert_record["digest_q"],
                "signature_residual_q": cert_record["signature_residual_q"],
                "event_q": cert_record["event_q"],
                "window_state": cert_record["window_state"],
                "state": cert_record["state"],
                "modulus_error_bound": cert_record["step_error_bound"]["modulus_quantization"],
                "time_error_bound": cert_record["step_error_bound"]["time_window"],
                "path_error_bound": cert_record["step_error_bound"]["path_length"],
                "cumulative_error_bound": cert_record["step_error_bound"]["cumulative"],
                "saturation_applied": cert_record["saturation_applied"],
                "affected_by_error": record["affected_by_error"],
                "selection_reason": record["selection_reason"],
            }
            verification_rows.append(row)
            if record["affected_by_error"] or record["decision"] == "INDETERMINATE":
                indeterminate_rows.append(row)
    return {
        "model_version": 1,
        "spec_digest": digest_hex(spec.spec_digest_input()),
        "cert_source_digest": spec.cert_source_digest,
        "handshake_source_digest": spec.handshake_source_digest,
        "render_order": {
            "certificates": "certificate_id_ascending",
            "verification": "message_seq_ascending_then_chain_step",
            "invalidation": "recorded_order",
            "ledger_batches": "batch_index_ascending",
            "unknown_messages": "message_seq_ascending",
        },
        "certificates": certificates,
        "verification_rows": verification_rows,
        "chain_error_bounds": chain_error_bounds,
        "indeterminate_rows": indeterminate_rows,
        "invalidation_trace": [
            *session.cert_invalidation,
            *session.message_invalidation,
        ],
        "ledger": ledger_view,
        "decision_counts": session.decision_counts(),
        "affected_by_error_count": session.affected_count(),
        "expected_mismatch_seqs": session.expected_mismatch_seqs(),
        "error_decomposition": {
            "modulus_quantization": sum(
                row["modulus_quantization"] for row in chain_error_bounds
            ),
            "time_window": sum(row["time_window"] for row in chain_error_bounds),
            "path_length": sum(row["path_length"] for row in chain_error_bounds),
            "total": sum(row["total"] for row in chain_error_bounds),
        },
        "ledger_content_digest": ledger_view["content_digest"],
        "unknown_message_seqs": ledger_view["unknown_sequences"],
    }


def model_digest(model: dict[str, Any]) -> str:
    return digest_hex(model)
