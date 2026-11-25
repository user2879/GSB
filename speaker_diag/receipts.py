"""Decode and normalize historical r1/r2/r3 receipts.

Normalization is deterministic: the same input bytes, context and format yield
the same canonical bytes on every call.  Missing historical fields are listed
explicitly together with the rule that filled them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .canonical import canonical_bytes, canonical_sha256, sha256_hex
from .errors import DiagnosticError
from .jsonpos import PositionedJSON, parse_positioned_json
from .model import ParameterCache
from .spec import WorkloadSpec


_HEX64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ReceiptContext:
    batch_id: str
    worker_id: str
    param_version: int
    param_digest: str
    parameter_snapshot: Mapping[str, Any]
    batch_window_order: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "worker_id": self.worker_id,
            "param_version": self.param_version,
            "param_digest": self.param_digest,
            "parameter_snapshot": dict(self.parameter_snapshot),
            "batch_window_order": list(self.batch_window_order),
        }


@dataclass(frozen=True)
class NormalizedReceipt:
    data: Mapping[str, Any]
    canonical: bytes
    sha256: str
    attribution_digest: str
    source_format: str
    filled_fields: tuple[str, ...]

    @property
    def batch_id(self) -> str:
        return str(self.data["batch_id"])

    @property
    def worker_id(self) -> str:
        return str(self.data["worker_id"])

    @property
    def param_version(self) -> int:
        return int(self.data["param_version"])

    @property
    def windows(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.data["windows"])

    def to_dict(self) -> dict[str, Any]:
        return dict(self.data)


def context_for(spec: WorkloadSpec, batch_id: str, cache: ParameterCache | None = None) -> ReceiptContext:
    request = spec.request_by_batch(batch_id)
    worker = spec.worker_by_id(request.worker_id)
    parameters = (cache or ParameterCache()).get(worker, spec.audio.speaker_count)
    return ReceiptContext(
        batch_id=batch_id,
        worker_id=worker.worker_id,
        param_version=worker.parameter_version,
        param_digest=parameters.digest,
        parameter_snapshot=parameters.snapshot,
        batch_window_order=request.window_ids,
    )


def _format_of(receipt: Mapping[str, Any]) -> tuple[str, int]:
    raw = receipt.get("format", receipt.get("receipt_format", receipt.get("format_version")))
    if raw is None:
        if any(
            key in receipt
            for key in ("parameter_digest", "parameter_snapshot", "participant_window_ids")
        ) or any(
            isinstance(window, Mapping)
            and any(key in window for key in ("rank", "duplicate_of"))
            for window in receipt.get("windows", [])
        ):
            return ("r3", 3)
        if any(key in receipt for key in ("worker_id", "param_version", "elapsed_ms")):
            return ("r2", 2)
        if "batch_id" in receipt and "windows" in receipt:
            return ("r1", 1)
    if isinstance(raw, str):
        lowered = raw.lower().strip()
        aliases = {
            "r1": ("r1", 1),
            "receipt-r1": ("r1", 1),
            "receipt_r1": ("r1", 1),
            "r2": ("r2", 2),
            "receipt-r2": ("r2", 2),
            "receipt_r2": ("r2", 2),
            "r3": ("r3", 3),
            "receipt-r3": ("r3", 3),
            "receipt_r3": ("r3", 3),
        }
        if lowered in aliases:
            return aliases[lowered]
    if raw in (1, 2, 3) and type(raw) is int:
        return (f"r{raw}", int(raw))
    raise DiagnosticError(
        "E_FIELD",
        "receipt format must identify r1, r2, or r3",
        field="format",
    )


class _Decoder:
    def __init__(self, receipt: Mapping[str, Any], positions: PositionedJSON | None = None):
        self.receipt = receipt
        self.positions = positions

    def line(self, path: tuple[Any, ...] | None) -> int | None:
        if not self.positions or path is None:
            return None
        return self.positions.line(path, default=1)

    def fail(self, code: str, field: str, message: str, *, path: tuple[Any, ...] | None = None, **details: Any) -> None:
        raise DiagnosticError(code, message, field=field, line=self.line(path), details=details)

    def integer(self, value: Any, field: str, path: tuple[Any, ...], *, minimum: int | None = None) -> int:
        if type(value) is not int:
            self.fail("E_FIELD", field, "expected an integer", path=path)
        if minimum is not None and value < minimum:
            self.fail("E_FIELD", field, f"must be >= {minimum}", path=path, value=value)
        return value

    def decode(self) -> tuple[str, int, dict[str, Any]]:
        source_format, format_version = _format_of(self.receipt)
        batch_id = self.receipt.get("batch_id")
        if not isinstance(batch_id, str) or not batch_id:
            self.fail("E_FIELD", "batch_id", "batch_id must be a non-empty string", path=("batch_id",))
        raw_windows = self.receipt.get("windows")
        if not isinstance(raw_windows, list) or not raw_windows:
            self.fail("E_FIELD", "windows", "windows must be a non-empty array", path=("windows",))
        windows: list[dict[str, Any]] = []
        seen_window_ids: set[str] = set()
        for index, raw_window in enumerate(raw_windows):
            path = ("windows", index)
            if not isinstance(raw_window, dict):
                self.fail("E_FIELD", f"windows[{index}]", "window entry must be an object", path=path)
            window_id = raw_window.get("window_id")
            if not isinstance(window_id, str) or not window_id:
                self.fail("E_FIELD", f"windows[{index}].window_id", "window_id must be a non-empty string", path=path + ("window_id",))
            if window_id in seen_window_ids:
                self.fail("E_FIELD", f"windows[{index}].window_id", "window_id must be unique in a receipt", path=path + ("window_id",))
            seen_window_ids.add(window_id)
            attribution = raw_window.get("attribution", raw_window.get("speaker"))
            if not isinstance(attribution, str) or not attribution:
                self.fail(
                    "E_FIELD",
                    f"windows[{index}].attribution",
                    "attribution must be a non-empty string",
                    path=path + ("attribution",),
                )
            window = {
                "window_id": window_id,
                "attribution": attribution,
                "rank": raw_window.get("rank"),
                "duplicate_of": raw_window.get("duplicate_of"),
                "scores": raw_window.get("scores"),
                "confidence_q12": raw_window.get("confidence_q12"),
                "score_margin": raw_window.get("score_margin"),
                "tie_break_used": raw_window.get("tie_break_used"),
            }
            if source_format == "r3" and "duplicate_of" not in raw_window:
                self.fail(
                    "E_FIELD",
                    f"windows[{index}].duplicate_of",
                    "r3 requires the duplicate delivery reference field (use null when unused)",
                    path=path + ("duplicate_of",),
                )
            windows.append(window)

        worker_id = self.receipt.get("worker_id")
        param_version = self.receipt.get("param_version")
        elapsed_ms = self.receipt.get("elapsed_ms", 0 if source_format == "r1" else None)
        if source_format in ("r2", "r3"):
            if not isinstance(worker_id, str) or not worker_id:
                self.fail("E_FIELD", "worker_id", "worker_id must be a non-empty string", path=("worker_id",))
            param_version = self.integer(param_version, "param_version", ("param_version",), minimum=1)
            elapsed_ms = self.integer(elapsed_ms, "elapsed_ms", ("elapsed_ms",), minimum=0)
        elif "elapsed_ms" in self.receipt:
            elapsed_ms = self.integer(elapsed_ms, "elapsed_ms", ("elapsed_ms",), minimum=0)
        else:
            elapsed_ms = 0

        param_digest = self.receipt.get("parameter_digest", self.receipt.get("param_digest"))
        parameter_snapshot = self.receipt.get("parameter_snapshot")
        participant_window_ids = self.receipt.get("participant_window_ids")
        if source_format == "r3":
            if not isinstance(param_digest, str) or not _HEX64.fullmatch(param_digest):
                self.fail("E_FIELD", "parameter_digest", "parameter_digest must be a lowercase SHA-256 hex string", path=("parameter_digest",))
            if not isinstance(parameter_snapshot, dict) or not parameter_snapshot:
                self.fail("E_FIELD", "parameter_snapshot", "r3 requires a non-empty parameter snapshot", path=("parameter_snapshot",))
            snapshot_digest = parameter_snapshot.get("digest")
            if snapshot_digest is not None and snapshot_digest != param_digest:
                self.fail(
                    "E_FIELD",
                    "parameter_snapshot.digest must equal parameter_digest",
                    path=("parameter_snapshot", "digest"),
                )
            if not isinstance(participant_window_ids, list) or not participant_window_ids:
                self.fail("E_FIELD", "participant_window_ids", "r3 requires the participant window id set", path=("participant_window_ids",))
            participant_ids = []
            seen_participants: set[str] = set()
            for index, participant in enumerate(participant_window_ids):
                if not isinstance(participant, str) or not participant:
                    self.fail("E_FIELD", f"participant_window_ids[{index}]", "participant id must be a non-empty string", path=("participant_window_ids", index))
                if participant in seen_participants:
                    self.fail("E_FIELD", f"participant_window_ids[{index}]", "participant ids must be unique", path=("participant_window_ids", index))
                seen_participants.add(participant)
                participant_ids.append(participant)
            receipt_ids = [window["window_id"] for window in windows]
            unknown_receipt_ids = sorted(set(receipt_ids) - set(participant_ids))
            if unknown_receipt_ids:
                self.fail(
                    "E_REF",
                    "participant_window_ids",
                    "receipt window is not in participant_window_ids",
                    path=("participant_window_ids",),
                    window_id=unknown_receipt_ids[0],
                )
            missing_receipt_ids = sorted(set(participant_ids) - set(receipt_ids))
            if missing_receipt_ids:
                self.fail(
                    "E_FIELD",
                    "participant_window_ids",
                    "participant window has no receipt entry",
                    path=("participant_window_ids",),
                    window_id=missing_receipt_ids[0],
                )
            ranks: list[int] = []
            for index, window in enumerate(windows):
                rank = self.integer(window["rank"], f"windows[{index}].rank", ("windows", index, "rank"), minimum=1)
                if rank > len(windows):
                    self.fail(
                        "E_FIELD",
                        f"windows[{index}].rank",
                        "rank is out of the batch range",
                        path=("windows", index, "rank"),
                        value=rank,
                        batch_size=len(windows),
                    )
                ranks.append(rank)
            if len(set(ranks)) != len(ranks):
                self.fail("E_FIELD", "windows[].rank", "r3 ranks must be unique", path=("windows",))
            for index, window in enumerate(windows):
                reference = window["duplicate_of"]
                if reference is not None:
                    if not isinstance(reference, str):
                        self.fail(
                            "E_REF",
                            f"windows[{index}].duplicate_of",
                            "duplicate_of must be null or a window id",
                            path=("windows", index, "duplicate_of"),
                        )
                    if reference not in seen_window_ids:
                        self.fail(
                            "E_REF",
                            f"windows[{index}].duplicate_of",
                            "duplicate_of must reference an existing window in this receipt",
                            path=("windows", index, "duplicate_of"),
                            reference=reference,
                        )
                    if reference == window["window_id"]:
                        self.fail(
                            "E_REF",
                            f"windows[{index}].duplicate_of",
                            "duplicate_of must not reference itself",
                            path=("windows", index, "duplicate_of"),
                        )
        else:
            participant_ids = None
            if participant_window_ids is not None:
                if not isinstance(participant_window_ids, list):
                    self.fail("E_FIELD", "participant_window_ids", "participant_window_ids must be an array", path=("participant_window_ids",))
                participant_ids = participant_window_ids

        return source_format, format_version, {
            "batch_id": batch_id,
            "worker_id": worker_id,
            "param_version": param_version,
            "elapsed_ms": elapsed_ms,
            "param_digest": param_digest,
            "parameter_snapshot": parameter_snapshot,
            "participant_window_ids": participant_ids,
            "windows": windows,
        }


def _fill_default_snapshot(digest: str) -> dict[str, Any]:
    return {
        "digest": digest,
        "scheme": "unavailable",
        "fractional_bits": 0,
        "basis": "historical field absent and no normalization context supplied",
    }


def _canonical_attribution_data(data: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "batch_id": data["batch_id"],
        "worker_id": data["worker_id"],
        "param_version": data["param_version"],
        "param_digest": data["param_digest"],
        "parameter_snapshot": data["parameter_snapshot"],
        "participant_window_ids": data["participant_window_ids"],
        "windows": [
            {
                "window_id": window["window_id"],
                "attribution": window["attribution"],
                "rank": window["rank"],
                "duplicate_of": window["duplicate_of"],
            }
            for window in sorted(data["windows"], key=lambda item: item["rank"])
        ],
    }


def normalize_receipt(
    receipt: Mapping[str, Any],
    *,
    context: ReceiptContext | None = None,
    positions: PositionedJSON | None = None,
) -> NormalizedReceipt:
    if not isinstance(receipt, Mapping):
        raise DiagnosticError("E_FIELD", "receipt must be a JSON object", field="$")
    decoder = _Decoder(receipt, positions)
    source_format, format_version, decoded = decoder.decode()
    filled: set[str] = set()
    basis: dict[str, str] = {}

    if context is not None and decoded["batch_id"] != context.batch_id:
        raise DiagnosticError(
            "E_FIELD",
            "receipt batch_id does not match normalization context",
            field="batch_id",
            details={"receipt": decoded["batch_id"], "context": context.batch_id},
        )

    if decoded["worker_id"] is None:
        if context is not None:
            decoded["worker_id"] = context.worker_id
            basis["worker_id"] = "context.request.worker_id (absent in r1)"
        else:
            decoded["worker_id"] = "UNKNOWN"
            basis["worker_id"] = "standalone sentinel; no context supplied"
        filled.add("worker_id")
    elif context is not None and decoded["worker_id"] != context.worker_id:
        raise DiagnosticError(
            "E_FIELD",
            "receipt worker_id does not match normalization context",
            field="worker_id",
            details={"receipt": decoded["worker_id"], "context": context.worker_id},
        )

    if decoded["param_version"] is None:
        if context is not None:
            decoded["param_version"] = context.param_version
            basis["param_version"] = "context.worker.parameter_version (absent in r1)"
        else:
            decoded["param_version"] = 0
            basis["param_version"] = "standalone sentinel 0; no context supplied"
        filled.add("param_version")
    elif context is not None and decoded["param_version"] != context.param_version:
        raise DiagnosticError(
            "E_FIELD",
            "receipt parameter version does not match context",
            field="param_version",
            details={"receipt": decoded["param_version"], "context": context.param_version},
        )

    if decoded["param_digest"] is None:
        if context is not None:
            decoded["param_digest"] = context.param_digest
            basis["param_digest"] = "context.parameter_snapshot.digest (absent in r1/r2)"
        else:
            decoded["param_digest"] = sha256_hex(b"speaker-diag:missing-param-digest")
            basis["param_digest"] = "standalone deterministic sentinel; no context supplied"
        filled.add("param_digest")
    elif context is not None and decoded["param_digest"] != context.param_digest:
        raise DiagnosticError(
            "E_FIELD",
            "receipt parameter digest does not match context",
            field="param_digest",
            details={"receipt": decoded["param_digest"], "context": context.param_digest},
        )

    if decoded["parameter_snapshot"] is None:
        if context is not None:
            decoded["parameter_snapshot"] = dict(context.parameter_snapshot)
            basis["parameter_snapshot"] = "context.parameter_snapshot (absent in r1/r2)"
        else:
            decoded["parameter_snapshot"] = _fill_default_snapshot(decoded["param_digest"])
            basis["parameter_snapshot"] = "standalone explicit absence marker; no context supplied"
        filled.add("parameter_snapshot")

    if source_format == "r1":
        basis.setdefault("elapsed_ms", "r1 schema has no elapsed field; normalized to integer 0")
        filled.add("elapsed_ms")

    if decoded["participant_window_ids"] is None or source_format != "r3":
        if context is not None:
            participant_ids = list(context.batch_window_order)
            basis["participant_window_ids"] = "context.request.window_ids order (absent before r3)"
        else:
            participant_ids = [window["window_id"] for window in decoded["windows"]]
            basis["participant_window_ids"] = "receipt window order (absent before r3)"
        filled.add("participant_window_ids")
    else:
        participant_ids = list(decoded["participant_window_ids"])

    receipt_set = {window["window_id"] for window in decoded["windows"]}
    participant_set = set(participant_ids)
    extra = sorted(receipt_set - participant_set)
    missing = sorted(participant_set - receipt_set)
    if extra:
        raise DiagnosticError(
            "E_REF",
            "receipt contains a window outside the authoritative participant set",
            field="windows",
            details={"window_id": extra[0]},
        )
    if missing:
        raise DiagnosticError(
            "E_FIELD",
            "authoritative participant set has no receipt window",
            field="participant_window_ids",
            details={"window_id": missing[0]},
        )

    if source_format != "r3":
        order = {window_id: index + 1 for index, window_id in enumerate(participant_ids)}
        for window in decoded["windows"]:
            window["rank"] = order[window["window_id"]]
        basis["windows[].rank"] = "context.request.window_ids order" if context is not None else "receipt window order"
        filled.add("windows[].rank")

    for window in decoded["windows"]:
        if window["rank"] is None:
            raise DiagnosticError("E_FIELD", "window rank is missing", field="windows[].rank")
        if "duplicate_of" not in window or (source_format != "r3" and window["duplicate_of"] is None):
            window["duplicate_of"] = None
            filled.add("windows[].duplicate_of")
            basis["windows[].duplicate_of"] = "null for pre-r3 formats"

    sorted_windows = sorted(decoded["windows"], key=lambda item: item["rank"])
    normalized_data = {
        "schema": "speaker-diag-normalized-receipt-v1",
        "source_format": source_format,
        "source_format_version": format_version,
        "batch_id": decoded["batch_id"],
        "worker_id": decoded["worker_id"],
        "param_version": decoded["param_version"],
        "elapsed_ms": decoded["elapsed_ms"],
        "param_digest": decoded["param_digest"],
        "parameter_snapshot": decoded["parameter_snapshot"],
        "participant_window_ids": participant_ids,
        "windows": [
            {
                "window_id": window["window_id"],
                "attribution": window["attribution"],
                "rank": window["rank"],
                "duplicate_of": window["duplicate_of"],
                "scores": window["scores"],
                "confidence_q12": window["confidence_q12"],
                "score_margin": window["score_margin"],
                "tie_break_used": window["tie_break_used"],
            }
            for window in sorted_windows
        ],
        "filled_fields": sorted(filled),
        "fill_basis": {key: basis[key] for key in sorted(basis)},
    }
    encoded = canonical_bytes(normalized_data)
    attribution_digest = canonical_sha256(_canonical_attribution_data(normalized_data))
    return NormalizedReceipt(
        data=normalized_data,
        canonical=encoded,
        sha256=sha256_hex(encoded),
        attribution_digest=attribution_digest,
        source_format=source_format,
        filled_fields=tuple(normalized_data["filled_fields"]),
    )


def decode_receipt_text(
    text: str,
    *,
    context: ReceiptContext | None = None,
    source_name: str = "<memory>",
) -> NormalizedReceipt:
    try:
        positioned = parse_positioned_json(text)
    except Exception as exc:
        raise DiagnosticError(
            "E_FIELD",
            f"invalid receipt JSON: {exc}",
            field="$",
            line=getattr(exc, "lineno", 1),
            details={"source": source_name},
        ) from exc
    if not isinstance(positioned.value, dict):
        raise DiagnosticError("E_FIELD", "receipt must be a JSON object", field="$", line=1)
    return normalize_receipt(positioned.value, context=context, positions=positioned)


def normalize_receipt_file(path: str | Path, *, context: ReceiptContext | None = None) -> NormalizedReceipt:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise DiagnosticError("E_FIELD", f"cannot read receipt: {exc}", field="file") from exc
    return decode_receipt_text(text, context=context, source_name=str(source))


def normalized_receipts(
    receipts: Mapping[str, Mapping[str, Any]],
    *,
    contexts: Mapping[str, ReceiptContext] | None = None,
) -> dict[str, NormalizedReceipt]:
    return {
        batch_id: normalize_receipt(receipt, context=(contexts or {}).get(batch_id))
        for batch_id, receipt in receipts.items()
    }


def strip_to_r2(normalized_or_raw: Mapping[str, Any]) -> dict[str, Any]:
    receipt = normalized_or_raw
    return {
        "receipt_format": "r2",
        "format_version": 2,
        "batch_id": receipt["batch_id"],
        "worker_id": receipt["worker_id"],
        "param_version": receipt["param_version"],
        "elapsed_ms": receipt.get("elapsed_ms", 0),
        "windows": [
            {"window_id": window["window_id"], "attribution": window["attribution"]}
            for window in receipt["windows"]
        ],
    }


def strip_to_r1(normalized_or_raw: Mapping[str, Any]) -> dict[str, Any]:
    receipt = normalized_or_raw
    return {
        "receipt_format": "r1",
        "format_version": 1,
        "batch_id": receipt["batch_id"],
        "windows": [
            {"window_id": window["window_id"], "attribution": window["attribution"]}
            for window in receipt["windows"]
        ],
    }
