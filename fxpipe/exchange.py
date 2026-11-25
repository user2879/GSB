"""e1 fixed-width text  <->  e2 JSON exchange layer.

e1 record (exactly 28 ASCII chars, left-padded digit fields):

    chain_id (6) | op_seq (4) | permille (10) | width (8)

e2 is a JSON document carrying a float scale, a named quantization scheme,
per-token summaries and optional downgrade bypass metadata. Both parsers return
:class:`QuantRecord` values so downstream code sees one canonical representation
while the originating exchange version is preserved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .config import (
    ALLOWED_WIDTHS,
    PERMILLE_MAX,
    PERMILLE_MIN,
    QuantSpec,
    SCHEME_TO_WIDTH,
    WIDTH_TO_SCHEME,
)
from .errors import Code, IssueLog

E1_LINE_LEN = 28
E1_WIDTHS = (6, 4, 10, 8)
MISSING_TOKEN = "NA"
SCALE_DIVISOR = 1000.0


@dataclass
class QuantRecord:
    chain_id: int
    seq: int
    quant: QuantSpec
    source_version: str
    summary: dict | None = None
    bypass: dict | None = None

    def node_id(self) -> str:
        return f"c{self.chain_id}op{self.seq}"

    def to_e2_dict(self) -> dict:
        quant = self.quant
        data: dict = {
            "chain_id": self.chain_id,
            "op_seq": self.seq,
            "scale": None if quant.permille is None else quant.permille / SCALE_DIVISOR,
            "scheme": quant.scheme,
            "precision_label": quant.precision_label,
            "completeness": quant.completeness,
            "missing": list(quant.missing),
        }
        if self.summary is not None:
            data["summary"] = self.summary
        if quant.residual:
            data["residual"] = quant.residual
        if self.bypass:
            data["bypass"] = self.bypass
        return data


def _blank_quant(missing: list[str], precision_label: str = "unknown") -> QuantSpec:
    return QuantSpec(
        permille=None,
        width=None,
        scheme=None,
        precision_label=precision_label,
        completeness="missing:" + ",".join(missing) if missing else "complete",
        residual=0.0,
        missing=tuple(missing),
    )


# --------------------------------------------------------------------------- #
# e1
# --------------------------------------------------------------------------- #
class FieldTruncated(Exception):
    """Raised when an e1 field contains illegal characters or is malformed."""


def _parse_e1_field(chunk: str, name: str, location: str, log: IssueLog, *, fatal: bool = False):
    token = chunk.strip()
    if token == MISSING_TOKEN:
        if fatal:
            log.add(Code.E_FIELD_TRUNCATED, location, f"required field {name} is missing")
            raise FieldTruncated(name)
        return None
    if not token or not token.isdigit():
        log.add(
            Code.E_FIELD_TRUNCATED,
            location,
            f"field {name} has illegal/non-numeric characters: {chunk!r}",
        )
        raise FieldTruncated(name)
    return int(token)


def parse_e1_line(line: str, line_no: int, log: IssueLog) -> QuantRecord | None:
    location = f"e1.line[{line_no}]"
    if len(line) != E1_LINE_LEN:
        log.add(
            Code.E_FIELD_TRUNCATED,
            location,
            f"line width {len(line)} != {E1_LINE_LEN}",
            {"line": line},
        )
        return None
    offset = 0
    chunks = []
    for width in E1_WIDTHS:
        chunks.append(line[offset:offset + width])
        offset += width
    try:
        chain_id = _parse_e1_field(chunks[0], "chain_id", location, log, fatal=True)
        seq = _parse_e1_field(chunks[1], "op_seq", location, log, fatal=True)
        permille = _parse_e1_field(chunks[2], "permille", location, log)
        width = _parse_e1_field(chunks[3], "width", location, log)
    except FieldTruncated:
        return None

    missing: list[str] = []
    if permille is None:
        missing.append("permille")
    elif not PERMILLE_MIN <= permille <= PERMILLE_MAX:
        log.add(Code.E_SCHEMA, location, f"permille {permille} out of field range")
        return None
    if width is None:
        missing.append("width")
    elif width not in ALLOWED_WIDTHS:
        log.add(Code.E_SCHEMA, location, f"width {width} not in {ALLOWED_WIDTHS}")
        return None

    if missing:
        quant = _blank_quant(missing)
    else:
        quant = QuantSpec(
            permille=permille,
            width=width,
            scheme=WIDTH_TO_SCHEME[width],
            precision_label="e1-exact",
            completeness="complete",
        )
    return QuantRecord(chain_id=chain_id, seq=seq, quant=quant, source_version="e1")


def parse_e1_file(path: str | Path, log: IssueLog) -> list[QuantRecord]:
    records: list[QuantRecord] = []
    for line_no, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        record = parse_e1_line(raw, line_no, log)
        if record is not None:
            records.append(record)
    return records


def encode_e1_line(record: QuantRecord) -> str:
    quant = record.quant
    chain = str(record.chain_id).rjust(E1_WIDTHS[0], "0")
    seq = str(record.seq).rjust(E1_WIDTHS[1], "0")
    permille = (
        MISSING_TOKEN.rjust(E1_WIDTHS[2])
        if quant.permille is None
        else str(quant.permille).rjust(E1_WIDTHS[2], "0")
    )
    width = (
        MISSING_TOKEN.rjust(E1_WIDTHS[3])
        if quant.width is None
        else str(quant.width).rjust(E1_WIDTHS[3], "0")
    )
    line = chain + seq + permille + width
    assert len(line) == E1_LINE_LEN
    return line


# --------------------------------------------------------------------------- #
# e2
# --------------------------------------------------------------------------- #
def parse_e2_obj(obj: dict, index: int, log: IssueLog, bypass_map: dict | None = None) -> QuantRecord | None:
    location = f"e2.entry[{index}]"
    chain_id = obj.get("chain_id")
    seq = obj.get("op_seq")
    if not isinstance(chain_id, int) or not isinstance(seq, int):
        log.add(Code.E_SCHEMA, location, "chain_id/op_seq must be integers", obj)
        return None

    scale = obj.get("scale")
    scheme = obj.get("scheme")
    missing: list[str] = list(obj.get("missing", []))
    if scale is None:
        missing.append("permille") if "permille" not in missing else None
    if scheme is None:
        missing.append("width") if "width" not in missing else None

    node_id = f"c{chain_id}op{seq}"
    bypass = dict(obj["bypass"]) if isinstance(obj.get("bypass"), dict) else None
    if bypass_map and node_id in bypass_map and bypass is None:
        bypass = dict(bypass_map[node_id])

    width: int | None = None
    if scheme is not None:
        if scheme not in SCHEME_TO_WIDTH:
            log.add(Code.E_SCHEMA, location, f"scheme {scheme!r} not in {list(SCHEME_TO_WIDTH)}", obj)
            return None
        width = SCHEME_TO_WIDTH[scheme]

    permille: int | None = None
    residual = 0.0
    precision_label = obj.get("precision_label")
    if scale is not None:
        if not isinstance(scale, (int, float)):
            log.add(Code.E_SCHEMA, location, f"scale must be numeric, got {scale!r}")
            return None
        if bypass is not None and isinstance(bypass.get("permille"), int):
            permille = int(bypass["permille"])
            residual = float(bypass.get("residual", obj.get("residual", 0.0)))
            if not precision_label:
                precision_label = bypass.get("precision_label", "e2-restored")
        else:
            scaled = float(scale) * SCALE_DIVISOR
            permille = int(scaled + 0.5 if scaled >= 0 else scaled - 0.5)
            residual = round(scaled - permille, 12)
            if not precision_label:
                precision_label = "e2-float"

    if missing:
        quant = _blank_quant(sorted(set(missing)), precision_label or "unknown")
    else:
        quant = QuantSpec(
            permille=permille,
            width=width,
            scheme=scheme,
            precision_label=precision_label or "e2-float",
            completeness="complete",
            residual=residual,
        )
    return QuantRecord(
        chain_id=chain_id,
        seq=seq,
        quant=quant,
        source_version="e2",
        summary=obj.get("summary"),
        bypass=bypass,
    )


def parse_e2_file(path: str | Path, log: IssueLog, bypass_map: dict | None = None) -> list[QuantRecord]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = payload.get("chains", []) if isinstance(payload, dict) else payload
    records: list[QuantRecord] = []
    for index, obj in enumerate(entries):
        record = parse_e2_obj(obj, index, log, bypass_map)
        if record is not None:
            records.append(record)
    return records


# --------------------------------------------------------------------------- #
# Bidirectional conversion with downgrade bypass
# --------------------------------------------------------------------------- #
def e1_to_e2(record: QuantRecord) -> QuantRecord:
    """Promote: exact integer -> exact float plus precision tag/name map."""
    assert record.source_version == "e1"
    quant = record.quant
    if quant.missing:
        promoted = _blank_quant(list(quant.missing), "e1-missing")
    else:
        promoted = QuantSpec(
            permille=quant.permille,
            width=quant.width,
            scheme=WIDTH_TO_SCHEME[quant.width],
            precision_label="e1-exact",
            completeness="complete",
            residual=0.0,
        )
    return QuantRecord(
        chain_id=record.chain_id,
        seq=record.seq,
        quant=promoted,
        source_version="e2",
        summary=record.summary,
    )


def e2_to_e1(record: QuantRecord) -> tuple[QuantRecord, dict]:
    """Downgrade: float -> rounded permille, residual/refs parked in bypass."""
    assert record.source_version == "e2"
    quant = record.quant
    bypass: dict = {
        "permille": quant.permille,
        "residual": quant.residual,
        "precision_label": quant.precision_label,
        "overwritten_refs": [record.node_id()],
        "scale": None if quant.permille is None else quant.permille / SCALE_DIVISOR,
    }
    if quant.missing:
        downgraded = _blank_quant(list(quant.missing), quant.precision_label)
        bypass["missing"] = list(quant.missing)
    else:
        downgraded = QuantSpec(
            permille=quant.permille,
            width=quant.width,
            scheme=quant.scheme,
            precision_label=quant.precision_label,
            completeness="complete",
            residual=0.0,
        )
    return (
        QuantRecord(
            chain_id=record.chain_id,
            seq=record.seq,
            quant=downgraded,
            source_version="e1",
            summary=record.summary,
        ),
        bypass,
    )


def restore_e2_from_bypass(e1_record: QuantRecord, bypass: dict) -> QuantRecord:
    """Re-promote an e1 record while restoring original precision/residual."""
    quant = e1_record.quant
    if quant.missing:
        restored = _blank_quant(list(quant.missing), bypass.get("precision_label", "restored"))
    else:
        restored = QuantSpec(
            permille=int(bypass["permille"]),
            width=quant.width,
            scheme=WIDTH_TO_SCHEME[quant.width],
            precision_label=bypass.get("precision_label", "e2-restored"),
            completeness="complete",
            residual=float(bypass.get("residual", 0.0)),
        )
    return QuantRecord(
        chain_id=e1_record.chain_id,
        seq=e1_record.seq,
        quant=restored,
        source_version="e2",
        summary=e1_record.summary,
        bypass=bypass,
    )


def write_e1(path: str | Path, records: list[QuantRecord]) -> None:
    text = "\n".join(encode_e1_line(record) for record in records) + "\n"
    Path(path).write_text(text, encoding="utf-8")


def write_e2(path: str | Path, records: list[QuantRecord], extra: dict | None = None) -> None:
    payload = {"format": "e2", "chains": [record.to_e2_dict() for record in records]}
    if extra:
        payload.update(extra)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_bypass(path: str | Path, bypass_map: dict) -> None:
    Path(path).write_text(json.dumps(bypass_map, indent=2, sort_keys=True), encoding="utf-8")


def read_bypass(path: str | Path) -> dict:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    return json.loads(file_path.read_text(encoding="utf-8"))
