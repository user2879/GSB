"""v1 fixed-width lines and v2 JSON exchange formats.

Both formats parse into the same internal :class:`ExchangeDocument`.

v1 layout (one fixed-width line per token distribution entry)::

    HDR v1 ...
    DIC REQ <request_id(8)> <seq(6)> <step(4)>
    DIC TOK <token(8)> <name(16)>
    DAT <request_no(6)> <step(4)> <per-mille int(6)> <token code(8)> <flag(1)>

Illegal characters or lines shorter than their declared width raise a
deterministic business code instead of being silently padded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

from ..config import QUANT_TIERS
from ..errors import BizCode, BizError

V1_MAGIC = "DCV1"
V2_MAGIC = "decchain-exchange/v2"

# Explicit fixed column widths (zero-padded numeric fields).
W_REQUEST_NO = 6
W_SEQ = 6
W_STEP = 4
W_MILLE = 6
W_TOKEN = 8
W_FLAG = 1
W_REQ_ID = 8
W_NAME = 16

V1_DAT_WIDTH = W_REQUEST_NO + W_SEQ + W_STEP + W_MILLE + W_TOKEN + W_FLAG
DIGITS = set("0123456789")


@dataclass
class ExchangeRecord:
    request_no: int
    request_id: str
    seq_id: int
    step_index: int
    token: int
    mille_prob: int                    # per-mille probability integer
    token_name: str | None = None
    quant_residual: int = 0
    ref_int: int | None = None
    covered_ref: str | None = None     # overwritten reference marker
    precision_label: str = "W3"

    def sidecar(self) -> dict:
        return {
            "quant_residual": self.quant_residual,
            "ref_int": self.ref_int,
            "covered_ref": self.covered_ref,
            "precision_label": self.precision_label,
            "token_name": self.token_name,
        }


@dataclass
class ExchangeDocument:
    version: str
    width: int
    tier: str
    config_fingerprint: str
    request_index: dict[str, int]       # request_id -> request_no
    token_names: dict[int, str]
    records: list[ExchangeRecord] = field(default_factory=list)
    sidecar: dict[str, dict] = field(default_factory=dict)

    def _key(self, rec: ExchangeRecord) -> str:
        return f"{rec.request_no}:{rec.step_index}:{rec.token}"


# --------------------------------------------------------------------------- #
# v1
# --------------------------------------------------------------------------- #
def _fixed(value: int, width: int, location: str) -> str:
    text = str(value)
    if len(text) > width:
        raise _exc(BizCode.V1_LINE_TOO_SHORT, location,
                   f"value {value} needs width {len(text)} > {width}")
    return text.zfill(width)


def _check_ascii(text: str, location: str) -> None:
    try:
        text.encode("ascii")
    except UnicodeEncodeError:
        raise _exc(BizCode.V1_ILLEGAL_CHAR, location,
                   "non-ascii character in fixed-width v1 line")


def _parse_int_field(raw: str, location: str) -> int:
    if raw == "" or any(ch not in DIGITS for ch in raw):
        raise _exc(BizCode.V1_ILLEGAL_CHAR, location,
                   f"field {raw!r} must be zero-padded digits")
    return int(raw)


def _exc(code: int, location: str, message: str) -> BizError:
    return BizError(code, location, message)


def to_v1(doc: ExchangeDocument) -> str:
    lines = [f"HDR {V1_MAGIC} width={doc.width} tier={doc.tier} "
             f"cfg={doc.config_fingerprint}"]
    for req_id, no in sorted(doc.request_index.items(),
                             key=lambda kv: kv[1]):
        rid = req_id
        if len(rid) > W_REQ_ID:
            rid = rid[:W_REQ_ID]
        lines.append("DIC REQ " + _fixed(no, W_REQUEST_NO, f"req={req_id}")
                     + " " + rid.ljust(W_REQ_ID))
    for token in sorted(doc.token_names):
        name = doc.token_names[token]
        if len(name) > W_NAME:
            name = name[:W_NAME]
        lines.append("DIC TOK " + _fixed(token, W_TOKEN, f"token={token}")
                     + " " + name.ljust(W_NAME))
    for rec in sorted(doc.records,
                      key=lambda r: (r.request_no, r.step_index, r.token)):
        flag = 1 if rec.covered_ref else 0
        line = (_fixed(rec.request_no, W_REQUEST_NO, "dat.request_no")
                + _fixed(rec.seq_id, W_SEQ, "dat.seq")
                + _fixed(rec.step_index, W_STEP, "dat.step")
                + _fixed(rec.mille_prob, W_MILLE, "dat.mille")
                + _fixed(rec.token, W_TOKEN, "dat.token")
                + str(flag))
        lines.append("DAT " + line)
    text = "\n".join(lines) + "\n"
    for idx, line in enumerate(lines, start=1):
        _check_ascii(line, f"v1.line={idx}")
    return text


def parse_v1(text: str) -> ExchangeDocument:
    lines = text.splitlines()
    if not lines or not lines[0].startswith("HDR " + V1_MAGIC):
        raise _exc(BizCode.V1_BAD_HEADER, "v1.header",
                   "missing or invalid v1 header")
    header = lines[0]
    width = _header_int(header, "width", "v1.header")
    tier = _header_str(header, "tier", "v1.header")
    cfg = _header_str(header, "cfg", "v1.header")

    request_index: dict[str, int] = {}
    no_to_id: dict[int, str] = {}
    token_names: dict[int, str] = {}
    records: list[ExchangeRecord] = []

    for lineno, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        loc = f"v1.line={lineno}"
        if line.startswith("DIC REQ "):
            body = line[len("DIC REQ "):]
            if len(body) != W_REQUEST_NO + 1 + W_REQ_ID:
                raise _exc(BizCode.V1_LINE_TOO_SHORT, loc,
                           f"request dict width {len(body)} invalid")
            no = _parse_int_field(body[:W_REQUEST_NO], loc)
            rid = body[W_REQUEST_NO + 1:].strip()
            no_to_id[no] = rid
            request_index[rid] = no
        elif line.startswith("DIC TOK "):
            body = line[len("DIC TOK "):]
            if len(body) != W_TOKEN + 1 + W_NAME:
                raise _exc(BizCode.V1_LINE_TOO_SHORT, loc,
                           f"token dict width {len(body)} invalid")
            token = _parse_int_field(body[:W_TOKEN], loc)
            name = body[W_TOKEN + 1:].strip()
            token_names[token] = name
        elif line.startswith("DAT "):
            body = line[4:]
            if len(body) != V1_DAT_WIDTH:
                raise _exc(BizCode.V1_LINE_TOO_SHORT, loc,
                           f"data line width {len(body)} != {V1_DAT_WIDTH}")
            cursor = 0
            no = _parse_int_field(body[cursor:cursor + W_REQUEST_NO], loc)
            cursor += W_REQUEST_NO
            seq_id = _parse_int_field(body[cursor:cursor + W_SEQ], loc)
            cursor += W_SEQ
            step = _parse_int_field(body[cursor:cursor + W_STEP], loc)
            cursor += W_STEP
            mille = _parse_int_field(body[cursor:cursor + W_MILLE], loc)
            cursor += W_MILLE
            token = _parse_int_field(body[cursor:cursor + W_TOKEN], loc)
            cursor += W_TOKEN
            flag = body[cursor:cursor + W_FLAG]
            if flag not in ("0", "1"):
                raise _exc(BizCode.V1_ILLEGAL_CHAR, loc,
                           f"flag {flag!r} must be 0 or 1")
            rid = no_to_id.get(no)
            if rid is None:
                raise _exc(BizCode.UNKNOWN_REQUEST, loc,
                           f"request_no {no} has no dictionary entry")
            records.append(ExchangeRecord(
                request_no=no, request_id=rid, seq_id=seq_id, step_index=step,
                token=token, mille_prob=mille,
                token_name=token_names.get(token),
                covered_ref="v1-cover" if flag == "1" else None,
                precision_label=f"W{width}"))
        else:
            raise _exc(BizCode.V1_ILLEGAL_CHAR, loc,
                       f"unknown record tag {line[:7]!r}")

    return ExchangeDocument(
        version=V1_MAGIC, width=width, tier=tier, config_fingerprint=cfg,
        request_index=request_index, token_names=token_names, records=records)


def _header_int(header: str, key: str, loc: str) -> int:
    raw = _header_str(header, key, loc)
    if not raw or any(ch not in DIGITS for ch in raw):
        raise _exc(BizCode.V1_BAD_HEADER, loc, f"header {key} not int")
    return int(raw)


def _header_str(header: str, key: str, loc: str) -> str:
    token = f"{key}="
    for part in header.split():
        if part.startswith(token):
            return part[len(token):]
    raise _exc(BizCode.V1_BAD_HEADER, loc, f"missing header key {key}")


# --------------------------------------------------------------------------- #
# v2
# --------------------------------------------------------------------------- #
def to_v2(doc: ExchangeDocument) -> str:
    payload = {
        "magic": V2_MAGIC,
        "width": doc.width,
        "tier": doc.tier,
        "config_fingerprint": doc.config_fingerprint,
        "request_index": doc.request_index,
        "token_names": {str(k): v for k, v in doc.token_names.items()},
        "records": [
            {
                "request_no": rec.request_no,
                "request_id": rec.request_id,
                "seq_id": rec.seq_id,
                "step_index": rec.step_index,
                "token": rec.token,
                "token_name": rec.token_name,
                "probability": rec.mille_prob / 1000.0,
                "quant_scheme": f"FIXED_W{doc.width}_{doc.tier}",
            }
            for rec in doc.records
        ],
        "sidecar": {
            f"{rec.request_no}:{rec.step_index}:{rec.token}": rec.sidecar()
            for rec in doc.records
        },
    }
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)


def parse_v2(text: str) -> ExchangeDocument:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _exc(BizCode.V2_BAD_JSON, "v2.body", str(exc))
    if payload.get("magic") != V2_MAGIC:
        raise _exc(BizCode.V2_UNKNOWN_VERSION, "v2.magic",
                   f"magic={payload.get('magic')!r}")
    tier = payload["tier"]
    if tier not in QUANT_TIERS:
        raise _exc(BizCode.INVALID_TIER, "v2.tier", f"tier={tier!r}")
    width = payload["width"]
    token_names = {int(k): v for k, v in payload.get("token_names", {}).items()}
    request_index = payload["request_index"]
    sidecar = payload.get("sidecar", {})
    records: list[ExchangeRecord] = []

    for idx, item in enumerate(payload.get("records", [])):
        loc = f"v2.record={idx}"
        prob = item.get("probability")
        if not isinstance(prob, (int, float)) or isinstance(prob, bool) \
                or not (0.0 <= float(prob) <= 1.0):
            raise _exc(BizCode.V2_BAD_PROBABILITY, loc,
                       f"probability={prob!r} outside [0,1]")
        token = item.get("token")
        if not isinstance(token, int) or isinstance(token, bool) or token < 0:
            raise _exc(BizCode.V2_BAD_TOKEN, loc, f"token={token!r}")
        mille = round(float(prob) * 1000)
        rec = ExchangeRecord(
            request_no=item["request_no"],
            request_id=item["request_id"],
            seq_id=item.get("seq_id", 0),
            step_index=item["step_index"],
            token=token,
            mille_prob=mille,
            token_name=item.get("token_name") or token_names.get(token),
            precision_label=f"W{width}",
        )
        meta = sidecar.get(f"{rec.request_no}:{rec.step_index}:{rec.token}")
        if meta:
            rec.quant_residual = meta.get("quant_residual", 0)
            rec.ref_int = meta.get("ref_int")
            rec.covered_ref = meta.get("covered_ref")
            rec.precision_label = meta.get("precision_label", rec.precision_label)
            if meta.get("token_name"):
                rec.token_name = meta["token_name"]
        records.append(rec)

    return ExchangeDocument(
        version=V2_MAGIC, width=width, tier=tier,
        config_fingerprint=payload["config_fingerprint"],
        request_index=request_index, token_names=token_names,
        records=records, sidecar=sidecar)


def convert(doc: ExchangeDocument, target: str) -> ExchangeDocument:
    """Identity-internal conversion; only the rendered text changes."""
    if target not in ("v1", "v2"):
        raise _exc(BizCode.V2_UNKNOWN_VERSION, "convert.target",
                   f"target={target!r}")
    # Re-attach sidecar entries from records so a round trip restores them.
    doc.sidecar = {
        doc._key(rec): rec.sidecar() for rec in doc.records
    }
    return doc


def render(doc: ExchangeDocument, version: str) -> str:
    convert(doc, version)
    if version == "v1":
        return to_v1(doc)
    return to_v2(doc)


def parse(text: str, version: str) -> ExchangeDocument:
    if version == "v1":
        return parse_v1(text)
    if version == "v2":
        return parse_v2(text)
    raise _exc(BizCode.V2_UNKNOWN_VERSION, "parse.version",
               f"version={version!r}")


def documents_equal(a: ExchangeDocument, b: ExchangeDocument) -> bool:
    """Field-wise equivalence used by round-trip tests."""
    if (a.version != b.version or a.width != b.width or a.tier != b.tier
            or a.config_fingerprint != b.config_fingerprint
            or a.request_index != b.request_index
            or a.token_names != b.token_names
            or len(a.records) != len(b.records)):
        return False
    ka = sorted(a.records, key=lambda r: (r.request_no, r.step_index, r.token))
    kb = sorted(b.records, key=lambda r: (r.request_no, r.step_index, r.token))
    for ra, rb in zip(ka, kb):
        if asdict(ra) != asdict(rb):
            return False
    return True


# --------------------------------------------------------------------------- #
# v1 sidecar file (residuals / wider reference cannot fit fixed-width fields)
# --------------------------------------------------------------------------- #
def write_sidecar(doc: "ExchangeDocument", path: str) -> None:
    import json as _json
    payload = {
        "config_fingerprint": doc.config_fingerprint,
        "entries": {
            doc._key(rec): rec.sidecar() for rec in doc.records
        },
    }
    with open(path, "w", encoding="utf-8") as handle:
        _json.dump(payload, handle, sort_keys=True, indent=2,
                   ensure_ascii=False)


def load_sidecar(doc: "ExchangeDocument", path: str) -> "ExchangeDocument":
    import json as _json
    import os
    if not os.path.exists(path):
        raise _exc(BizCode.SIDECAR_MISSING, path, "v1 sidecar file not found")
    with open(path, "r", encoding="utf-8") as handle:
        payload = _json.load(handle)
    entries = payload.get("entries", {})
    for rec in doc.records:
        meta = entries.get(doc._key(rec))
        if not meta:
            continue
        rec.quant_residual = meta.get("quant_residual", 0)
        rec.ref_int = meta.get("ref_int")
        rec.covered_ref = meta.get("covered_ref")
        rec.precision_label = meta.get("precision_label", rec.precision_label)
        if meta.get("token_name"):
            rec.token_name = meta["token_name"]
    doc.sidecar = entries
    return doc
