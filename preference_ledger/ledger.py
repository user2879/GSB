from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from .codes import E_FIELD_TRUNCATED, E_VERSION_UNKNOWN
from .model import DECLARED_QUANT_SCHEMES, DECLARED_VERDICT_NAMES, Issue, VERDICT_NAMES


G1_FIELDS: list[tuple[str, int, str]] = [
    ("version", 2, "left"),
    ("kind", 1, "left"),
    ("scope_id", 28, "left"),
    ("session_id", 16, "left"),
    ("mean_permille", 8, "right"),
    ("verdict_code", 2, "right"),
    ("span", 4, "right_zero"),
    ("turn_range", 9, "left"),
]


def _layout() -> tuple[list[dict[str, Any]], int]:
    specs: list[dict[str, Any]] = []
    start = 0
    for name, width, align in G1_FIELDS:
        specs.append({"name": name, "start": start, "end": start + width, "width": width, "align": align})
        start += width + 1
    return specs, start - 1


G1_SPECS, G1_LINE_WIDTH = _layout()


@dataclass
class LedgerEntry:
    scope_id: str
    kind: str
    session_id: str
    mean_permille: int
    verdict_code: int
    span: int
    turn_refs: tuple[int, ...] = ()
    start_turn: int | None = None
    end_turn: int | None = None
    window_refs: tuple[str, ...] = ()
    float_mean: float | None = None
    verdict_name: str | None = None
    quant_scheme: str | None = None
    detail_summary: dict[str, Any] | None = None
    precision_label: str | None = None
    cache_key: str | None = None
    source_exchange: str = "internal"
    source_generation: str = "L2"
    missing_fields: tuple[str, ...] = ()
    sidecar: dict[str, Any] = field(default_factory=dict)

    def core_key(self) -> tuple[Any, ...]:
        return (
            self.scope_id,
            self.kind,
            self.session_id,
            self.mean_permille,
            self.verdict_code,
            self.span,
            self.turn_refs,
            self.window_refs,
        )

    def to_dict(self) -> dict[str, Any]:
        data = dict(self.__dict__)
        data["turn_refs"] = list(self.turn_refs)
        data["window_refs"] = list(self.window_refs)
        data["missing_fields"] = list(self.missing_fields)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LedgerEntry":
        payload = dict(data)
        payload["turn_refs"] = tuple(payload.get("turn_refs", ()))
        payload["window_refs"] = tuple(payload.get("window_refs", ()))
        payload["missing_fields"] = tuple(payload.get("missing_fields", ()))
        allowed = {name for name in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in payload.items() if key in allowed})


@dataclass
class LedgerBook:
    entries: list[LedgerEntry] = field(default_factory=list)
    generation: str = "L2"
    exchange_version: str = "g2"
    metadata: dict[str, Any] = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)

    def sorted_entries(self) -> list[LedgerEntry]:
        return sorted(self.entries, key=lambda e: (e.session_id, e.kind, e.start_turn or 0, e.scope_id))

    def to_g2_dict(self, generation: str | None = None) -> dict[str, Any]:
        gen = generation or self.generation
        entries = self.sorted_entries()
        if gen == "L1":
            return {
                "schema": "preference-reward-ledger/1",
                "exchange": "g2",
                "generation": "L1",
                "declared_verdicts": sorted(DECLARED_VERDICT_NAMES),
                "declared_quant_schemes": sorted(DECLARED_QUANT_SCHEMES),
                "score_weights": self.metadata.get("score_weights", []),
                "rows": [
                    {
                        "id": e.scope_id,
                        "kind": e.kind[0].upper(),
                        "sid": e.session_id,
                        "pm": e.mean_permille,
                        "code": e.verdict_code,
                        "span": e.span,
                        "turns": list(e.turn_refs),
                        "window_refs": list(e.window_refs),
                    }
                    for e in entries
                ],
                "summary": self.metadata.get("batch_summary", {}),
            }
        return {
            "schema": "preference-reward-ledger/2",
            "exchange": "g2",
            "generation": "L2",
            "declared_verdicts": sorted(DECLARED_VERDICT_NAMES),
            "declared_quant_schemes": sorted(DECLARED_QUANT_SCHEMES),
            "metadata": self.metadata,
            "entries": [e.to_dict() for e in entries],
        }

    def to_g2_json(self, generation: str | None = None) -> str:
        return json.dumps(self.to_g2_dict(generation), ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _fit_field(value: Any, spec: dict[str, Any]) -> str:
    width = spec["width"]
    if spec["align"] == "right_zero":
        text = str(value)
        return text.zfill(width)[-width:]
    text = str(value)
    if len(text) > width:
        raise ValueError(f"field {spec['name']} is too wide: {len(text)}>{width}: {text}")
    if spec["align"] == "right":
        return text.rjust(width)
    return text.ljust(width)


def serialize_g1_entry(entry: LedgerEntry, generation: str = "L2") -> str:
    mean_text = f"{'+' if entry.mean_permille >= 0 else '-'}{abs(entry.mean_permille):07d}"
    range_text = f"{(entry.start_turn or 0):04d}-{(entry.end_turn or 0):04d}"
    values = {
        "version": "G1",
        "kind": "W" if entry.kind == "window" else "P",
        "scope_id": entry.scope_id,
        "session_id": entry.session_id,
        "mean_permille": mean_text,
        "verdict_code": str(entry.verdict_code),
        "span": entry.span,
        "turn_range": range_text,
    }
    return "|".join(_fit_field(values[spec["name"]], spec) for spec in G1_SPECS)


def serialize_g1_book(book: LedgerBook, generation: str | None = None, config_fingerprint: str = "") -> str:
    gen = generation or book.generation
    header = f"# PREF_LEDGER_G1 generation={gen} config_fp={config_fingerprint[:16]}"
    lines = [header]
    for entry in book.sorted_entries():
        lines.append(serialize_g1_entry(entry, gen))
    return "\n".join(lines) + "\n"


def _issue(code: str, target: str, message: str, injection_id: str | None = None) -> Issue:
    return Issue(code, "ledger_parse:g1", target, message, injection_id)


def parse_g1_line(
    line: str,
    line_number: int = 1,
    injection_id: str | None = None,
) -> tuple[LedgerEntry | None, list[Issue]]:
    target = f"g1_line:{line_number}"
    if len(line) >= 2 and line[:2] != "G1":
        return None, [_issue(E_VERSION_UNKNOWN, target, f"unknown g1 version prefix {line[:2]!r}", injection_id)]
    if len(line) < G1_LINE_WIDTH:
        return None, [_issue(E_FIELD_TRUNCATED, target, f"line length {len(line)} < fixed width {G1_LINE_WIDTH}", injection_id)]
    if len(line) > G1_LINE_WIDTH:
        line = line[:G1_LINE_WIDTH]

    for index, spec in enumerate(G1_SPECS[:-1]):
        separator_pos = spec["end"]
        if separator_pos >= len(line) or line[separator_pos] != "|":
            return None, [_issue(E_FIELD_TRUNCATED, target, f"column separator {index + 1} misaligned at {separator_pos}", injection_id)]

    raw: dict[str, str] = {}
    for spec in G1_SPECS:
        raw[spec["name"]] = line[spec["start"]:spec["end"]]

    version = raw["version"]
    if version != "G1":
        return None, [_issue(E_VERSION_UNKNOWN, target, f"unknown version {version!r}", injection_id)]

    kind_text = raw["kind"].strip()
    kind = {"W": "window", "P": "partition"}.get(kind_text)
    if kind is None:
        return None, [_issue(E_FIELD_TRUNCATED, target, f"unknown kind column {kind_text!r}", injection_id)]

    try:
        mean_text = raw["mean_permille"].strip()
        if not mean_text or mean_text[0] not in "+-" or not mean_text[1:].isdigit():
            raise ValueError(raw["mean_permille"])
        mean_permille = int(mean_text)
        verdict_code = int(raw["verdict_code"].strip())
        if verdict_code not in VERDICT_NAMES:
            raise ValueError(raw["verdict_code"])
        span = int(raw["span"])
        if span <= 0:
            raise ValueError(raw["span"])
        range_text = raw["turn_range"].strip()
        start_text, end_text = range_text.split("-", 1)
        start_turn, end_turn = int(start_text), int(end_text)
    except (ValueError, IndexError) as exc:
        return None, [_issue(E_FIELD_TRUNCATED, target, f"field width/alignment parse failure: {exc}", injection_id)]

    scope_id = raw["scope_id"].strip()
    session_id = raw["session_id"].strip()
    if not scope_id or not session_id:
        return None, [_issue(E_FIELD_TRUNCATED, target, "scope_id and session_id must not be blank", injection_id)]

    turn_refs = tuple(range(start_turn, end_turn + 1)) if end_turn >= start_turn else ()
    missing = ("float_mean", "verdict_name", "quant_scheme", "detail_summary", "cache_key")
    entry = LedgerEntry(
        scope_id=scope_id,
        kind=kind,
        session_id=session_id,
        mean_permille=mean_permille,
        verdict_code=verdict_code,
        span=span,
        turn_refs=turn_refs,
        start_turn=start_turn,
        end_turn=end_turn,
        verdict_name=None,
        float_mean=None,
        quant_scheme=None,
        detail_summary=None,
        precision_label="permille_exact:P3",
        source_exchange="g1",
        source_generation="L1",
        missing_fields=missing,
    )
    return entry, []


def parse_g1_text(text: str, injection_id: str | None = None) -> tuple[LedgerBook, list[Issue]]:
    entries: list[LedgerEntry] = []
    issues: list[Issue] = []
    generation = "L1"
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        if line.startswith("#"):
            if "generation=L2" in line:
                generation = "L2"
            continue
        entry, line_issues = parse_g1_line(line, line_number, injection_id)
        if entry is not None:
            entry.source_generation = generation
            entries.append(entry)
        issues.extend(line_issues)
    return LedgerBook(entries=entries, generation=generation, exchange_version="g1", issues=issues), issues


def upgrade_g1_to_g2(book: LedgerBook, sidecar: dict[str, Any] | None = None) -> LedgerBook:
    sidecar = sidecar or {}
    restored: list[LedgerEntry] = []
    for entry in book.entries:
        packet = sidecar.get(entry.scope_id)
        if packet and "overwritten_entry" in packet:
            restored.append(LedgerEntry.from_dict(packet["overwritten_entry"]))
            continue
        upgraded = LedgerEntry.from_dict(entry.to_dict())
        upgraded.source_exchange = "g2"
        upgraded.float_mean = entry.mean_permille / 1000
        upgraded.verdict_name = VERDICT_NAMES[entry.verdict_code]
        upgraded.precision_label = "permille_exact:P3"
        upgraded.missing_fields = tuple(
            field_name for field_name in entry.missing_fields if field_name not in {"float_mean", "verdict_name"}
        )
        if packet:
            upgraded.sidecar = packet
        restored.append(upgraded)
    return LedgerBook(entries=restored, generation=book.generation, exchange_version="g2", metadata=dict(book.metadata))


def downgrade_g2_to_g1(book: LedgerBook) -> tuple[LedgerBook, dict[str, Any]]:
    g1_entries: list[LedgerEntry] = []
    sidecar: dict[str, Any] = {}
    for entry in book.sorted_entries():
        mean_permille = int(Decimal(str(entry.float_mean if entry.float_mean is not None else entry.mean_permille / 1000)) * 1000)
        # Half-up rounding from the canonical decimal representation.
        decimal_mean = Decimal(str(entry.float_mean if entry.float_mean is not None else entry.mean_permille / 1000))
        rounded = int((decimal_mean * 1000).to_integral_value(rounding="ROUND_HALF_UP"))
        residual = decimal_mean * 1000 - Decimal(rounded)
        compact = LedgerEntry(
            scope_id=entry.scope_id,
            kind=entry.kind,
            session_id=entry.session_id,
            mean_permille=rounded,
            verdict_code=entry.verdict_code,
            span=entry.span,
            turn_refs=entry.turn_refs,
            start_turn=entry.start_turn,
            end_turn=entry.end_turn,
            window_refs=entry.window_refs,
            source_exchange="g1",
            source_generation=entry.source_generation,
            precision_label="permille_rounded:P3",
            missing_fields=("float_mean", "verdict_name", "quant_scheme", "detail_summary", "cache_key"),
        )
        g1_entries.append(compact)
        sidecar[entry.scope_id] = {
            "g1_mean_permille": rounded,
            "truncation_residual_permille": str(residual),
            "source_precision_label": entry.precision_label,
            "precision_label": "permille_rounded:P3",
            "overwritten_entry": entry.to_dict(),
        }
    return LedgerBook(entries=g1_entries, generation=book.generation, exchange_version="g1", metadata=dict(book.metadata)), sidecar


def parse_g2_dict(raw: Any, source: str = "g2") -> tuple[LedgerBook | None, list[Issue]]:
    issues: list[Issue] = []
    if not isinstance(raw, dict):
        return None, [Issue("E_FIELD_TRUNCATED", "ledger_parse:g2", source, "g2 document must be an object")]
    exchange = raw.get("exchange", raw.get("format", "g2"))
    schema = str(raw.get("schema", ""))
    if exchange not in ("g2",) and not schema.endswith("/1"):
        if exchange not in ("g1",):
            issues.append(Issue(E_VERSION_UNKNOWN, "ledger_parse:g2", source, f"unknown exchange version {exchange!r}"))
            return None, issues
    generation = raw.get("generation", "L2")
    if generation not in ("L1", "L2"):
        issues.append(Issue(E_VERSION_UNKNOWN, "ledger_parse:g2", source, f"unknown generation {generation!r}"))
        return None, issues

    entries: list[LedgerEntry] = []
    raw_entries = raw.get("entries")
    if raw_entries is None and isinstance(raw.get("rows"), list):
        raw_entries = []
        for row in raw["rows"]:
            kind_code = row.get("kind", "W")
            raw_entries.append(
                {
                    "scope_id": row.get("id"),
                    "kind": "partition" if kind_code == "P" else "window",
                    "session_id": row.get("sid"),
                    "mean_permille": row.get("pm"),
                    "verdict_code": row.get("code"),
                    "span": row.get("span"),
                    "turn_refs": row.get("turns", []),
                    "window_refs": row.get("window_refs", []),
                    "source_exchange": "g2",
                    "source_generation": "L1",
                    "missing_fields": ["float_mean", "verdict_name", "quant_scheme", "detail_summary"],
                }
            )
    if not isinstance(raw_entries, list):
        return None, [Issue(E_FIELD_TRUNCATED, "ledger_parse:g2", source, "entries/rows must be a list")]

    for index, data in enumerate(raw_entries):
        target = f"g2_entry:{index}"
        if not isinstance(data, dict):
            issues.append(Issue(E_FIELD_TRUNCATED, "ledger_parse:g2", target, "entry must be object"))
            continue
        try:
            entry = LedgerEntry.from_dict(data)
        except TypeError as exc:
            issues.append(Issue(E_FIELD_TRUNCATED, "ledger_parse:g2", target, str(exc)))
            continue
        if entry.start_turn is None and entry.turn_refs:
            entry.start_turn = min(entry.turn_refs)
        if entry.end_turn is None and entry.turn_refs:
            entry.end_turn = max(entry.turn_refs)
        if entry.verdict_name is not None and entry.verdict_name not in DECLARED_VERDICT_NAMES:
            issues.append(Issue("E_NAMED_VERDICT_UNDECLARED", "ledger_parse:g2", target, f"verdict {entry.verdict_name!r} not declared"))
        if entry.quant_scheme is not None and entry.quant_scheme not in DECLARED_QUANT_SCHEMES:
            issues.append(Issue("E_NAMED_SCHEME_UNDECLARED", "ledger_parse:g2", target, f"quant scheme {entry.quant_scheme!r} not declared"))
        entries.append(entry)

    if "summary" in raw:
        metadata = {
            "batch_summary": raw.get("summary", {}),
            "score_weights": raw.get("score_weights", []),
        }
    else:
        metadata = raw.get("metadata", {})
    book = LedgerBook(entries=entries, generation=generation, exchange_version="g2", metadata=metadata or {}, issues=issues)
    return book, issues


def parse_g2_text(text: str, source: str = "g2") -> tuple[LedgerBook | None, list[Issue]]:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, [Issue(E_FIELD_TRUNCATED, "ledger_parse:g2", source, str(exc))]
    return parse_g2_dict(raw, source)


def write_roundtrip_pair(book: LedgerBook, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    g1_book, sidecar = downgrade_g2_to_g1(book)
    paths = {
        "g2": out / "ledger.g2.json",
        "g1": out / "ledger.g1.txt",
        "sidecar": out / "ledger.g1.sidecar.json",
    }
    paths["g2"].write_text(book.to_g2_json("L2"), encoding="utf-8")
    paths["g1"].write_text(serialize_g1_book(g1_book, "L1"), encoding="utf-8")
    paths["sidecar"].write_text(json.dumps(sidecar, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return paths
