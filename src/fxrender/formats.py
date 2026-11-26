"""v1 fixed-width lines and v2 JSONL frame records with lossless round trip.

Internal representation (FrameRecord) is version agnostic. v1 -> v2 expands
permille guidance to an exact float and maps the integer noise code to a
named scheme plus a precision label. v2 -> v1 rounds guidance back to
permille and records every overwritten field, residual and precision label
in sidecar metadata so a later upgrade can restore the original v2 exactly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .codes import BizCode

V1_MAGIC = "FXF1"
V2_MAGIC = "FXF2"

NOISE_CODE_TO_SCHEME = {0: "uniform", 1: "gaussian", 2: "laplacian"}
NOISE_SCHEME_TO_CODE = {v: k for k, v in NOISE_CODE_TO_SCHEME.items()}

# (field name, column width). Widths are part of the v1 contract.
V1_COLUMNS = (
    ("magic", 4),
    ("frame_id", 6),
    ("step", 5),
    ("tile_row", 4),
    ("guidance_permille", 7),
    ("noise_code", 4),
    ("latent_fp", 220),
    ("step_delta_fp", 220),
    ("pixel_digest", 32),
    ("step_cost", 6),
    ("merged", 5),
    ("cache_key", 40),
    ("noise_scale_fp", 14),
    ("condition_scale_fp", 14),
)
V1_LINE_WIDTH = sum(w for _, w in V1_COLUMNS)
MISSING_TOKEN = "*"


@dataclass
class FrameRecord:
    frame_id: int
    step: int
    tile_row: int
    guidance_permille: int | None = None
    guidance_float: float | None = None
    noise_code: int | None = None
    noise_scheme: str | None = None
    latent_fp: list[int] = field(default_factory=list)
    step_delta_fp: list[int] = field(default_factory=list)
    pixel_digest: str | None = None
    step_cost: int | None = None
    merged: bool | None = None
    cache_key: str | None = None
    noise_scale_fp: int | None = None
    condition_scale_fp: int | None = None
    source_version: int | None = None
    missing: list[str] = field(default_factory=list)

    def precision_label(self) -> str:
        if self.guidance_float is None:
            return "permille-only"
        residual = abs(self.guidance_float - self.guidance_float * 1000 / 1000)
        # exact when the float is representable at 1e-3 granularity
        scaled = self.guidance_float * 1000.0
        if abs(scaled - round(scaled)) < 1e-12:
            return "exact-permille"
        return "sub-permille"

    def canonical(self) -> dict:
        data = asdict(self)
        data["latent_fp"] = list(self.latent_fp)
        data["step_delta_fp"] = list(self.step_delta_fp)
        data["missing"] = sorted(self.missing)
        return data

    @classmethod
    def from_canonical(cls, data: dict) -> "FrameRecord":
        fields = cls.__dataclass_fields__  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in fields})


def _fmt_int_list(values: list[int], width: int) -> str:
    text = ";".join(str(v) for v in values)
    if len(text) > width:
        raise ValueError(f"{BizCode.E_PARSE.value}: latent column overflow")
    return text.ljust(width)


def _parse_int_list(text: str) -> list[int]:
    text = text.strip()
    if not text:
        return []
    return [int(part) for part in text.split(";") if part != ""]


def record_to_v1_line(record: FrameRecord) -> str:
    if record.guidance_permille is None:
        guidance = MISSING_TOKEN
    else:
        guidance = str(record.guidance_permille)
    noise = MISSING_TOKEN if record.noise_code is None else str(record.noise_code)
    values = {
        "magic": V1_MAGIC,
        "frame_id": str(record.frame_id),
        "step": str(record.step),
        "tile_row": str(record.tile_row),
        "guidance_permille": guidance,
        "noise_code": noise,
        "latent_fp": _fmt_int_list(record.latent_fp, 220),
        "step_delta_fp": _fmt_int_list(record.step_delta_fp, 220),
        "pixel_digest": record.pixel_digest or MISSING_TOKEN,
        "step_cost": MISSING_TOKEN if record.step_cost is None else str(record.step_cost),
        "merged": MISSING_TOKEN if record.merged is None else ("T" if record.merged else "F"),
        "cache_key": record.cache_key or MISSING_TOKEN,
        "noise_scale_fp": MISSING_TOKEN if record.noise_scale_fp is None
        else str(record.noise_scale_fp),
        "condition_scale_fp": MISSING_TOKEN if record.condition_scale_fp is None
        else str(record.condition_scale_fp),
    }
    parts = []
    for name, width in V1_COLUMNS:
        cell = values[name]
        if cell == MISSING_TOKEN:
            parts.append(MISSING_TOKEN.ljust(width))
        elif len(cell) > width:
            raise ValueError(f"{BizCode.E_PARSE.value}: column {name} overflow")
        else:
            parts.append(cell.ljust(width))
    line = "".join(parts)
    if len(line) != V1_LINE_WIDTH:  # pragma: no cover - defensive
        raise AssertionError("v1 width contract violated")
    return line


def parse_v1_line(line: str) -> FrameRecord:
    if len(line.rstrip("\n")) != V1_LINE_WIDTH:
        raise ValueError(
            f"{BizCode.E_PARSE.value}: v1 line width {len(line.rstrip(chr(10)))}")
    cells: dict[str, str] = {}
    cursor = 0
    for name, width in V1_COLUMNS:
        cells[name] = line[cursor:cursor + width]
        cursor += width
    if cells["magic"].strip() != V1_MAGIC:
        raise ValueError(f"{BizCode.E_PARSE.value}: bad v1 magic")

    def scalar(name: str, cast):
        raw = cells[name].strip()
        if raw == MISSING_TOKEN or raw == "":
            return None
        return cast(raw)

    record = FrameRecord(
        frame_id=int(cells["frame_id"].strip()),
        step=int(cells["step"].strip()),
        tile_row=int(cells["tile_row"].strip()),
        guidance_permille=scalar("guidance_permille", int),
        noise_code=scalar("noise_code", int),
        latent_fp=_parse_int_list(cells["latent_fp"]),
        step_delta_fp=_parse_int_list(cells["step_delta_fp"]),
        pixel_digest=scalar("pixel_digest", str),
        step_cost=scalar("step_cost", int),
        merged=({"T": True, "F": False}.get(cells["merged"].strip())
                if cells["merged"].strip() != MISSING_TOKEN else None),
        cache_key=scalar("cache_key", str),
        noise_scale_fp=scalar("noise_scale_fp", int),
        condition_scale_fp=scalar("condition_scale_fp", int),
        source_version=1,
    )
    if record.guidance_permille is not None:
        record.guidance_float = record.guidance_permille / 1000.0
    if record.noise_code is not None:
        scheme = NOISE_CODE_TO_SCHEME.get(record.noise_code)
        if scheme is None:
            raise ValueError(
                f"{BizCode.E_PARSE.value}: noise_code={record.noise_code}")
        record.noise_scheme = scheme
    for name in ("guidance_permille", "noise_code", "pixel_digest", "step_cost",
                 "merged", "cache_key", "noise_scale_fp", "condition_scale_fp"):
        if getattr(record, name) is None:
            record.missing.append(name)
    return record


def v1_to_v2(record: FrameRecord) -> FrameRecord:
    upgraded = FrameRecord.from_canonical(record.canonical())
    upgraded.source_version = 2
    if upgraded.guidance_permille is not None and upgraded.guidance_float is None:
        upgraded.guidance_float = upgraded.guidance_permille / 1000.0
    if upgraded.noise_code is not None and upgraded.noise_scheme is None:
        upgraded.noise_scheme = NOISE_CODE_TO_SCHEME[upgraded.noise_code]
    upgraded.missing = sorted(set(upgraded.missing))
    return upgraded


def record_to_v2_dict(record: FrameRecord, precision: str | None = None,
                      sidecar: dict | None = None) -> dict:
    data = {
        "magic": V2_MAGIC,
        "frame_id": record.frame_id,
        "step": record.step,
        "tile_row": record.tile_row,
        "guidance_float": record.guidance_float,
        "noise_scheme": record.noise_scheme,
        "latent_fp": list(record.latent_fp),
        "step_delta_fp": list(record.step_delta_fp),
        "pixel_digest": record.pixel_digest,
        "step_cost": record.step_cost,
        "merged": record.merged,
        "cache_key": record.cache_key,
        "noise_scale_fp": record.noise_scale_fp,
        "condition_scale_fp": record.condition_scale_fp,
        "source_version": record.source_version or 2,
        "missing": sorted(record.missing),
        "precision_label": precision or record.precision_label(),
    }
    if sidecar is not None:
        data["sidecar"] = sidecar
    return data


def parse_v2_dict(data: dict) -> FrameRecord:
    if data.get("magic") != V2_MAGIC:
        raise ValueError(f"{BizCode.E_PARSE.value}: bad v2 magic")
    scheme = data.get("noise_scheme")
    code = NOISE_SCHEME_TO_CODE.get(scheme) if scheme is not None else None
    record = FrameRecord(
        frame_id=int(data["frame_id"]),
        step=int(data["step"]),
        tile_row=int(data["tile_row"]),
        guidance_float=(None if data.get("guidance_float") is None
                        else float(data["guidance_float"])),
        noise_scheme=scheme,
        noise_code=code,
        latent_fp=[int(v) for v in data.get("latent_fp", [])],
        step_delta_fp=[int(v) for v in data.get("step_delta_fp", [])],
        pixel_digest=data.get("pixel_digest"),
        step_cost=data.get("step_cost"),
        merged=data.get("merged"),
        cache_key=data.get("cache_key"),
        noise_scale_fp=data.get("noise_scale_fp"),
        condition_scale_fp=data.get("condition_scale_fp"),
        source_version=2,
        missing=sorted(data.get("missing", [])),
    )
    if record.guidance_float is not None and record.guidance_permille is None:
        record.guidance_permille = int(round(record.guidance_float * 1000.0))
    return record


def v2_to_v1(record: FrameRecord) -> tuple[FrameRecord, str, dict]:
    """Downgrade to v1 and return (v1 record, v1 line, sidecar metadata)."""
    sidecar: dict = {
        "overwritten_fields": [],
        "residuals": {},
        "precision_label": record.precision_label(),
        "source_guidance_float": record.guidance_float,
        "source_noise_scheme": record.noise_scheme,
        "source_missing": sorted(record.missing),
    }
    downgraded = FrameRecord.from_canonical({**record.canonical(), "source_version": 1})
    if record.guidance_float is not None:
        permille = int(round(record.guidance_float * 1000.0))
        residual = record.guidance_float - permille / 1000.0
        downgraded.guidance_permille = permille
        downgraded.guidance_float = permille / 1000.0
        sidecar["residuals"]["guidance"] = residual
        if abs(residual) > 0:
            sidecar["overwritten_fields"].append("guidance_float")
    if record.noise_scheme is not None:
        downgraded.noise_code = NOISE_SCHEME_TO_CODE[record.noise_scheme]
        sidecar["overwritten_fields"].append("noise_scheme")
    line = record_to_v1_line(downgraded)
    sidecar["overwritten_fields"] = sorted(set(sidecar["overwritten_fields"]))
    return downgraded, line, sidecar


def restore_v2_from_sidecar(v1_record: FrameRecord, sidecar: dict) -> FrameRecord:
    restored = v1_to_v2(v1_record)
    if sidecar.get("source_guidance_float") is not None:
        restored.guidance_float = float(sidecar["source_guidance_float"])
        restored.guidance_permille = int(
            round(restored.guidance_float * 1000.0))
    if sidecar.get("source_noise_scheme") is not None:
        restored.noise_scheme = sidecar["source_noise_scheme"]
        restored.noise_code = NOISE_SCHEME_TO_CODE[restored.noise_scheme]
    restored.missing = sorted(sidecar.get("source_missing", restored.missing))
    restored.source_version = 2
    return restored


def write_v1(path: str | Path, records: list[FrameRecord]) -> None:
    lines = [record_to_v1_line(r) for r in records]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_v1(path: str | Path) -> list[FrameRecord]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(parse_v1_line(line))
    return out


def write_v2(path: str | Path, records: list[FrameRecord],
             sidecars: dict[int, dict] | None = None) -> None:
    lines = []
    for record in records:
        sidecar = (sidecars or {}).get(record.frame_id)
        lines.append(json.dumps(
            record_to_v2_dict(record, sidecar=sidecar),
            sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_v2(path: str | Path) -> list[FrameRecord]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(parse_v2_dict(json.loads(line)))
    return out
