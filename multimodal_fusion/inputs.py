"""Sample discovery and the E_SAMPLE_INVALID validation layer."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import FusionConfig
from .errors import E_SAMPLE_INVALID
from .png import PNGError, decode_png_bytes
from .util import canonical_json, hash_json, sha256_text


@dataclass(frozen=True)
class SampleInput:
    sample_id: str
    image_path: str
    image_hash: str
    width: int
    height: int
    pixels: np.ndarray
    assertions: tuple[dict[str, Any], ...]
    assertions_hash: str
    truth_path: str | None
    truth_hash: str | None


@dataclass
class InputLoadResult:
    samples: list[SampleInput]
    failures: list[dict[str, Any]]
    scan: dict[str, Any] = field(default_factory=dict)


class _FailureCollector:
    def __init__(self) -> None:
        self._details: dict[str, set[str]] = {}

    def add(self, sample_id: str, detail: str) -> None:
        self._details.setdefault(sample_id, set()).add(detail)

    def as_list(self) -> list[dict[str, Any]]:
        failures: list[dict[str, Any]] = []
        for sample_id in sorted(self._details):
            details = sorted(self._details[sample_id])
            failures.append(
                {
                    "sample_id": sample_id,
                    "code": E_SAMPLE_INVALID,
                    "message": "sample failed validation",
                    "details": details,
                }
            )
        return failures

    def invalid_ids(self) -> set[str]:
        return set(self._details)


def _read_jsonl(path: Path, failures: _FailureCollector) -> dict[str, list[dict[str, Any]]]:
    entries: dict[str, list[dict[str, Any]]] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        failures.add("__input__", f"cannot read text JSONL: {exc}")
        return entries

    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        synthetic_id = f"line-{line_number:06d}"
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            failures.add(synthetic_id, f"invalid JSON at line {line_number}: {exc.msg}")
            continue
        if not isinstance(value, dict):
            failures.add(synthetic_id, f"line {line_number} must be a JSON object")
            continue
        sample_id = value.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            failures.add(synthetic_id, f"line {line_number} has empty sample_id")
            continue
        sample_id = sample_id.strip()
        entries.setdefault(sample_id, []).append(value)
    return entries


def _validate_assertions(
    sample_id: str,
    entry: dict[str, Any],
    *,
    strict: bool,
    failures: _FailureCollector,
) -> list[dict[str, Any]]:
    assertions = entry.get("assertions", [])
    if assertions is None:
        assertions = []
    if not isinstance(assertions, list):
        failures.add(sample_id, "assertions must be a list")
        return []
    normalized: list[dict[str, Any]] = []
    for index, assertion in enumerate(assertions):
        if not isinstance(assertion, dict):
            failures.add(sample_id, f"assertion[{index}] must be an object")
            continue
        fields = {name: assertion.get(name) for name in ("entity", "attribute", "value")}
        missing = [name for name, value in fields.items() if not isinstance(value, str) or not value.strip()]
        if missing:
            if strict:
                failures.add(sample_id, f"assertion[{index}] has empty fields: {','.join(sorted(missing))}")
                continue
        normalized.append({name: (value.strip() if isinstance(value, str) else "") for name, value in fields.items()})
    return normalized


def _validate_truth(
    sample_id: str,
    path: Path,
    image_width: int,
    image_height: int,
    *,
    required: bool,
    failures: _FailureCollector,
) -> tuple[str | None, str | None]:
    if not path.exists():
        if required:
            failures.add(sample_id, f"missing truth JSON: {path.name}")
        return None, None
    try:
        raw_bytes = path.read_bytes()
        truth = json.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        failures.add(sample_id, f"invalid truth JSON: {exc}")
        return None, None
    if not isinstance(truth, dict):
        failures.add(sample_id, "truth JSON must be an object")
        return None, None
    if truth.get("sample_id") != sample_id:
        failures.add(sample_id, "truth sample_id mismatch")
    if truth.get("width") != image_width or truth.get("height") != image_height:
        failures.add(sample_id, "truth dimensions do not match image")
    regions = truth.get("regions")
    if not isinstance(regions, list):
        failures.add(sample_id, "truth regions must be a list")
    else:
        for index, region in enumerate(regions):
            if not isinstance(region, dict):
                failures.add(sample_id, f"truth region[{index}] must be an object")
                continue
            box = region.get("bbox")
            if (
                not isinstance(box, list)
                or len(box) != 4
                or any(isinstance(value, bool) or not isinstance(value, int) for value in box)
            ):
                failures.add(sample_id, f"truth region[{index}].bbox must be four integers")
            if not isinstance(region.get("label"), str) or not region["label"].strip():
                failures.add(sample_id, f"truth region[{index}].label must be non-empty")
    return str(path), hashlib.sha256(raw_bytes).hexdigest()


def load_inputs(config: FusionConfig, input_dir: str | Path) -> InputLoadResult:
    input_root = Path(input_dir)
    failures = _FailureCollector()
    if not input_root.is_dir():
        failures.add("__input__", f"input directory does not exist: {input_root}")
        return InputLoadResult(samples=[], failures=failures.as_list())

    image_root = input_root / config.image_dir
    truth_root = input_root / config.truth_dir
    text_path = input_root / config.text_jsonl
    descriptions = _read_jsonl(text_path, failures)

    for sample_id, entries in sorted(descriptions.items()):
        if len(entries) > 1:
            failures.add(sample_id, f"duplicate text descriptions: {len(entries)}")

    image_paths = sorted(
        set(image_root.glob("*.png")) | set(image_root.glob("*.PNG")),
        key=lambda path: path.name,
    ) if image_root.is_dir() else []
    known_ids = {path.stem for path in image_paths}
    for sample_id in sorted(descriptions):
        if sample_id not in known_ids:
            failures.add(sample_id, "missing PNG image")

    decoded: dict[str, tuple[Path, str, Any]] = {}
    for image_path in image_paths:
        sample_id = image_path.stem
        try:
            data = image_path.read_bytes()
        except OSError as exc:
            failures.add(sample_id, f"cannot read PNG: {exc}")
            continue
        try:
            image = decode_png_bytes(data)
        except PNGError as exc:
            failures.add(sample_id, f"invalid PNG: {exc}")
            continue
        decoded[sample_id] = (image_path, sha256_text(data.hex()), image)

    if config.expected_width is not None or config.expected_height is not None:
        expected_width = config.expected_width
        expected_height = config.expected_height
    else:
        first = next((decoded[sid][2] for sid in sorted(decoded)), None)
        expected_width = first.width if first is not None else None
        expected_height = first.height if first is not None else None

    samples: list[SampleInput] = []
    for sample_id in sorted(decoded):
        image_path, image_hash, image = decoded[sample_id]
        width, height = image.width, image.height
        if expected_width is not None and width != expected_width:
            failures.add(sample_id, f"image width {width} != expected {expected_width}")
        if expected_height is not None and height != expected_height:
            failures.add(sample_id, f"image height {height} != expected {expected_height}")
        if width % config.block_size != 0 or height % config.block_size != 0:
            failures.add(
                sample_id,
                f"block size {config.block_size} does not divide {width}x{height}",
            )
        if sample_id not in descriptions:
            failures.add(sample_id, "missing text description")
            continue
        if len(descriptions[sample_id]) != 1:
            continue
        entry = descriptions[sample_id][0]
        assertions = _validate_assertions(
            sample_id,
            entry,
            strict=config.strict_assertions,
            failures=failures,
        )
        truth_path, truth_hash = _validate_truth(
            sample_id,
            truth_root / f"{sample_id}.truth.json",
            width,
            height,
            required=config.truth_required,
            failures=failures,
        )
        assertions_sorted = sorted(
            assertions,
            key=lambda value: (
                str(value.get("entity", "")),
                str(value.get("attribute", "")),
                str(value.get("value", "")),
            ),
        )
        samples.append(
            SampleInput(
                sample_id=sample_id,
                image_path=str(image_path),
                image_hash=image_hash,
                width=width,
                height=height,
                pixels=image.pixels,
                assertions=tuple(assertions_sorted),
                assertions_hash=hash_json(assertions_sorted, namespace="assertions-v1"),
                truth_path=truth_path,
                truth_hash=truth_hash,
            )
        )

    invalid_ids = failures.invalid_ids()
    samples = [sample for sample in samples if sample.sample_id not in invalid_ids]
    samples.sort(key=lambda sample: sample.sample_id)
    scan = {
        "input_root": str(input_root),
        "image_count": len(image_paths),
        "description_count": len(descriptions),
        "valid_sample_count": len(samples),
        "invalid_sample_count": len(invalid_ids),
        "expected_width": expected_width,
        "expected_height": expected_height,
    }
    return InputLoadResult(samples=samples, failures=failures.as_list(), scan=scan)


def input_content_hash(samples: list[SampleInput], config: FusionConfig) -> str:
    payload = {
        "schema": "input-v1",
        "block_size": config.block_size,
        "visual_dim": config.visual_dim,
        "text_dim": config.text_dim,
        "samples": [
            {
                "sample_id": sample.sample_id,
                "image_hash": sample.image_hash,
                "assertions_hash": sample.assertions_hash,
                "truth_hash": sample.truth_hash,
                "width": sample.width,
                "height": sample.height,
            }
            for sample in sorted(samples, key=lambda value: value.sample_id)
        ],
    }
    return hash_json(payload, namespace="input-content-v1")


def assertions_for_report(sample: SampleInput) -> str:
    return canonical_json(list(sample.assertions))
