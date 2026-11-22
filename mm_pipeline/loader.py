"""Sample discovery and load-time validation.

Failures are classified into business codes, appended to a failure list and
never abort the remaining samples. Injection hooks let the diagnostic layer
apply real input faults at a precise (sample, stage) coordinate.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image, UnidentifiedImageError

from .config import PipelineConfig, STAGE_LOAD
from .errors import (
    PipelineError,
    FailureRecord,
    E_IMAGE_FORMAT,
    E_ASSERTION_SCHEMA,
    E_RELATION_REF,
)
from .hashing import hash_bytes, hash_json
from .models import Assertion, Relation, Sample, RELATION_TYPES

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
IMAGE_EXPECTED_SIZE = (256, 256)

# A load hook receives (sample_id, stage, file_path, raw_bytes) and may return
# replacement bytes (or raise) to emulate a real fault.
LoadHook = Callable[[str, str, str, Optional[bytes]], Optional[bytes]]


def discover_sample_ids(sample_root: str | Path) -> list[str]:
    root = Path(sample_root)
    if not root.is_dir():
        return []
    ids = [p.name for p in root.iterdir() if p.is_dir()]
    return sorted(ids)


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _validate_image(sample_id: str, image_path: Path, data: bytes) -> tuple[int, int]:
    if not data.startswith(PNG_SIGNATURE):
        raise PipelineError(
            E_IMAGE_FORMAT,
            "image does not start with the PNG container signature",
            stage=STAGE_LOAD,
            sample=sample_id,
        )
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise PipelineError(
            E_IMAGE_FORMAT,
            f"corrupt PNG container: {exc}",
            stage=STAGE_LOAD,
            sample=sample_id,
        ) from exc
    try:
        with Image.open(io.BytesIO(data)) as img:
            width, height = img.size
            mode = img.mode
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise PipelineError(
            E_IMAGE_FORMAT,
            f"unreadable PNG payload: {exc}",
            stage=STAGE_LOAD,
            sample=sample_id,
        ) from exc
    if (width, height) != IMAGE_EXPECTED_SIZE:
        raise PipelineError(
            E_IMAGE_FORMAT,
            f"image size {width}x{height} != expected {IMAGE_EXPECTED_SIZE[0]}x"
            f"{IMAGE_EXPECTED_SIZE[1]}",
            stage=STAGE_LOAD,
            sample=sample_id,
        )
    if mode != "RGB":
        raise PipelineError(
            E_IMAGE_FORMAT,
            f"image mode {mode} != RGB",
            stage=STAGE_LOAD,
            sample=sample_id,
        )
    return width, height


def _validate_assertions(sample_id: str, raw: Any) -> list[Assertion]:
    if not isinstance(raw, dict):
        raise PipelineError(
            E_ASSERTION_SCHEMA,
            "assertion document root must be an object",
            stage=STAGE_LOAD,
            sample=sample_id,
        )
    if raw.get("sample_id") != sample_id:
        raise PipelineError(
            E_ASSERTION_SCHEMA,
            f"assertion sample_id '{raw.get('sample_id')}' != directory '{sample_id}'",
            stage=STAGE_LOAD,
            sample=sample_id,
        )
    entities = raw.get("entities")
    if not isinstance(entities, list) or not all(isinstance(e, str) for e in entities):
        raise PipelineError(
            E_ASSERTION_SCHEMA,
            "'entities' must be a list of strings",
            stage=STAGE_LOAD,
            sample=sample_id,
        )
    raw_assertions = raw.get("assertions")
    if not isinstance(raw_assertions, list):
        raise PipelineError(
            E_ASSERTION_SCHEMA,
            "'assertions' must be a list",
            stage=STAGE_LOAD,
            sample=sample_id,
        )
    seen_ids: set[str] = set()
    assertions: list[Assertion] = []
    for index, item in enumerate(raw_assertions):
        if not isinstance(item, dict):
            raise PipelineError(
                E_ASSERTION_SCHEMA,
                f"assertion #{index} must be an object",
                stage=STAGE_LOAD,
                sample=sample_id,
            )
        for field_name in ("assertion_id", "entity", "attribute", "value"):
            if field_name not in item:
                raise PipelineError(
                    E_ASSERTION_SCHEMA,
                    f"assertion #{index} missing field '{field_name}'",
                    stage=STAGE_LOAD,
                    sample=sample_id,
                )
        aid = item["assertion_id"]
        entity = item["entity"]
        attribute = item["attribute"]
        value = item["value"]
        if not isinstance(aid, str) or not aid:
            raise PipelineError(
                E_ASSERTION_SCHEMA,
                f"assertion #{index} has invalid assertion_id",
                stage=STAGE_LOAD,
                sample=sample_id,
            )
        if aid in seen_ids:
            raise PipelineError(
                E_ASSERTION_SCHEMA,
                f"duplicate assertion_id '{aid}'",
                stage=STAGE_LOAD,
                sample=sample_id,
            )
        seen_ids.add(aid)
        if not isinstance(entity, str) or entity not in entities:
            raise PipelineError(
                E_ASSERTION_SCHEMA,
                f"assertion '{aid}' references undeclared entity '{entity}'",
                stage=STAGE_LOAD,
                sample=sample_id,
            )
        if not isinstance(attribute, str) or not attribute:
            raise PipelineError(
                E_ASSERTION_SCHEMA,
                f"assertion '{aid}' has invalid attribute",
                stage=STAGE_LOAD,
                sample=sample_id,
            )
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PipelineError(
                E_ASSERTION_SCHEMA,
                f"assertion '{aid}' numeric value must be a number",
                stage=STAGE_LOAD,
                sample=sample_id,
            )
        assertions.append(
            Assertion(
                assertion_id=aid,
                entity=entity,
                attribute=attribute,
                value=float(value),
            )
        )
    return assertions


def _validate_relations(
    sample_id: str,
    raw: Any,
    assertions: list[Assertion],
    region_count: int,
) -> list[Relation]:
    if not isinstance(raw, dict):
        raise PipelineError(
            E_RELATION_REF,
            "relation document root must be an object",
            stage=STAGE_LOAD,
            sample=sample_id,
        )
    if raw.get("sample_id") != sample_id:
        raise PipelineError(
            E_RELATION_REF,
            f"relation sample_id '{raw.get('sample_id')}' != directory '{sample_id}'",
            stage=STAGE_LOAD,
            sample=sample_id,
        )
    raw_relations = raw.get("relations")
    if not isinstance(raw_relations, list):
        raise PipelineError(
            E_RELATION_REF,
            "'relations' must be a list",
            stage=STAGE_LOAD,
            sample=sample_id,
        )
    assertion_ids = {a.assertion_id for a in assertions}
    seen_ids: set[str] = set()
    relations: list[Relation] = []
    for index, item in enumerate(raw_relations):
        if not isinstance(item, dict):
            raise PipelineError(
                E_RELATION_REF,
                f"relation #{index} must be an object",
                stage=STAGE_LOAD,
                sample=sample_id,
            )
        for field_name in ("relation_id", "relation_type", "assertion_id", "region", "weight"):
            if field_name not in item:
                raise PipelineError(
                    E_RELATION_REF,
                    f"relation #{index} missing field '{field_name}'",
                    stage=STAGE_LOAD,
                    sample=sample_id,
                )
        rid = item["relation_id"]
        rtype = item["relation_type"]
        aid = item["assertion_id"]
        region = item["region"]
        weight = item["weight"]
        if not isinstance(rid, str) or rid in seen_ids:
            raise PipelineError(
                E_RELATION_REF,
                f"relation #{index} has missing or duplicate relation_id",
                stage=STAGE_LOAD,
                sample=sample_id,
            )
        seen_ids.add(rid)
        if rtype not in RELATION_TYPES:
            raise PipelineError(
                E_RELATION_REF,
                f"relation '{rid}' has unknown type '{rtype}'",
                stage=STAGE_LOAD,
                sample=sample_id,
            )
        if aid not in assertion_ids:
            raise PipelineError(
                E_RELATION_REF,
                f"relation '{rid}' references nonexistent assertion '{aid}'",
                stage=STAGE_LOAD,
                sample=sample_id,
            )
        if isinstance(region, bool) or not isinstance(region, int) or not (
            0 <= region < region_count
        ):
            raise PipelineError(
                E_RELATION_REF,
                f"relation '{rid}' references region {region} outside [0,"
                f"{region_count - 1}]",
                stage=STAGE_LOAD,
                sample=sample_id,
            )
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise PipelineError(
                E_RELATION_REF,
                f"relation '{rid}' weight must be numeric",
                stage=STAGE_LOAD,
                sample=sample_id,
            )
        relations.append(
            Relation(
                relation_id=rid,
                relation_type=rtype,
                assertion_id=aid,
                region=region,
                weight=float(weight),
            )
        )
    return relations


def load_samples(
    sample_root: str | Path,
    config: PipelineConfig,
    *,
    failures: Optional[list[FailureRecord]] = None,
    load_hook: Optional[LoadHook] = None,
) -> list[Sample]:
    """Scan and validate every sample directory, continuing past failures."""
    failures = failures if failures is not None else []
    samples: list[Sample] = []
    for sample_id in discover_sample_ids(sample_root):
        sample_dir = Path(sample_root) / sample_id
        image_path = sample_dir / "image.png"
        assertions_path = sample_dir / "assertions.json"
        relations_path = sample_dir / "relations.json"
        try:
            for required in (image_path, assertions_path, relations_path):
                if not required.exists():
                    code = (
                        E_IMAGE_FORMAT
                        if required == image_path
                        else E_ASSERTION_SCHEMA
                        if required == assertions_path
                        else E_RELATION_REF
                    )
                    raise PipelineError(
                        code,
                        f"required file missing: {required.name}",
                        stage=STAGE_LOAD,
                        sample=sample_id,
                    )

            image_bytes = _read_bytes(image_path)
            if load_hook is not None:
                hooked = load_hook(sample_id, STAGE_LOAD, str(image_path), image_bytes)
                if hooked is not None:
                    image_bytes = hooked
            width, height = _validate_image(sample_id, image_path, image_bytes)

            assertion_bytes = _read_bytes(assertions_path)
            if load_hook is not None:
                hooked = load_hook(
                    sample_id, STAGE_LOAD, str(assertions_path), assertion_bytes
                )
                if hooked is not None:
                    assertion_bytes = hooked
            try:
                assertion_doc = json.loads(assertion_bytes.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise PipelineError(
                    E_ASSERTION_SCHEMA,
                    f"assertion JSON unparseable: {exc}",
                    stage=STAGE_LOAD,
                    sample=sample_id,
                ) from exc
            assertions = _validate_assertions(sample_id, assertion_doc)

            relation_bytes = _read_bytes(relations_path)
            if load_hook is not None:
                hooked = load_hook(
                    sample_id, STAGE_LOAD, str(relations_path), relation_bytes
                )
                if hooked is not None:
                    relation_bytes = hooked
            try:
                relation_doc = json.loads(relation_bytes.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise PipelineError(
                    E_RELATION_REF,
                    f"relation JSON unparseable: {exc}",
                    stage=STAGE_LOAD,
                    sample=sample_id,
                ) from exc
            relations = _validate_relations(
                sample_id, relation_doc, assertions, config.region_count
            )

            samples.append(
                Sample(
                    sample_id=sample_id,
                    image_path=str(image_path),
                    assertions_path=str(assertions_path),
                    relations_path=str(relations_path),
                    image_bytes=image_bytes,
                    width=width,
                    height=height,
                    assertions=assertions,
                    relations=relations,
                    image_hash=hash_bytes(image_bytes, prefix="image"),
                    assertion_hash=hash_json(
                        [a.to_dict() for a in assertions], prefix="assertions"
                    ),
                    relation_hash=hash_json(
                        [r.to_dict() for r in relations], prefix="relations"
                    ),
                )
            )
        except PipelineError as exc:
            failures.append(
                FailureRecord(
                    sample=exc.sample,
                    region=exc.region,
                    stage=exc.stage,
                    code=exc.code,
                    message=exc.message,
                    injection_id=exc.injection_id,
                )
            )
    return samples
