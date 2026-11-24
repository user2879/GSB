"""Load and strongly validate the JSON declaration bundle.

Validation order is fixed (document order) and the first violation is
raised as :class:`SourceDeclarationError` (``E_SOURCE``) carrying the
offending source/candidate identifier and field name.
"""

from __future__ import annotations

import json
import os
from typing import Any

from .errors import SourceDeclarationError
from .models import (
    DeclarationBundle,
    CandidateRule,
    SourceDeclaration,
    SUPPORTED_CALIBERS,
    SUPPORTED_SOURCE_TYPES,
    SUPPORTED_VERSIONS,
)

_ALLOWED_ALGEBRA_KEYS = frozenset(
    {"percentile_rank", "trim_ratio", "min_samples", "weight", "window_ms"}
)


def load_declaration(path: str) -> DeclarationBundle:
    path = os.path.abspath(path)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        raise SourceDeclarationError(
            "declaration file not found", ref=path, field="path"
        )
    except json.JSONDecodeError as exc:
        raise SourceDeclarationError(
            f"declaration is not valid JSON: {exc.msg}", ref=path,
            field=f"line={exc.lineno}",
        )

    base_dir = os.path.dirname(path)
    _require_mapping(payload, "<root>", "<root>")
    schema_version = _nonneg_int(payload, "schema_version", "<root>")
    if schema_version != 1:
        raise SourceDeclarationError(
            f"unsupported declaration schema_version {schema_version}",
            ref="<root>", field="schema_version",
        )

    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise SourceDeclarationError(
            "sources must be a non-empty list", ref="<root>", field="sources"
        )
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list):
        raise SourceDeclarationError(
            "candidates must be a list", ref="<root>", field="candidates"
        )

    sources: list[SourceDeclaration] = []
    seen_source_ids: set[str] = set()
    # Pass 1: identity/type/version/integer validation and uniqueness.
    # Frame-file existence is deferred to pass 2 so duplicate identifiers
    # are always reported before filesystem errors.
    for index, raw in enumerate(raw_sources):
        sources.append(_parse_source(raw, index, seen_source_ids, base_dir,
                                     check_file=False))
    for source in sources:
        if not os.path.isfile(source.frame_file):
            raise SourceDeclarationError(
                f"frame_file not found: {source.frame_file}",
                ref=source.source_id, field="frame_file",
            )

    candidates: list[CandidateRule] = []
    seen_candidate_ids: set[str] = set()
    known_ids = {src.source_id for src in sources}
    for index, raw in enumerate(raw_candidates):
        candidates.append(
            _parse_candidate(raw, index, seen_candidate_ids, known_ids)
        )

    return DeclarationBundle(
        declaration_path=path,
        schema_version=schema_version,
        sources=tuple(sources),
        candidates=tuple(candidates),
    )


def _parse_source(raw: Any, index: int, seen: set[str], base_dir: str,
                  check_file: bool = True) -> SourceDeclaration:
    ident = f"sources[{index}]"
    _require_mapping(raw, ident, "<source>")
    source_id = _require_str(raw, "source_id", ident)
    if source_id in seen:
        raise SourceDeclarationError(
            f"duplicate source_id {source_id!r}", ref=source_id,
            field="source_id",
        )

    source_type = _require_str(raw, "source_type", source_id)
    if source_type not in SUPPORTED_SOURCE_TYPES:
        raise SourceDeclarationError(
            f"unsupported source_type {source_type!r}; supported="
            f"{list(SUPPORTED_SOURCE_TYPES)}",
            ref=source_id, field="source_type",
        )

    format_version = _nonneg_int(raw, "format_version", source_id)
    if format_version not in SUPPORTED_VERSIONS[source_type]:
        raise SourceDeclarationError(
            f"unsupported format_version {format_version} for {source_type}; "
            f"supported={list(SUPPORTED_VERSIONS[source_type])}",
            ref=source_id, field="format_version",
        )

    sample_rows = _nonneg_int(raw, "sample_rows", source_id)
    sample_period_ms = _nonneg_int(raw, "sample_period_ms", source_id)

    latency_caliber = _require_str(raw, "latency_caliber", source_id)
    if latency_caliber not in SUPPORTED_CALIBERS:
        raise SourceDeclarationError(
            f"unsupported latency_caliber {latency_caliber!r}; supported="
            f"{list(SUPPORTED_CALIBERS)}",
            ref=source_id, field="latency_caliber",
        )

    algebra = raw.get("latency_algebra", {})
    if not isinstance(algebra, dict):
        raise SourceDeclarationError(
            "latency_algebra must be an object",
            ref=source_id, field="latency_algebra",
        )
    for key, value in algebra.items():
        if not isinstance(key, str) or key not in _ALLOWED_ALGEBRA_KEYS:
            raise SourceDeclarationError(
                f"unknown latency_algebra key {key!r}",
                ref=source_id, field=f"latency_algebra.{key}",
            )
        if not _is_nonneg_int(value):
            raise SourceDeclarationError(
                "latency_algebra values must be non-negative integers",
                ref=source_id, field=f"latency_algebra.{key}",
            )

    seen.add(source_id)
    frame_file = _require_str(raw, "frame_file", source_id)
    resolved = frame_file if os.path.isabs(frame_file) else os.path.join(base_dir, frame_file)
    if check_file and not os.path.isfile(resolved):
        raise SourceDeclarationError(
            f"frame_file not found: {frame_file}",
            ref=source_id, field="frame_file",
        )
    return SourceDeclaration(
        source_id=source_id,
        source_type=source_type,
        format_version=format_version,
        sample_rows=sample_rows,
        latency_caliber=latency_caliber,
        sample_period_ms=sample_period_ms,
        latency_algebra=dict(algebra),
        frame_file=os.path.abspath(resolved),
    )


def _parse_candidate(raw: Any, index: int, seen: set[str],
                     known_source_ids: set[str]) -> CandidateRule:
    ident = f"candidates[{index}]"
    _require_mapping(raw, ident, "<candidate>")
    candidate_id = _require_str(raw, "candidate_id", ident)
    if candidate_id in seen:
        raise SourceDeclarationError(
            f"duplicate candidate_id {candidate_id!r}",
            ref=candidate_id, field="candidate_id",
        )

    original_query = _require_str(raw, "original_query", candidate_id)
    rewritten_sql = _require_str(raw, "rewritten_sql", candidate_id)
    expected_gain_bps = _nonneg_int(raw, "expected_gain_bps", candidate_id)
    if expected_gain_bps > 10000:
        raise SourceDeclarationError(
            "expected_gain_bps must be within [0, 10000]",
            ref=candidate_id, field="expected_gain_bps",
        )

    target_sources = raw.get("target_sources")
    if not isinstance(target_sources, list) or not target_sources:
        raise SourceDeclarationError(
            "target_sources must be a non-empty list",
            ref=candidate_id, field="target_sources",
        )
    resolved_targets: list[str] = []
    for pos, target in enumerate(target_sources):
        if not isinstance(target, str) or not target:
            raise SourceDeclarationError(
                "target_sources entries must be non-empty strings",
                ref=candidate_id, field=f"target_sources[{pos}]",
            )
        if target not in known_source_ids:
            raise SourceDeclarationError(
                f"candidate references unknown source {target!r}",
                ref=candidate_id, field=f"target_sources[{pos}]",
            )
        if target not in resolved_targets:
            resolved_targets.append(target)

    seen.add(candidate_id)
    return CandidateRule(
        candidate_id=candidate_id,
        original_query=original_query,
        rewritten_sql=rewritten_sql,
        target_sources=tuple(resolved_targets),
        expected_gain_bps=expected_gain_bps,
    )


def _require_mapping(raw: Any, ref: str, field: str) -> None:
    if not isinstance(raw, dict):
        raise SourceDeclarationError(
            "expected a JSON object", ref=ref, field=field
        )


def _require_str(raw: dict, key: str, ref: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise SourceDeclarationError(
            f"field {key!r} must be a non-empty string", ref=ref, field=key
        )
    return value


def _nonneg_int(raw: dict, key: str, ref: str) -> int:
    if key not in raw:
        raise SourceDeclarationError(
            f"missing required integer field {key!r}", ref=ref, field=key
        )
    value = raw[key]
    if not _is_nonneg_int(value):
        raise SourceDeclarationError(
            f"field {key!r} must be a non-negative integer", ref=ref, field=key
        )
    return int(value)


def _is_nonneg_int(value: Any) -> bool:
    # bool is a subclass of int; explicitly reject it.
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0
