"""Parsers for r1/r2/r3 receipt rows and unified normalization."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import (
    E_RECEIPT_MALFORMED,
    E_UNKNOWN_PARAM_VERSION,
    Issue,
)
from .faults import FaultInjector
from .model import MISSING, AppConfig, Receipt, missing
from .util import is_int, is_finite_number, safe_relpath


_VERSION_FIELDS = {
    "r1": ("batch_id", "request_id", "output_digest"),
    "r2": (
        "batch_id",
        "request_id",
        "output_digest",
        "param_version",
        "duration_ms",
        "worker_id",
    ),
    "r3": (
        "batch_id",
        "request_id",
        "output_digest",
        "param_version",
        "duration_ms",
        "worker_id",
        "rank",
        "covered_refs",
        "param_digest",
    ),
}

_OPTIONAL_MISSING = {
    "r1": (
        "param_version",
        "duration_ms",
        "worker_id",
        "rank",
        "covered_refs",
        "param_digest",
    ),
    "r2": ("rank", "covered_refs", "param_digest"),
    "r3": (),
}

_ALLOWED_KEYS = {
    "batch_id",
    "request_id",
    "output_digest",
    "param_version",
    "duration_ms",
    "worker_id",
    "rank",
    "covered_refs",
    "param_digest",
    "missing_fields",
}

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_FILE_VERSION_RE = re.compile(r"(?:^|[._-])r([123])(?:[._-]|$)", re.IGNORECASE)


@dataclass
class ParseOutcome:
    receipt: Receipt | None = None
    issues: list[Issue] = field(default_factory=list)


@dataclass
class ReceiptLoadResult:
    receipts: tuple[Receipt, ...]
    issues: list[Issue]
    file_fingerprints: dict[str, str]
    files: tuple[str, ...]

    @property
    def failures(self) -> list[dict[str, Any]]:
        return [issue.to_dict() for issue in sorted(self.issues, key=Issue.sort_key)]


def _malformed(
    message: str,
    *,
    path: str,
    line: int,
    injection_ids: tuple[str, ...] = (),
    item: str = "",
    detail_code: str = "",
) -> Issue:
    return Issue(
        code=E_RECEIPT_MALFORMED,
        message=message,
        path=path,
        line=line,
        item=item,
        injection_id=",".join(injection_ids),
        detail_code=detail_code,
    )


def detect_version(path: str | Path, declared: str | None = None) -> str | None:
    if declared in _VERSION_FIELDS:
        return declared
    match = _FILE_VERSION_RE.search(Path(path).name)
    if match:
        return "r" + match.group(1)
    return None


def _parse_missing_marker(value: Any, field_name: str) -> bool:
    return (
        isinstance(value, dict)
        and value.get("__missing__") is not None
        and (
            value.get("__missing__") in {True, field_name, "unknown"}
            or value.get("field") == field_name
        )
    )


def _validate_id(
    value: Any,
    *,
    field_name: str,
    maximum_width: int,
    path: str,
    line: int,
    injection_ids: tuple[str, ...],
) -> tuple[str | None, Issue | None]:
    if not isinstance(value, str) or not value:
        return None, _malformed(
            f"{field_name} must be a non-empty string",
            path=path,
            line=line,
            item=field_name,
            injection_ids=injection_ids,
        )
    if len(value) > maximum_width:
        return None, _malformed(
            f"{field_name} width {len(value)} exceeds {maximum_width}",
            path=path,
            line=line,
            item=field_name,
            injection_ids=injection_ids,
            detail_code="FIELD_WIDTH",
        )
    if "/" in value or "\\" in value or any(character.isspace() for character in value):
        return None, _malformed(
            f"{field_name} contains a forbidden character",
            path=path,
            line=line,
            item=field_name,
            injection_ids=injection_ids,
        )
    return value, None


def _validate_digest(
    value: Any,
    *,
    field_name: str,
    path: str,
    line: int,
    injection_ids: tuple[str, ...],
) -> tuple[str | None, Issue | None]:
    if not isinstance(value, str):
        return None, _malformed(
            f"{field_name} must be a string",
            path=path,
            line=line,
            item=field_name,
            injection_ids=injection_ids,
        )
    if len(value) != 71:
        return None, _malformed(
            f"{field_name} width must be 71",
            path=path,
            line=line,
            item=field_name,
            injection_ids=injection_ids,
            detail_code="FIELD_WIDTH",
        )
    if not _DIGEST_RE.fullmatch(value):
        return None, _malformed(
            f"{field_name} must match sha256:<64 lowercase hex>",
            path=path,
            line=line,
            item=field_name,
            injection_ids=injection_ids,
        )
    return value, None


def parse_receipt_object(
    obj: Any,
    *,
    version: str,
    source_path: str = "",
    line: int = 0,
    default_worker_id: str | None = None,
    default_param_version: int | None = None,
    injection_ids: tuple[str, ...] = (),
    allow_explicit_missing: bool = False,
) -> ParseOutcome:
    issues: list[Issue] = []
    if version not in _VERSION_FIELDS:
        issues.append(
            _malformed(
                f"unknown receipt version {version!r}",
                path=source_path,
                line=line,
                injection_ids=injection_ids,
            )
        )
        return ParseOutcome(None, issues)
    if not isinstance(obj, dict):
        issues.append(
            _malformed(
                "receipt row must be a JSON object",
                path=source_path,
                line=line,
                injection_ids=injection_ids,
            )
        )
        return ParseOutcome(None, issues)

    extra_keys = sorted(set(obj) - _ALLOWED_KEYS)
    if extra_keys:
        issues.append(
            _malformed(
                f"unexpected fields: {', '.join(extra_keys)}",
                path=source_path,
                line=line,
                item=",".join(extra_keys),
                injection_ids=injection_ids,
            )
        )

    declared_missing = obj.get("missing_fields", [])
    if not isinstance(declared_missing, list) or not all(
        isinstance(value, str) for value in declared_missing
    ):
        issues.append(
            _malformed(
                "missing_fields must be a list of strings",
                path=source_path,
                line=line,
                item="missing_fields",
                injection_ids=injection_ids,
            )
        )
        declared_missing = []
    declared_missing_set = set(declared_missing)

    batch_id, issue = _validate_id(
        obj.get("batch_id"),
        field_name="batch_id",
        maximum_width=40,
        path=source_path,
        line=line,
        injection_ids=injection_ids,
    )
    if issue:
        issues.append(issue)
    request_id, issue = _validate_id(
        obj.get("request_id"),
        field_name="request_id",
        maximum_width=64,
        path=source_path,
        line=line,
        injection_ids=injection_ids,
    )
    if issue:
        issues.append(issue)
    output_digest, issue = _validate_digest(
        obj.get("output_digest"),
        field_name="output_digest",
        path=source_path,
        line=line,
        injection_ids=injection_ids,
    )
    if issue:
        issues.append(issue)

    values: dict[str, Any] = {
        "param_version": MISSING,
        "duration_ms": MISSING,
        "worker_id": MISSING,
        "rank": MISSING,
        "covered_refs": MISSING,
        "param_digest": MISSING,
    }
    missing_fields: list[str] = []
    required = set(_VERSION_FIELDS[version])

    if "param_version" in obj:
        raw_value = obj["param_version"]
        if _parse_missing_marker(raw_value, "param_version"):
            values["param_version"] = missing("param_version")
            missing_fields.append("param_version")
        elif not is_int(raw_value) or raw_value < 0:
            issues.append(
                _malformed(
                    "param_version must be a non-negative integer",
                    path=source_path,
                    line=line,
                    item="param_version",
                    injection_ids=injection_ids,
                )
            )
        else:
            values["param_version"] = int(raw_value)
    elif "param_version" in required:
        issues.append(
            _malformed(
                "param_version is required",
                path=source_path,
                line=line,
                item="param_version",
                injection_ids=injection_ids,
                detail_code="MISSING_FIELD",
            )
        )
    else:
        values["param_version"] = missing("param_version")
        missing_fields.append("param_version")

    if "duration_ms" in obj:
        raw_value = obj["duration_ms"]
        if _parse_missing_marker(raw_value, "duration_ms"):
            values["duration_ms"] = missing("duration_ms")
            missing_fields.append("duration_ms")
        elif not is_finite_number(raw_value):
            issues.append(
                _malformed(
                    "duration_ms must be finite",
                    path=source_path,
                    line=line,
                    item="duration_ms",
                    injection_ids=injection_ids,
                    detail_code="NON_FINITE",
                )
            )
        elif not is_int(raw_value) or raw_value < 0:
            issues.append(
                _malformed(
                    "duration_ms must be a non-negative integer",
                    path=source_path,
                    line=line,
                    item="duration_ms",
                    injection_ids=injection_ids,
                )
            )
        else:
            values["duration_ms"] = int(raw_value)
    elif "duration_ms" in required:
        issues.append(
            _malformed(
                "duration_ms is required",
                path=source_path,
                line=line,
                item="duration_ms",
                injection_ids=injection_ids,
                detail_code="MISSING_FIELD",
            )
        )
    else:
        values["duration_ms"] = missing("duration_ms")
        missing_fields.append("duration_ms")

    if "worker_id" in obj:
        raw_value = obj["worker_id"]
        if _parse_missing_marker(raw_value, "worker_id"):
            values["worker_id"] = missing("worker_id")
            missing_fields.append("worker_id")
        else:
            worker_id, issue = _validate_id(
                raw_value,
                field_name="worker_id",
                maximum_width=32,
                path=source_path,
                line=line,
                injection_ids=injection_ids,
            )
            if issue:
                issues.append(issue)
            else:
                values["worker_id"] = worker_id
    elif "worker_id" in required:
        issues.append(
            _malformed(
                "worker_id is required",
                path=source_path,
                line=line,
                item="worker_id",
                injection_ids=injection_ids,
                detail_code="MISSING_FIELD",
            )
        )
    else:
        values["worker_id"] = missing("worker_id")
        missing_fields.append("worker_id")

    if "rank" in obj:
        raw_value = obj["rank"]
        if _parse_missing_marker(raw_value, "rank"):
            values["rank"] = missing("rank")
            missing_fields.append("rank")
        elif not is_int(raw_value) or raw_value < 0:
            issues.append(
                _malformed(
                    "rank must be a non-negative integer",
                    path=source_path,
                    line=line,
                    item="rank",
                    injection_ids=injection_ids,
                )
            )
        else:
            values["rank"] = int(raw_value)
    elif "rank" in required:
        issues.append(
            _malformed(
                "rank is required",
                path=source_path,
                line=line,
                item="rank",
                injection_ids=injection_ids,
                detail_code="MISSING_FIELD",
            )
        )
    else:
        values["rank"] = missing("rank")
        missing_fields.append("rank")

    if "covered_refs" in obj:
        raw_value = obj["covered_refs"]
        if _parse_missing_marker(raw_value, "covered_refs"):
            values["covered_refs"] = missing("covered_refs")
            missing_fields.append("covered_refs")
        elif not isinstance(raw_value, list) or not all(
            isinstance(value, str) and 1 <= len(value) <= 128 for value in raw_value
        ):
            issues.append(
                _malformed(
                    "covered_refs must be a list of width-1..128 strings",
                    path=source_path,
                    line=line,
                    item="covered_refs",
                    injection_ids=injection_ids,
                    detail_code="FIELD_WIDTH",
                )
            )
        elif any(value.count("/") != 1 for value in raw_value):
            issues.append(
                _malformed(
                    "each covered_ref must be batch_id/request_id",
                    path=source_path,
                    line=line,
                    item="covered_refs",
                    injection_ids=injection_ids,
                )
            )
        else:
            values["covered_refs"] = tuple(raw_value)
    elif "covered_refs" in required:
        issues.append(
            _malformed(
                "covered_refs is required",
                path=source_path,
                line=line,
                item="covered_refs",
                injection_ids=injection_ids,
                detail_code="MISSING_FIELD",
            )
        )
    else:
        values["covered_refs"] = missing("covered_refs")
        missing_fields.append("covered_refs")

    if "param_digest" in obj:
        raw_value = obj["param_digest"]
        if _parse_missing_marker(raw_value, "param_digest"):
            values["param_digest"] = missing("param_digest")
            missing_fields.append("param_digest")
        else:
            param_digest, issue = _validate_digest(
                raw_value,
                field_name="param_digest",
                path=source_path,
                line=line,
                injection_ids=injection_ids,
            )
            if issue:
                issues.append(issue)
            else:
                values["param_digest"] = param_digest
    elif "param_digest" in required:
        issues.append(
            _malformed(
                "param_digest is required",
                path=source_path,
                line=line,
                item="param_digest",
                injection_ids=injection_ids,
                detail_code="MISSING_FIELD",
            )
        )
    else:
        values["param_digest"] = missing("param_digest")
        missing_fields.append("param_digest")

    if declared_missing_set - set(missing_fields):
        issues.append(
            _malformed(
                "missing_fields declares fields that are present or unknown",
                path=source_path,
                line=line,
                item="missing_fields",
                injection_ids=injection_ids,
            )
        )
    if missing_fields and not allow_explicit_missing:
        for field_name in sorted(set(missing_fields) & required):
            issues.append(
                _malformed(
                    f"{field_name} is required for {version}",
                    path=source_path,
                    line=line,
                    item=field_name,
                    injection_ids=injection_ids,
                    detail_code="MISSING_FIELD",
                )
            )
    canonical_missing = tuple(sorted(set(missing_fields)))
    completeness = "complete" if not canonical_missing else "explicit-missing"
    receipt = Receipt(
        version=version,
        batch_id=batch_id or "",
        request_id=request_id or "",
        output_digest=output_digest or "",
        param_version=values["param_version"],
        duration_ms=values["duration_ms"],
        worker_id=values["worker_id"],
        rank=values["rank"],
        covered_refs=values["covered_refs"],
        param_digest=values["param_digest"],
        missing_fields=canonical_missing,
        completeness=completeness,
        source_path=source_path,
        source_line=line,
        default_worker_id=default_worker_id,
        default_param_version=default_param_version,
        injection_ids=tuple(injection_ids),
        raw=dict(obj),
    )
    if issues:
        return ParseOutcome(None, issues)
    return ParseOutcome(receipt, [])


def parse_receipt_text(
    text: str,
    *,
    version: str,
    source_path: str = "",
    line: int = 0,
    default_worker_id: str | None = None,
    default_param_version: int | None = None,
    injection_ids: tuple[str, ...] = (),
    force_code: str = "",
    allow_explicit_missing: bool = False,
) -> ParseOutcome:
    if force_code:
        return ParseOutcome(
            None,
            [
                _malformed(
                    "fault injection replaced this row with a malformed row",
                    path=source_path,
                    line=line,
                    injection_ids=injection_ids,
                )
            ],
        )
    try:
        obj = json.loads(
            text,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        non_finite = "non-finite" in str(exc).lower()
        return ParseOutcome(
            None,
            [
                _malformed(
                    f"invalid receipt JSON: {exc}",
                    path=source_path,
                    line=line,
                    injection_ids=injection_ids,
                    detail_code="NON_FINITE" if non_finite else "",
                )
            ],
        )
    return parse_receipt_object(
        obj,
        version=version,
        source_path=source_path,
        line=line,
        default_worker_id=default_worker_id,
        default_param_version=default_param_version,
        injection_ids=injection_ids,
        allow_explicit_missing=allow_explicit_missing,
    )


def receipt_from_dict(data: dict[str, Any]) -> Receipt:
    """Restore a normalized receipt serialized by ``Receipt.to_dict``."""

    provenance = data.get("provenance", {}) if isinstance(data.get("provenance"), dict) else {}
    obj = {
        "batch_id": data.get("batch_id"),
        "request_id": data.get("request_id"),
        "output_digest": data.get("output_digest"),
        "param_version": data.get("param_version"),
        "duration_ms": data.get("duration_ms"),
        "worker_id": data.get("worker_id"),
        "rank": data.get("rank"),
        "covered_refs": data.get("covered_refs"),
        "param_digest": data.get("param_digest"),
    }
    outcome = parse_receipt_object(
        obj,
        version=str(data.get("version", "r3")),
        source_path=str(provenance.get("path", "")),
        line=int(provenance.get("line", 0) or 0),
        default_worker_id=provenance.get("default_worker_id"),
        default_param_version=provenance.get("default_param_version"),
        allow_explicit_missing=True,
    )
    if outcome.receipt is None:
        raise ValueError(
            "cannot restore cached receipt: "
            + "; ".join(issue.message for issue in outcome.issues)
        )
    restored = outcome.receipt.with_fields(
        completeness=str(data.get("completeness", outcome.receipt.completeness)),
        missing_fields=tuple(data.get("missing_fields", outcome.receipt.missing_fields)),
        injection_ids=tuple(data.get("injection_ids", ())),
    )
    return restored


def _validate_known_version(receipt: Receipt, config: AppConfig, *, project_root: Path) -> Issue | None:
    version = receipt.resolved_param_version
    worker_id = receipt.resolved_worker_id
    if version is None:
        return None
    if worker_id is None:
        known_versions = {
            version
            for worker in config.workers
            for version in (worker.known_param_versions or (worker.param_version,))
        }
    else:
        worker = config.worker_by_id.get(worker_id)
        if worker is None:
            return Issue(
                code=E_UNKNOWN_PARAM_VERSION,
                message=f"worker {worker_id!r} is not present in configuration",
                path=(
                    receipt.source_path
                    if not Path(receipt.source_path).is_absolute()
                    else safe_relpath(Path(receipt.source_path), project_root)
                )
                if receipt.source_path
                else "",
                line=receipt.source_line,
                item=receipt.key,
            )
        known_versions = set(worker.known_param_versions or (worker.param_version,))
    if version not in known_versions:
        return Issue(
            code=E_UNKNOWN_PARAM_VERSION,
            message=f"parameter version {version} is not active for {worker_id or 'receipt'}",
            path=(
                receipt.source_path
                if not Path(receipt.source_path).is_absolute()
                else safe_relpath(Path(receipt.source_path), project_root)
            )
            if receipt.source_path
            else "",
            line=receipt.source_line,
            item=receipt.key,
        )
    return None


def load_receipts(
    config: AppConfig,
    *,
    project_root: Path,
    injector: FaultInjector | None = None,
) -> ReceiptLoadResult:
    issues: list[Issue] = []
    receipts: list[Receipt] = []
    fingerprints: dict[str, str] = {}
    files: list[str] = []
    from .util import sha256_file

    for source in config.receipt_sources:
        path = (project_root / source.path).resolve()
        relative = safe_relpath(path, project_root)
        files.append(relative)
        filesystem_path = Path(source.path).expanduser()
        if not filesystem_path.is_absolute() and not path.exists() and filesystem_path.exists():
            path = filesystem_path.resolve()
            relative = safe_relpath(path, project_root)
        version = detect_version(relative, source.format)
        if version is None:
            issues.append(
                _malformed(
                    "cannot determine receipt version from path or source.format",
                    path=relative,
                    line=0,
                )
            )
            fingerprints[relative] = "missing"
            continue
        if not path.exists():
            issues.append(
                _malformed(
                    "receipt file does not exist",
                    path=relative,
                    line=0,
                )
            )
            fingerprints[relative] = "missing"
            continue
        try:
            fingerprints[relative] = sha256_file(path)
            handle = path.open("rb")
        except OSError as exc:
            issues.append(_malformed(f"cannot read receipt file: {exc}", path=relative, line=0))
            fingerprints[relative] = "unreadable"
            continue
        with handle:
            for line_number, raw_line in enumerate(handle, start=1):
                try:
                    text = raw_line.decode("utf-8").rstrip("\r\n")
                except UnicodeDecodeError:
                    issues.append(
                        _malformed(
                            "receipt row is not valid UTF-8",
                            path=relative,
                            line=line_number,
                        )
                    )
                    continue
                if not text.strip():
                    continue
                effect = (
                    injector.transform_text(text, path, line_number)
                    if injector is not None
                    else None
                )
                parse_text = effect.text if effect is not None else text
                injection_ids = effect.injection_ids if effect is not None else ()
                force_code = effect.force_code if effect is not None else ""
                outcome = parse_receipt_text(
                    parse_text,
                    version=version,
                    source_path=relative,
                    line=line_number,
                    default_worker_id=source.default_worker_id,
                    default_param_version=source.default_param_version,
                    injection_ids=injection_ids,
                    force_code=force_code,
                )
                if outcome.receipt is None:
                    issues.extend(outcome.issues)
                    continue
                receipt = outcome.receipt
                if injector is not None:
                    injector.observe_rank(path, receipt.rank)
                version_issue = _validate_known_version(receipt, config, project_root=project_root)
                if version_issue is not None:
                    if injection_ids:
                        version_issue = Issue(
                            code=version_issue.code,
                            message=version_issue.message,
                            path=version_issue.path,
                            line=version_issue.line,
                            item=version_issue.item,
                            injection_id=",".join(injection_ids),
                        )
                    issues.append(version_issue)
                    continue
                receipts.append(receipt)
    receipts.sort(key=lambda item: (item.source_path, item.source_line))
    return ReceiptLoadResult(
        receipts=tuple(receipts),
        issues=issues,
        file_fingerprints=dict(sorted(fingerprints.items())),
        files=tuple(files),
    )
