"""Trust-store revision-chain loading and normalization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .canonical import (
    digest_obj,
    normalize_fingerprint,
    normalize_serial,
    read_json,
)
from .certfamily import DAY_MS, Family
from .errors import E_REVOKE, E_TRUST, WorkbenchError


@dataclass(frozen=True)
class Revision:
    revision: int
    parent_revision: int | None
    trusted_root_fingerprints: tuple[str, ...]
    revoked_serial_numbers: tuple[str, ...]
    path_len_overrides: tuple[tuple[str, int], ...]
    validity_evaluation_epoch_ms: int
    label: str | None = None

    def content(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "revision": self.revision,
            "parent_revision": self.parent_revision,
            "trusted_root_fingerprints": sorted(set(self.trusted_root_fingerprints)),
            "revoked_serial_numbers": sorted(set(self.revoked_serial_numbers)),
            "path_len_overrides": [
                {"certificate_fingerprint": fp, "path_len": path_len}
                for fp, path_len in sorted(self.path_len_overrides)
            ],
            "validity_evaluation_epoch_ms": self.validity_evaluation_epoch_ms,
        }
        if self.label is not None:
            data["label"] = self.label
        return data

    def digest(self) -> str:
        return digest_obj(self.content())

    def revoked_set(self) -> set[str]:
        return set(self.revoked_serial_numbers)

    def trusted_roots_set(self) -> set[str]:
        return set(self.trusted_root_fingerprints)

    def path_len_map(self) -> dict[str, int]:
        return dict(self.path_len_overrides)


@dataclass(frozen=True)
class TrustDocument:
    revisions: tuple[Revision, ...]
    source_path: str | None = None

    @property
    def chain_digest(self) -> str:
        return digest_obj([revision.content() for revision in self.revisions])

    @property
    def revision_numbers(self) -> list[int]:
        return [revision.revision for revision in self.revisions]

    def by_number(self, number: int) -> Revision:
        for revision in self.revisions:
            if revision.revision == number:
                return revision
        raise KeyError(number)

    def to_json(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "revisions": [revision.content() for revision in self.revisions],
        }


def _trust_error(revision: int | None, field_path: str, message: str, **details: Any) -> WorkbenchError:
    return WorkbenchError(
        E_TRUST,
        message,
        revision=revision,
        field_path=field_path,
        details=details,
    )


def _expect_int(value: Any, revision: int | None, field_path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _trust_error(revision, field_path, f"expected integer >= {minimum}")
    return value


def _parse_path_overrides(
    value: Any,
    revision: int,
    field_path: str,
    family: Family,
) -> tuple[tuple[str, int], ...]:
    result: dict[str, int] = {}
    if value is None:
        return tuple()
    if isinstance(value, list):
        for index, item in enumerate(value):
            item_path = f"{field_path}[{index}]"
            if not isinstance(item, dict):
                raise _trust_error(revision, item_path, "path override must be an object")
            fp_raw = item.get("certificate_fingerprint")
            path_len = item.get("path_len")
            if not isinstance(fp_raw, str):
                raise _trust_error(
                    revision,
                    item_path + ".certificate_fingerprint",
                    "missing certificate fingerprint",
                )
            try:
                fingerprint = normalize_fingerprint(fp_raw)
            except ValueError as exc:
                raise _trust_error(
                    revision,
                    item_path + ".certificate_fingerprint",
                    str(exc),
                ) from exc
            if fingerprint not in family.certificate_fingerprints:
                raise _trust_error(
                    revision,
                    item_path + ".certificate_fingerprint",
                    "path override fingerprint is not in the certificate family",
                    fingerprint=fingerprint,
                )
            path_value = _expect_int(path_len, revision, item_path + ".path_len", minimum=0)
            if path_value > 16:
                raise _trust_error(revision, item_path + ".path_len", "path_len is unreasonably large")
            if fingerprint in result:
                raise _trust_error(
                    revision,
                    item_path + ".certificate_fingerprint",
                    "duplicate path override fingerprint",
                )
            result[fingerprint] = path_value
    elif isinstance(value, dict):
        for index, (fp_raw, path_len) in enumerate(value.items()):
            item_path = f"{field_path}[{index}]"
            try:
                fingerprint = normalize_fingerprint(str(fp_raw))
            except ValueError as exc:
                raise _trust_error(
                    revision,
                    item_path + ".certificate_fingerprint",
                    str(exc),
                ) from exc
            if fingerprint not in family.certificate_fingerprints:
                raise _trust_error(
                    revision,
                    item_path + ".certificate_fingerprint",
                    "path override fingerprint is not in the certificate family",
                    fingerprint=fingerprint,
                )
            path_value = _expect_int(path_len, revision, item_path + ".path_len", minimum=0)
            if path_value > 16:
                raise _trust_error(revision, item_path + ".path_len", "path_len is unreasonably large")
            result[fingerprint] = path_value
    else:
        raise _trust_error(revision, field_path, "path_len_overrides must be an object or list")
    return tuple(sorted(result.items()))


def _parse_one_revision(
    raw: Any,
    index: int,
    family: Family,
    *,
    strict_revoke: bool,
) -> Revision:
    prefix = f"revisions[{index}]"
    if not isinstance(raw, dict):
        raise _trust_error(None, prefix, "revision must be an object")
    revision = _expect_int(raw.get("revision"), None, prefix + ".revision", minimum=1)
    parent_raw = raw.get("parent_revision", None)
    if isinstance(parent_raw, bool):
        raise _trust_error(revision, prefix + ".parent_revision", "parent_revision must not be boolean")
    if parent_raw is None or parent_raw == 0:
        parent: int | None = None
    else:
        parent = _expect_int(parent_raw, revision, prefix + ".parent_revision", minimum=1)

    roots_raw = raw.get("trusted_root_fingerprints")
    if not isinstance(roots_raw, list) or not roots_raw:
        raise _trust_error(
            revision,
            prefix + ".trusted_root_fingerprints",
            "trusted_root_fingerprints must be a non-empty list",
        )
    roots: list[str] = []
    family_roots = set(family.root_fingerprints)
    for root_index, value in enumerate(roots_raw):
        path = f"{prefix}.trusted_root_fingerprints[{root_index}]"
        if not isinstance(value, str):
            raise _trust_error(revision, path, "root fingerprint must be a string")
        try:
            fingerprint = normalize_fingerprint(value)
        except ValueError as exc:
            raise _trust_error(revision, path, str(exc)) from exc
        if fingerprint not in family_roots:
            raise _trust_error(
                revision,
                path,
                "trusted root fingerprint is not a root in the certificate family",
                fingerprint=fingerprint,
            )
        roots.append(fingerprint)
    if len(set(roots)) != len(roots):
        raise _trust_error(
            revision,
            prefix + ".trusted_root_fingerprints",
            "duplicate trusted root fingerprints",
        )

    revoked_raw = raw.get("revoked_serial_numbers", [])
    if revoked_raw is None:
        revoked_raw = []
    if not isinstance(revoked_raw, list):
        raise _trust_error(
            revision,
            prefix + ".revoked_serial_numbers",
            "revoked_serial_numbers must be a list",
        )
    revoked: list[str] = []
    family_serials = set(family.serial_numbers)
    for serial_index, value in enumerate(revoked_raw):
        path = f"{prefix}.revoked_serial_numbers[{serial_index}]"
        if not isinstance(value, str):
            raise _trust_error(revision, path, "revocation serial must be a hexadecimal string")
        try:
            serial = normalize_serial(value)
        except (ValueError, TypeError) as exc:
            raise _trust_error(revision, path, str(exc)) from exc
        if strict_revoke and serial not in family_serials:
            raise WorkbenchError(
                E_REVOKE,
                "revocation references a serial number absent from the certificate family",
                revision=revision,
                field_path=path,
                details={"serial_number": serial},
            )
        revoked.append(serial)
    if len(set(revoked)) != len(revoked):
        raise _trust_error(
            revision,
            prefix + ".revoked_serial_numbers",
            "duplicate revocation serial numbers",
        )

    overrides = _parse_path_overrides(
        raw.get("path_len_overrides", {}),
        revision,
        prefix + ".path_len_overrides",
        family,
    )
    evaluation_ms = _expect_int(
        raw.get("validity_evaluation_epoch_ms"),
        revision,
        prefix + ".validity_evaluation_epoch_ms",
        minimum=0,
    )
    label_raw = raw.get("label")
    if label_raw is not None and not isinstance(label_raw, str):
        raise _trust_error(revision, prefix + ".label", "label must be a string")

    return Revision(
        revision=revision,
        parent_revision=parent,
        trusted_root_fingerprints=tuple(roots),
        revoked_serial_numbers=tuple(revoked),
        path_len_overrides=overrides,
        validity_evaluation_epoch_ms=evaluation_ms,
        label=label_raw,
    )


def parse_trust_document(
    raw: Any,
    family: Family,
    *,
    source_path: str | None = None,
    strict_revoke: bool = True,
) -> TrustDocument:
    if isinstance(raw, list):
        revisions_raw = raw
    elif isinstance(raw, dict):
        format_version = raw.get("format_version", 1)
        if isinstance(format_version, bool) or format_version != 1:
            raise _trust_error(None, "format_version", "unsupported trust document format_version")
        revisions_raw = raw.get("revisions")
    else:
        raise _trust_error(None, "$", "trust document must be an object or list")
    if not isinstance(revisions_raw, list) or not revisions_raw:
        raise _trust_error(None, "revisions", "revisions must be a non-empty list")

    revisions: list[Revision] = []
    seen_numbers: set[int] = set()
    for index, item in enumerate(revisions_raw):
        revision = _parse_one_revision(item, index, family, strict_revoke=strict_revoke)
        expected_number = revision.revision if index == 0 else revisions[-1].revision + 1
        if revision.revision != expected_number:
            raise _trust_error(
                revision.revision,
                f"revisions[{index}].revision",
                "revision numbers must be contiguous and monotonically increasing after the first revision",
                expected=expected_number,
                actual=revision.revision,
            )
        if revision.revision in seen_numbers:
            raise _trust_error(
                revision.revision,
                f"revisions[{index}].revision",
                "duplicate revision number",
            )
        seen_numbers.add(revision.revision)
        if index == 0:
            if revision.parent_revision is not None:
                raise _trust_error(
                    revision.revision,
                    f"revisions[{index}].parent_revision",
                    "the first revision must have a null or zero parent_revision",
                )
        elif revision.parent_revision is None:
            raise _trust_error(
                revision.revision,
                f"revisions[{index}].parent_revision",
                "parent_revision is required after the first revision",
            )
        revisions.append(revision)

    by_number = {revision.revision: revision for revision in revisions}
    for index, revision in enumerate(revisions):
        if revision.parent_revision is not None:
            parent = by_number.get(revision.parent_revision)
            if parent is None:
                raise _trust_error(
                    revision.revision,
                    f"revisions[{index}].parent_revision",
                    "parent revision reference does not exist",
                )
            if parent.revision >= revision.revision:
                raise _trust_error(
                    revision.revision,
                    f"revisions[{index}].parent_revision",
                    "parent revision must reference an earlier revision",
                )
        visited: set[int] = set()
        current: Revision | None = revision
        while current is not None and current.parent_revision is not None:
            if current.revision in visited:
                raise _trust_error(
                    revision.revision,
                    f"revisions[{index}].parent_revision",
                    "parent revision graph contains a cycle",
                )
            visited.add(current.revision)
            current = by_number[current.parent_revision]

    return TrustDocument(revisions=tuple(revisions), source_path=source_path)


def load_trust_document(
    path: str | Path,
    family: Family,
    *,
    strict_revoke: bool = True,
) -> TrustDocument:
    try:
        raw = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise _trust_error(None, "$", f"cannot load trust document: {exc}") from exc
    return parse_trust_document(
        raw,
        family,
        source_path=str(Path(path)),
        strict_revoke=strict_revoke,
    )


def write_trust_document(path: str | Path, revisions: Iterable[Revision]) -> None:
    from .canonical import write_canonical_json

    write_canonical_json(
        path,
        {
            "format_version": 1,
            "revisions": [revision.content() for revision in revisions],
        },
    )


def revision_from_content(content: dict[str, Any]) -> Revision:
    """Rehydrate a previously validated normalized revision from persistence."""

    if not isinstance(content, dict):
        raise _trust_error(None, "revision", "stored revision must be an object")
    revision = _expect_int(content.get("revision"), None, "revision.revision", minimum=1)
    parent_raw = content.get("parent_revision")
    parent = None if parent_raw in (None, 0) else _expect_int(parent_raw, revision, "revision.parent_revision", minimum=1)
    roots_raw = content.get("trusted_root_fingerprints")
    if not isinstance(roots_raw, list) or not roots_raw:
        raise _trust_error(revision, "revision.trusted_root_fingerprints", "missing trusted roots")
    roots = tuple(normalize_fingerprint(str(item)) for item in roots_raw)
    revoked_raw = content.get("revoked_serial_numbers", [])
    if not isinstance(revoked_raw, list):
        raise _trust_error(revision, "revision.revoked_serial_numbers", "revocations must be a list")
    if any(not isinstance(item, str) for item in revoked_raw):
        raise _trust_error(revision, "revision.revoked_serial_numbers", "revocation serials must be strings")
    revoked = tuple(normalize_serial(item) for item in revoked_raw)
    overrides_raw = content.get("path_len_overrides", [])
    overrides: list[tuple[str, int]] = []
    if isinstance(overrides_raw, dict):
        for fp, path_len in overrides_raw.items():
            overrides.append((normalize_fingerprint(str(fp)), _expect_int(path_len, revision, "revision.path_len_overrides")))
    elif isinstance(overrides_raw, list):
        for index, item in enumerate(overrides_raw):
            if not isinstance(item, dict):
                raise _trust_error(revision, f"revision.path_len_overrides[{index}]", "override must be an object")
            overrides.append(
                (
                    normalize_fingerprint(str(item.get("certificate_fingerprint"))),
                    _expect_int(item.get("path_len"), revision, f"revision.path_len_overrides[{index}].path_len"),
                )
            )
    else:
        raise _trust_error(revision, "revision.path_len_overrides", "invalid overrides")
    evaluation = _expect_int(
        content.get("validity_evaluation_epoch_ms"),
        revision,
        "revision.validity_evaluation_epoch_ms",
        minimum=0,
    )
    label = content.get("label")
    return Revision(
        revision=revision,
        parent_revision=parent,
        trusted_root_fingerprints=roots,
        revoked_serial_numbers=revoked,
        path_len_overrides=tuple(sorted(overrides)),
        validity_evaluation_epoch_ms=evaluation,
        label=label if isinstance(label, str) else None,
    )


def document_from_revision_contents(
    contents: Iterable[dict[str, Any]],
    *,
    source_path: str | None = None,
) -> TrustDocument:
    revisions = tuple(revision_from_content(item) for item in contents)
    if not revisions:
        raise _trust_error(None, "revisions", "stored revision chain is empty")
    for index, revision in enumerate(revisions):
        expected = revision.revision if index == 0 else revisions[index - 1].revision + 1
        if revision.revision != expected:
            raise _trust_error(revision.revision, f"revisions[{index}].revision", "stored revisions are not contiguous")
        if (index == 0 and revision.parent_revision is not None) or (
            index > 0 and revision.parent_revision is None
        ):
            raise _trust_error(revision.revision, f"revisions[{index}].parent_revision", "stored parent chain is invalid")
    by_number = {revision.revision: revision for revision in revisions}
    for index, revision in enumerate(revisions):
        if revision.parent_revision is None:
            continue
        parent = by_number.get(revision.parent_revision)
        if parent is None or parent.revision >= revision.revision:
            raise _trust_error(
                revision.revision,
                f"revisions[{index}].parent_revision",
                "stored parent revision is missing or not earlier",
            )
    return TrustDocument(revisions=revisions, source_path=source_path)


def demo_revisions(family: Family) -> list[Revision]:
    """Build the deterministic revision chain used by the demo."""

    leaf = family.leaf("leaf-normal-01")
    chain = list(leaf["chain_fingerprints"])
    roots = family.root_fingerprints
    root1 = chain[-1]
    normal_eval = int(family.data["recommended_evaluation_ms"])
    validity_days = int(family.data["params"]["validity_days"])
    levels = int(family.data["params"]["intermediate_levels"])

    other_roots = sorted(root for root in roots if root != root1)
    wrong_root = other_roots[0] if other_roots else root1
    override_fp: str | None = None
    if levels >= 2:
        override_fp = chain[-2]
    elif levels == 1:
        override_fp = chain[-1]
    if override_fp is None:
        raise ValueError("the demo PATH_LEN revision requires at least one intermediate level")

    expired_eval = normal_eval + (validity_days + 2) * DAY_MS
    raw_revisions: list[dict[str, Any]] = [
        {
            "revision": 1,
            "parent_revision": None,
            "trusted_root_fingerprints": [root1],
            "revoked_serial_numbers": [],
            "path_len_overrides": {},
            "validity_evaluation_epoch_ms": normal_eval,
            "label": "baseline-trusted",
        },
        {
            "revision": 2,
            "parent_revision": 1,
            "trusted_root_fingerprints": [wrong_root],
            "revoked_serial_numbers": [],
            "path_len_overrides": {},
            "validity_evaluation_epoch_ms": normal_eval,
            "label": "anchor-rotated-away",
        },
        {
            "revision": 3,
            "parent_revision": 2,
            "trusted_root_fingerprints": [root1],
            "revoked_serial_numbers": [leaf["serial_number"]],
            "path_len_overrides": {},
            "validity_evaluation_epoch_ms": normal_eval,
            "label": "leaf-revoked",
        },
        {
            "revision": 4,
            "parent_revision": 3,
            "trusted_root_fingerprints": [root1],
            "revoked_serial_numbers": [],
            "path_len_overrides": {},
            "validity_evaluation_epoch_ms": expired_eval,
            "label": "evaluation-after-expiry",
        },
        {
            "revision": 5,
            "parent_revision": 4,
            "trusted_root_fingerprints": [root1],
            "revoked_serial_numbers": [],
            "path_len_overrides": {override_fp: 0},
            "validity_evaluation_epoch_ms": normal_eval,
            "label": "tightened-path-length",
        },
        {
            "revision": 6,
            "parent_revision": 5,
            "trusted_root_fingerprints": [root1],
            "revoked_serial_numbers": [],
            "path_len_overrides": {},
            "validity_evaluation_epoch_ms": normal_eval,
            "label": "recovered-trust",
        },
    ]
    return list(parse_trust_document({"revisions": raw_revisions}, family).revisions)
