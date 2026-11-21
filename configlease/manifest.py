"""Revision manifest loading and validation.

Manifest schema (local JSON file)::

    {
      "revisions": [
        {"rev_seq": 1, "parent": 0, "changes": {"key_a": 10, "key_b": 3}},
        ...
      ]
    }

``changes`` maps integer/string configuration keys to integer values and
``parent`` references the previous revision sequence (0 for the root).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import ManifestError


class Revision:
    def __init__(self, rev_seq: int, parent: int, changes: dict[str, int],
                 kind: str = "normal", rollback_target: int | None = None):
        self.rev_seq = int(rev_seq)
        self.parent = int(parent)
        self.changes = {str(k): int(v) for k, v in changes.items()}
        self.kind = kind  # normal | rollback
        self.rollback_target = rollback_target

    def to_dict(self) -> dict[str, Any]:
        data = {
            "rev_seq": self.rev_seq,
            "parent": self.parent,
            "changes": self.changes,
            "kind": self.kind,
        }
        if self.rollback_target is not None:
            data["rollback_target"] = self.rollback_target
        return data


def load_manifest(path: str | Path) -> list[Revision]:
    p = Path(path)
    try:
        raw = p.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ManifestError(f"manifest file not found: {p}", path=str(p)) from exc
    except OSError as exc:
        raise ManifestError(f"manifest unreadable: {p}", path=str(p)) from exc
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ManifestError(
            "manifest JSON is corrupted",
            path=str(p),
            detail=f"{exc.msg} at line {exc.lineno} column {exc.colno}",
        ) from exc
    return parse_manifest(doc, source=str(p))


def parse_manifest(doc: Any, source: str = "<manifest>") -> list[Revision]:
    if not isinstance(doc, dict) or not isinstance(doc.get("revisions"), list):
        raise ManifestError("manifest must contain a 'revisions' array", path=source)
    revisions: list[Revision] = []
    seen: set[int] = set()
    expected_parent = 0
    for index, item in enumerate(doc["revisions"]):
        if not isinstance(item, dict):
            raise ManifestError("revision entry must be an object", index=index, path=source)
        try:
            rev_seq = int(item["rev_seq"])
            parent = int(item["parent"])
            changes = item["changes"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ManifestError(
                "revision entry missing/invalid rev_seq, parent or changes",
                index=index, path=source,
            ) from exc
        if rev_seq in seen:
            raise ManifestError("duplicate rev_seq", rev_seq=rev_seq, path=source)
        if rev_seq <= 0:
            raise ManifestError("rev_seq must be positive", rev_seq=rev_seq, path=source)
        if parent != expected_parent:
            raise ManifestError(
                "non-contiguous parent reference",
                rev_seq=rev_seq, parent=parent, expected_parent=expected_parent, path=source,
            )
        if not isinstance(changes, dict):
            raise ManifestError("changes must be an object", rev_seq=rev_seq, path=source)
        clean_changes: dict[str, int] = {}
        for key, value in changes.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ManifestError(
                    "change values must be integers", rev_seq=rev_seq, key=str(key), path=source,
                )
            clean_changes[str(key)] = value
        seen.add(rev_seq)
        expected_parent = rev_seq
        revisions.append(Revision(rev_seq, parent, clean_changes))
    if not revisions:
        raise ManifestError("manifest contains no revisions", path=source)
    return revisions


def fold_view(revisions: list[Revision], upto_seq: int | None = None) -> dict[str, int]:
    """Fold revisions (in sequence order) into a full key -> int view."""
    view: dict[str, int] = {}
    for rev in revisions:
        if upto_seq is not None and rev.rev_seq > upto_seq:
            break
        view.update(rev.changes)
    return view


def key_universe(revisions: list[Revision]) -> set[str]:
    keys: set[str] = set()
    for rev in revisions:
        keys.update(rev.changes)
    return keys
