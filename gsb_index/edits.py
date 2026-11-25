"""Serializable edit sequences used for regression and incremental builds."""

from __future__ import annotations

import copy
import json
from typing import Any, Iterable, Mapping

from .canonical import canonical_json
from .codes import Issue
from .config import parse_project
from .models import Project


def project_to_config(project: Project) -> dict[str, Any]:
    revisions = []
    for node in project.revisions:
        item = {
            "version": node.version,
            "parent": node.parent,
            "bucket_count": node.profile.bucket_count,
            "bucket_weights": list(node.profile.bucket_weights),
            "encoding": node.profile.encoding,
        }
        if node.changes:
            item["changes"] = dict(node.changes)
        revisions.append(item)
    return {
        "vocabulary": {
            "terms": list(project.vocabulary.terms),
            "stopwords": sorted(project.vocabulary.stopwords),
        },
        "documents": [doc.to_dict() for doc in project.documents],
        "layout": {"shards": [shard.to_dict() for shard in project.shards]},
        "compression": project.profile_for_revision(project.current_revision).to_dict(),
        "budget": project.budget.to_dict(),
        "revisions": {
            "revisions": revisions,
            "current_revision": project.current_revision,
        },
        "scoring": project.scoring.to_dict(),
        "queries": [query.to_dict() for query in project.queries],
    }


def _copy_raw(project: Project) -> dict[str, Any]:
    return json.loads(canonical_json(project_to_config(project)))


def _replace_layout_revision(raw: dict[str, Any], revision: int) -> None:
    for shard in raw.get("layout", {}).get("shards", []):
        shard["revision"] = revision
    raw.setdefault("revisions", {})["current_revision"] = revision


def apply_one_edit(raw: dict[str, Any], edit: Mapping[str, Any]) -> dict[str, Any]:
    raw = copy.deepcopy(raw)
    edit_type = edit.get("type")
    if edit_type in {"append", "add_documents"}:
        new_docs = edit.get("documents", [])
        if isinstance(new_docs, Mapping):
            new_docs = [new_docs]
        if not isinstance(new_docs, list):
            raise ValueError("append edit documents must be a list or object")
        raw.setdefault("documents", []).extend(copy.deepcopy(new_docs))
    elif edit_type == "replace":
        document = edit.get("document")
        if not isinstance(document, Mapping):
            raise ValueError("replace edit requires a document object")
        doc_id = document.get("doc_id")
        existing = raw.setdefault("documents", [])
        matched = False
        for index, item in enumerate(existing):
            if item.get("doc_id") == doc_id:
                existing[index] = copy.deepcopy(dict(document))
                matched = True
                break
        if not matched:
            existing.append(copy.deepcopy(dict(document)))
    elif edit_type == "delete":
        doc_ids = edit.get("doc_ids", [edit.get("doc_id")])
        delete_set = {str(doc_id) for doc_id in doc_ids if doc_id is not None}
        raw["documents"] = [item for item in raw.get("documents", []) if item.get("doc_id") not in delete_set]
    elif edit_type == "set_shard":
        shard_id = edit.get("shard_id")
        updates = {
            key: value
            for key, value in edit.items()
            if key
            in {
                "port",
                "term_bucket_capacity",
                "concurrency_limit",
                "seed",
                "revision",
            }
        }
        matched = False
        for shard in raw.get("layout", {}).get("shards", []):
            if shard.get("shard_id") == shard_id:
                shard.update(updates)
                matched = True
                break
        if not matched:
            raise ValueError(f"unknown shard in edit: {shard_id}")
    elif edit_type in {"set_current_revision", "rollback"}:
        revision = edit.get("revision")
        if not isinstance(revision, int) or isinstance(revision, bool):
            raise ValueError("revision edit requires an integer revision")
        _replace_layout_revision(raw, revision)
        current_profile = None
        for revision_item in raw.get("revisions", {}).get("revisions", []):
            if revision_item.get("version") == revision:
                current_profile = {
                    "bucket_count": revision_item["bucket_count"],
                    "bucket_weights": revision_item["bucket_weights"],
                    "encoding": revision_item["encoding"],
                }
                break
        if current_profile is None:
            raise ValueError(f"unknown revision in edit: {revision}")
        raw["compression"] = current_profile
    elif edit_type == "add_revision":
        revision = edit.get("revision")
        if not isinstance(revision, Mapping):
            raise ValueError("add_revision edit requires a revision object")
        raw.setdefault("revisions", {}).setdefault("revisions", []).append(copy.deepcopy(dict(revision)))
        if edit.get("activate", True):
            version = revision.get("version")
            _replace_layout_revision(raw, int(version))
            if all(key in revision for key in ("bucket_count", "bucket_weights", "encoding")):
                raw["compression"] = {
                    "bucket_count": revision["bucket_count"],
                    "bucket_weights": revision["bucket_weights"],
                    "encoding": revision["encoding"],
                }
    elif edit_type == "set_compression":
        revision = edit.get("revision")
        if not isinstance(revision, Mapping):
            raise ValueError("set_compression edit requires a revision object")
        raw.setdefault("revisions", {}).setdefault("revisions", []).append(copy.deepcopy(dict(revision)))
        version = int(revision["version"])
        raw["compression"] = {
            "bucket_count": revision["bucket_count"],
            "bucket_weights": revision["bucket_weights"],
            "encoding": revision["encoding"],
        }
        _replace_layout_revision(raw, version)
    elif edit_type == "set_budget":
        raw.setdefault("budget", {}).update(copy.deepcopy(dict(edit.get("budget", {}))))
    elif edit_type == "set_scoring":
        raw.setdefault("scoring", {}).update(copy.deepcopy(dict(edit.get("scoring", {}))))
    elif edit_type == "set_queries":
        queries = edit.get("queries")
        if not isinstance(queries, list):
            raise ValueError("set_queries edit requires a list")
        raw["queries"] = copy.deepcopy(queries)
    else:
        raise ValueError(f"unsupported edit type: {edit_type}")
    return raw


def apply_edit_sequence(
    project: Project,
    edits: Iterable[Mapping[str, Any]],
) -> tuple[Project, list[dict[str, Any]]]:
    raw = _copy_raw(project)
    issues: list[dict[str, Any]] = []
    current = project
    for index, edit in enumerate(edits, start=1):
        edit_id = str(edit.get("edit_id", f"edit-{index:03d}"))
        try:
            raw = apply_one_edit(raw, edit)
        except ValueError as exc:
            parsed = Issue(
                code="E_EDIT_INVALID",
                message=str(exc),
                object_type="edit",
                object_id=edit_id,
            ).to_dict()
            issues.append(parsed)
            continue
        current, edit_issues = parse_project(raw)
        for item in edit_issues:
            annotated = dict(item)
            annotated["edit_id"] = edit_id
            issues.append(annotated)
    return current, issues

