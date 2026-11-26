from __future__ import annotations

from typing import Any

from .hashing import fingerprint


def bundle_digest(bundle: dict[str, Any]) -> str:
    return fingerprint(bundle.get("documents", {}))


def diff_annotation_versions(
    previous: dict[str, Any] | None,
    current: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not current:
        return []
    if not previous:
        return [
            {
                "kind": "ANNOTATION_BASELINE",
                "previous_version": None,
                "current_version": current.get("version_id"),
                "previous_digest": None,
                "current_digest": current.get("content_digest"),
                "image_id": None,
                "decision": "establish_baseline",
            }
        ]

    events: list[dict[str, Any]] = []
    version_changed = previous.get("version_id") != current.get("version_id")
    digest_changed = previous.get("content_digest") != current.get("content_digest")
    if version_changed and not digest_changed:
        events.append(
            {
                "kind": "ANNOTATION_VERSION_RELABEL",
                "previous_version": previous.get("version_id"),
                "current_version": current.get("version_id"),
                "previous_digest": previous.get("content_digest"),
                "current_digest": current.get("content_digest"),
                "image_id": None,
                "decision": "digest_unchanged_reuse_all_evidence",
            }
        )
    elif version_changed or digest_changed:
        events.append(
            {
                "kind": "ANNOTATION_BUNDLE_REPLACED",
                "previous_version": previous.get("version_id"),
                "current_version": current.get("version_id"),
                "previous_digest": previous.get("content_digest"),
                "current_digest": current.get("content_digest"),
                "version_changed": version_changed,
                "digest_changed": digest_changed,
                "image_id": None,
                "decision": "invalidate_only_changed_documents",
            }
        )

    old_docs = previous.get("documents", {}) if isinstance(previous, dict) else {}
    new_docs = current.get("documents", {}) if isinstance(current, dict) else {}
    for image_id in sorted(set(old_docs) | set(new_docs)):
        old_doc = old_docs.get(image_id)
        new_doc = new_docs.get(image_id)
        old_digest = (old_doc or {}).get("content_digest") if isinstance(old_doc, dict) else None
        new_digest = (new_doc or {}).get("content_digest") if isinstance(new_doc, dict) else None
        if old_doc is None:
            kind = "ANNOTATION_DOC_ADDED"
            decision = "compute_keypoint_and_reading"
        elif new_doc is None:
            kind = "ANNOTATION_DOC_REMOVED"
            decision = "invalidate_keypoint_and_reading"
        elif old_digest != new_digest:
            kind = "ANNOTATION_DOC_CHANGED"
            decision = "invalidate_keypoint_and_reading_only"
        else:
            kind = "ANNOTATION_DOC_UNCHANGED"
            decision = "reuse_window_component_keypoint_reading_keys"
        events.append(
            {
                "kind": kind,
                "image_id": image_id,
                "previous_version": (old_doc or {}).get("version_id") if isinstance(old_doc, dict) else None,
                "current_version": (new_doc or {}).get("version_id") if isinstance(new_doc, dict) else None,
                "previous_digest": old_digest,
                "current_digest": new_digest,
                "decision": decision,
            }
        )
    return events
