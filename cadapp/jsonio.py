"""JSON surface description format (read/write with migration)."""

from __future__ import annotations

import json

from .errors import CadError, ErrorCode
from .migrate import migrate_document
from .model import Model, canonical_bytes, model_from_dict, model_to_dict


def dumps_model(model: Model) -> str:
    """Canonical JSON text for the model (deterministic byte content)."""
    return canonical_bytes(model).decode("utf-8")


def loads_model(text: str):
    """Parse JSON text into a Model, migrating older schema versions.

    Returns (model, migration_actions).  Older documents are upgraded via
    the fixed migration table; newer documents raise E_SCHEMA_VERSION.
    """
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CadError(ErrorCode.E_FORMAT_HEADER,
                       f"invalid JSON: {exc}", "<root>") from exc
    doc, actions = migrate_document(doc)
    return model_from_dict(doc), actions


def model_to_json_dict(model: Model) -> dict:
    return model_to_dict(model)
