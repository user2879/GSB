"""Fixed, auditable schema migration table for NPR project documents."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping

from .errors import NprError, require
from .model import (
    CURRENT_SCHEMA_VERSION,
    DEFAULT_STROKE_SAMPLE,
    DEFAULT_STYLE,
    Project,
    canonical_hash,
    project_from_dict,
)


@dataclass
class MigrationAction:
    from_version: int
    to_version: int
    field: str
    operation: str
    value: Any
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_version": self.from_version,
            "to_version": self.to_version,
            "field": self.field,
            "operation": self.operation,
            "value": copy.deepcopy(self.value),
            "rationale": self.rationale,
            "source": "migration",
        }


@dataclass
class MigrationResult:
    project: Project
    source_version: int
    target_version: int
    actions: list[MigrationAction] = field(default_factory=list)
    content_hash_before: str | None = None
    content_hash_after: str = ""

    @property
    def changed(self) -> bool:
        return bool(self.actions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_version": self.source_version,
            "target_version": self.target_version,
            "changed": self.changed,
            "actions": [action.to_dict() for action in self.actions],
            "content_hash_before": self.content_hash_before,
            "content_hash_after": self.content_hash_after,
        }


def _schema_version(raw: Mapping[str, Any]) -> int:
    version = raw.get("schema_version")
    require(
        isinstance(version, int) and not isinstance(version, bool),
        "E_SCHEMA_VERSION",
        "schema_version must be an integer",
    )
    require(
        1 <= version <= CURRENT_SCHEMA_VERSION,
        "E_SCHEMA_VERSION",
        "schema version is outside the supported migration range",
        schema_version=version,
        minimum=1,
        maximum=CURRENT_SCHEMA_VERSION,
    )
    return version


def _derive_layer_order(raw: Mapping[str, Any]) -> list[str]:
    shapes = raw.get("shapes")
    require(
        isinstance(shapes, list),
        "E_SCHEMA_VERSION",
        "cannot derive layer_order without a shapes array",
    )
    order: list[str] = []
    for shape in shapes:
        require(
            isinstance(shape, Mapping) and isinstance(shape.get("layer"), str),
            "E_SCHEMA_VERSION",
            "cannot derive layer_order from a shape without a layer",
        )
        layer = shape["layer"]
        require(bool(layer), "E_SCHEMA_VERSION", "shape layer must not be empty")
        if layer not in order:
            order.append(layer)
    require(
        bool(order),
        "E_SCHEMA_VERSION",
        "cannot derive layer_order from an empty scene",
    )
    return order


def migrate_document(raw: Mapping[str, Any]) -> MigrationResult:
    """Migrate v1/v2 documents to v3 using a fixed default table.

    Missing fields are only filled when that schema version is allowed to omit
    them. A malformed current document is never silently coerced.
    """

    require(
        isinstance(raw, Mapping),
        "E_SCHEMA_VERSION",
        "project document must be an object",
    )
    source_version = _schema_version(raw)
    document = copy.deepcopy(dict(raw))

    if source_version == CURRENT_SCHEMA_VERSION:
        project = project_from_dict(document)
        digest = project.content_hash()
        return MigrationResult(
            project=project,
            source_version=source_version,
            target_version=CURRENT_SCHEMA_VERSION,
            actions=[],
            content_hash_before=digest,
            content_hash_after=digest,
        )

    actions: list[MigrationAction] = []

    if source_version == 1:
        if "style" not in document:
            document["style"] = copy.deepcopy(DEFAULT_STYLE)
            actions.append(
                MigrationAction(
                    from_version=1,
                    to_version=3,
                    field="style",
                    operation="default",
                    value=copy.deepcopy(DEFAULT_STYLE),
                    rationale=(
                        "v1 had no style block; v3 line-art defaults preserve the "
                        "legacy black-on-paper rendering"
                    ),
                )
            )
        if "stroke_sample" not in document:
            document["stroke_sample"] = copy.deepcopy(DEFAULT_STROKE_SAMPLE)
            actions.append(
                MigrationAction(
                    from_version=1,
                    to_version=3,
                    field="stroke_sample",
                    operation="default",
                    value=copy.deepcopy(DEFAULT_STROKE_SAMPLE),
                    rationale=(
                        "v1 used the fixed legacy flattener; the v3 default sample "
                        "table is numerically identical to that flattener"
                    ),
                )
            )
        if "layer_order" not in document:
            layer_order = _derive_layer_order(document)
            document["layer_order"] = layer_order
            actions.append(
                MigrationAction(
                    from_version=1,
                    to_version=3,
                    field="layer_order",
                    operation="derive_first_appearance",
                    value=layer_order,
                    rationale=(
                        "v1 stored shapes in draw order; first layer appearance "
                        "reproduces the original composition order"
                    ),
                )
            )
        document["schema_version"] = 3
        # A partially present style/sample block is still validated by the
        # strict current-schema codec below.
    elif source_version == 2:
        require(
            "style" in document,
            "E_SCHEMA_VERSION",
            "v2 document is missing its required style block",
        )
        if "stroke_sample" not in document:
            document["stroke_sample"] = copy.deepcopy(DEFAULT_STROKE_SAMPLE)
            actions.append(
                MigrationAction(
                    from_version=2,
                    to_version=3,
                    field="stroke_sample",
                    operation="default",
                    value=copy.deepcopy(DEFAULT_STROKE_SAMPLE),
                    rationale=(
                        "v2 omitted sampling settings; the v3 default matches the "
                        "legacy 12-step curve flattener"
                    ),
                )
            )
        if "layer_order" not in document:
            layer_order = _derive_layer_order(document)
            document["layer_order"] = layer_order
            actions.append(
                MigrationAction(
                    from_version=2,
                    to_version=3,
                    field="layer_order",
                    operation="derive_first_appearance",
                    value=layer_order,
                    rationale=(
                        "v2 had no explicit layer ordering; retain its shape draw "
                        "order as the canonical layer order"
                    ),
                )
            )
        document["schema_version"] = 3

    project = project_from_dict(document)
    after = project.content_hash()
    return MigrationResult(
        project=project,
        source_version=source_version,
        target_version=CURRENT_SCHEMA_VERSION,
        actions=actions,
        content_hash_before=None,
        content_hash_after=after,
    )


def load_project_migrated(raw: Mapping[str, Any]) -> MigrationResult:
    return migrate_document(raw)


def migrated_document(raw: Mapping[str, Any]) -> dict[str, Any]:
    return migrate_document(raw).project.to_dict()


def assert_idempotent(raw: Mapping[str, Any]) -> bool:
    first = migrate_document(raw)
    second = migrate_document(first.project.to_dict())
    return (
        not second.actions
        and first.content_hash_after == second.content_hash_after
        and canonical_hash(first.project.to_dict()) == canonical_hash(second.project.to_dict())
    )
