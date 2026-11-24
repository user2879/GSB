"""Scheme evolution, reference propagation, and whole-result invalidation."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .engine import MaterializationEngine
from .errors import ColmvError
from .ledger import LedgerStore
from .model import Change, RuntimeConfig, ViewDefinition
from .replay import split_uniform


VIEW_FIELDS = ("group_by", "aggregates", "value_column", "filters", "refresh_mode")


def _engine_with_changes(
    config: RuntimeConfig,
    changes: Sequence[Change],
    *,
    budget: int,
) -> MaterializationEngine:
    engine = MaterializationEngine(config, budget=budget, path_mode="resident")
    if changes:
        for batch_no, batch in enumerate(split_uniform(changes, 1024), start=1):
            engine.process_batch(batch, batch_no=batch_no)
    return engine


def _view_diff(old_view: ViewDefinition | None, new_view: ViewDefinition | None) -> list[str]:
    if old_view is None:
        return ["view_added"]
    if new_view is None:
        return ["view_removed"]
    old_payload = old_view.to_dict()
    new_payload = new_view.to_dict()
    return [field for field in VIEW_FIELDS if old_payload.get(field) != new_payload.get(field)]


def _group_sort_key(group: Mapping[str, Any]) -> tuple[str, str]:
    from .canonical import canonical_json

    return (str(group["view_id"]), canonical_json(group.get("group_key_typed", group.get("group_key"))))


def build_invalidation_plan(
    *,
    old_config: RuntimeConfig,
    new_config: RuntimeConfig,
    old_budget: int,
    new_budget: int,
    changes: Sequence[Change] = (),
) -> dict[str, Any]:
    if old_budget <= 0 or new_budget <= 0:
        raise ColmvError(
            "E_BUDGET",
            "evolution budgets must be positive integers",
            field="budget",
            details={"old_budget": old_budget, "new_budget": new_budget},
        )
    old_engine = _engine_with_changes(old_config, changes, budget=old_budget)
    new_engine = _engine_with_changes(new_config, changes, budget=new_budget)
    old_index = old_engine.collect_reference_index()
    new_index = new_engine.collect_reference_index()
    old_groups = {_group_sort_key(group): group for group in old_index["groups"]}
    new_groups = {_group_sort_key(group): group for group in new_index["groups"]}

    old_view_map = old_config.view_map
    new_view_map = new_config.view_map
    all_view_ids = sorted(set(old_view_map) | set(new_view_map))
    steps: list[dict[str, Any]] = []
    changed_views: dict[str, list[str]] = {}
    for view_id in all_view_ids:
        changed_fields = _view_diff(old_view_map.get(view_id), new_view_map.get(view_id))
        if changed_fields:
            changed_views[view_id] = changed_fields
            steps.append(
                {
                    "step": len(steps) + 1,
                    "kind": "view_definition_diff",
                    "view_id": view_id,
                    "changed_fields": changed_fields,
                    "basis": "canonical normalized view definition comparison",
                }
            )

    budget_changed = old_budget != new_budget
    if budget_changed:
        steps.append(
            {
                "step": len(steps) + 1,
                "kind": "budget_diff",
                "old_budget": old_budget,
                "new_budget": new_budget,
                "basis": "global slot budget participates in the scheme fingerprint and residency trace",
            }
        )

    affected_group_map: dict[tuple[str, str], dict[str, Any]] = {}
    for view_id, changed_fields in sorted(changed_views.items()):
        for identity, group in old_groups.items():
            if identity[0] == view_id:
                entry = affected_group_map.setdefault(
                    identity,
                    {**group, "affected_by": set(), "pages": set()},
                )
                entry["affected_by"].add("old_view_definition")
                entry["pages"].update(group.get("pages", []))
        for identity, group in new_groups.items():
            if identity[0] == view_id:
                entry = affected_group_map.setdefault(
                    identity,
                    {**group, "affected_by": set(), "pages": set()},
                )
                entry["affected_by"].add("new_view_definition")
                entry["pages"].update(group.get("pages", []))
        steps.append(
            {
                "step": len(steps) + 1,
                "kind": "group_propagation",
                "view_id": view_id,
                "changed_fields": changed_fields,
                "affected_group_count": sum(1 for key in affected_group_map if key[0] == view_id),
                "basis": "current matching rows -> non-null group keys -> group references",
            }
        )

    if budget_changed and not changed_views:
        for identity, group in sorted({**old_groups, **new_groups}.items()):
            entry = affected_group_map.setdefault(
                identity,
                {**group, "affected_by": set(), "pages": set()},
            )
            entry["affected_by"].add("budget_change")
            entry["pages"].update(group.get("pages", []))
        steps.append(
            {
                "step": len(steps) + 1,
                "kind": "group_propagation",
                "view_id": "<all>",
                "affected_group_count": len(affected_group_map),
                "basis": "budget-only evolution invalidates all group residency traces",
            }
        )
    elif budget_changed and changed_views:
        for entry in affected_group_map.values():
            entry["affected_by"].add("budget_change")
        steps.append(
            {
                "step": len(steps) + 1,
                "kind": "group_propagation",
                "view_id": "<affected_views>",
                "affected_group_count": len(affected_group_map),
                "basis": "budget change is combined with the affected view group set",
            }
        )

    affected_pages: dict[str, dict[str, Any]] = {}
    for identity, entry in sorted(affected_group_map.items()):
        for page_id in sorted(entry["pages"]):
            page_entry = affected_pages.setdefault(
                page_id,
                {"page_id": page_id, "referenced_by_groups": set()},
            )
            page_entry["referenced_by_groups"].add(identity)
    if affected_pages:
        steps.append(
            {
                "step": len(steps) + 1,
                "kind": "page_propagation",
                "affected_page_count": len(affected_pages),
                "basis": "affected group -> current row references -> columnar page ids",
            }
        )

    affected_groups_payload = []
    for identity, entry in sorted(affected_group_map.items()):
        affected_groups_payload.append(
            {
                "view_id": entry["view_id"],
                "group_key": entry.get("group_key"),
                "group_key_typed": entry.get("group_key_typed"),
                "pages": sorted(entry["pages"]),
                "basis": sorted(entry["affected_by"]),
            }
        )
    affected_pages_payload = [
        {
            "page_id": entry["page_id"],
            "referenced_by_groups": [
                {"view_id": identity[0], "group_sort": identity[1]}
                for identity in sorted(entry["referenced_by_groups"])
            ],
        }
        for entry in sorted(affected_pages.values(), key=lambda item: item["page_id"])
    ]
    return {
        "format_version": 1,
        "old_budget": old_budget,
        "new_budget": new_budget,
        "budget_changed": budget_changed,
        "changed_views": [
            {"view_id": view_id, "changed_fields": fields}
            for view_id, fields in sorted(changed_views.items())
        ],
        "steps": steps,
        "affected_groups": affected_groups_payload,
        "affected_pages": affected_pages_payload,
        "replacement": {
            "mode": "whole_scheme_replacement",
            "old_traces": "stale",
            "partial_group_patch_forbidden": True,
        },
    }


def apply_evolution(
    ledger: LedgerStore,
    *,
    new_config: RuntimeConfig,
    new_budget: int,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    current = ledger.current_scheme()
    old_version = int(current["scheme_version"]) if current is not None else None
    if not plan["changed_views"] and not plan["budget_changed"]:
        return {
            "old_scheme_version": old_version,
            "new_scheme_version": old_version,
            "scheme_version_changed": False,
            "old_traces_marked_stale": False,
            "snapshot_action": "unchanged",
        }
    new_version = ledger.register_scheme(new_config, budget=new_budget, invalidation=plan)
    return {
        "old_scheme_version": old_version,
        "new_scheme_version": new_version,
        "scheme_version_changed": old_version != new_version,
        "monotonic_increment": old_version is None or new_version > old_version,
        "old_traces_marked_stale": old_version is not None,
        "snapshot_action": "cleared_until_whole_recompute",
        "current_conclusions_visible": False,
        "requires_recompute": True,
    }
