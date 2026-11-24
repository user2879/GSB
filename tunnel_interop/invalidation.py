"""Precise dependency-based invalidation and selective recomputation."""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .fragmenter import (
    FragmentBundle,
    MessageBundle,
    fragment_message,
    reassemble_fragments,
)
from .samples import MessagePlan, generate_messages
from .spec import Route, TunnelSpec, validate_versions_monotonic


@dataclass(frozen=True)
class PropagationStep:
    stage: str
    source: str
    target: str
    reason: str
    path: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {
            "stage": self.stage,
            "source": self.source,
            "target": self.target,
            "reason": self.reason,
            "path": self.path,
            "detail": self.detail,
        }


@dataclass
class InvalidationResult:
    old_versions: dict[str, int]
    new_versions: dict[str, int]
    invalid_fragments: tuple[dict[str, Any], ...]
    invalid_reassemblies: tuple[str, ...]
    removed_messages: tuple[str, ...]
    added_messages: tuple[str, ...]
    recomputed_fragments: tuple[dict[str, Any], ...]
    preserved_fragments: tuple[dict[str, Any], ...]
    steps: tuple[PropagationStep, ...]
    bundle: FragmentBundle
    unchanged_checks: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return {
            "old_versions": self.old_versions,
            "new_versions": self.new_versions,
            "invalid_fragment_count": len(self.invalid_fragments),
            "invalid_fragments": list(self.invalid_fragments),
            "invalid_reassemblies": list(self.invalid_reassemblies),
            "removed_messages": list(self.removed_messages),
            "added_messages": list(self.added_messages),
            "recomputed_fragment_count": len(self.recomputed_fragments),
            "recomputed_fragments": list(self.recomputed_fragments),
            "preserved_fragment_count": len(self.preserved_fragments),
            "preserved_fragments": list(self.preserved_fragments),
            "propagation_steps": [step.to_dict() for step in self.steps],
            "unchanged_checks": self.unchanged_checks,
        }


def _message_id(plan: MessagePlan) -> str:
    return plan.message_id


def _route_identity(route: Route) -> str:
    return route.tunnel_id


def _diff_mapping(
    old: Mapping[str, Mapping[str, Any]],
    new: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], list[str], list[tuple[str, list[str]]]]:
    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    changed: list[tuple[str, list[str]]] = []
    for key in sorted(set(old) & set(new)):
        fields = sorted(
            field
            for field in set(old[key]) | set(new[key])
            if old[key].get(field) != new[key].get(field)
        )
        if fields:
            changed.append((key, fields))
    return added, removed, changed


def _select_route(
    routes: Iterable[Route],
    destination: ipaddress.IPv4Network | ipaddress.IPv6Network,
) -> Route | None:
    candidates = [
        route
        for route in routes
        if route.network.version == destination.version
        and destination.network_address in route.network
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda route: (
            -route.network.prefixlen,
            route.priority,
            route.tunnel_id,
            route.next_hop,
        ),
    )[0]


class _Propagation:
    def __init__(self) -> None:
        self.steps: list[PropagationStep] = []
        self._seen: set[tuple[str, str, str, str]] = set()

    def add(self, step: PropagationStep) -> None:
        key = (step.stage, step.source, step.target, step.reason)
        if key not in self._seen:
            self._seen.add(key)
            self.steps.append(step)


def evolve_bundle(
    old_bundle: FragmentBundle,
    *,
    old_spec: TunnelSpec | None = None,
    new_spec: TunnelSpec,
    new_plans: Iterable[MessagePlan] | None = None,
) -> InvalidationResult:
    old_spec = old_spec or old_bundle.spec
    validate_versions_monotonic(old_spec, new_spec)
    plans = tuple(new_plans) if new_plans is not None else generate_messages(new_spec)
    new_plan_map = {plan.message_key: plan for plan in plans}
    old_bundle_map = old_bundle.message_map
    old_keys = set(old_bundle_map)
    new_keys = set(new_plan_map)

    propagation = _Propagation()
    invalid_fragments: set[str] = set()
    invalid_reassemblies: set[str] = set()
    removed_messages = tuple(sorted(old_keys - new_keys))
    added_messages = tuple(sorted(new_keys - old_keys))

    def invalidate(message_key: str, *, fragments: bool, source: str, reason: str, path: str, detail: str) -> None:
        if message_key in old_keys:
            if fragments:
                invalid_fragments.add(message_key)
            invalid_reassemblies.add(message_key)
            propagation.add(
                PropagationStep(
                    stage="message",
                    source=source,
                    target=f"message:{message_key}",
                    reason=reason,
                    path=path,
                    detail=detail,
                )
            )

    old_endpoints = old_spec.endpoint_map
    new_endpoints = new_spec.endpoint_map
    old_endpoint_entries = {key: value.canonical() for key, value in old_endpoints.items()}
    new_endpoint_entries = {key: value.canonical() for key, value in new_endpoints.items()}
    endpoint_added, endpoint_removed, endpoint_changed = _diff_mapping(
        old_endpoint_entries, new_endpoint_entries
    )

    old_routes = old_spec.route_map
    new_routes = new_spec.route_map
    route_added, route_removed, route_changed = _diff_mapping(
        {key: value.canonical() for key, value in old_routes.items()},
        {key: value.canonical() for key, value in new_routes.items()},
    )

    for endpoint_id in endpoint_removed:
        propagation.add(
            PropagationStep(
                stage="endpoint",
                source=f"endpoint:{endpoint_id}",
                target="endpoint_set",
                reason="ENDPOINT_REMOVED",
                path=f"$.endpoints[id={endpoint_id}]",
                detail="endpoint was removed",
            )
        )
    for endpoint_id in endpoint_added:
        propagation.add(
            PropagationStep(
                stage="endpoint",
                source=f"endpoint:{endpoint_id}",
                target="endpoint_set",
                reason="ENDPOINT_ADDED",
                path=f"$.endpoints[id={endpoint_id}]",
                detail="endpoint was added",
            )
        )

    fragment_relevant_endpoint_fields = {"mtu", "encapsulation_overhead", "seed"}
    for endpoint_id, changed_fields in endpoint_changed:
        directly_fragment_relevant = bool(set(changed_fields) & fragment_relevant_endpoint_fields)
        affected_routes = sorted(
            {
                route.tunnel_id
                for route in list(old_routes.values()) + list(new_routes.values())
                if route.next_hop == endpoint_id
            }
        )
        propagation.add(
            PropagationStep(
                stage="endpoint",
                source=f"endpoint:{endpoint_id}",
                target="route_table",
                reason="ENDPOINT_CHANGED",
                path=f"$.endpoints[id={endpoint_id}]",
                detail="changed fields: " + ",".join(changed_fields),
            )
        )
        for route_id in affected_routes:
            propagation.add(
                PropagationStep(
                    stage="route",
                    source=f"route:{route_id}",
                    target=f"endpoint:{endpoint_id}",
                    reason="NEXT_HOP_ENDPOINT_CHANGED",
                    path=f"$.routes[tunnel_id={route_id}].next_hop",
                    detail="route references a changed endpoint",
                )
            )
            invalidate(
                route_id,
                fragments=directly_fragment_relevant,
                source=f"endpoint:{endpoint_id}",
                reason="ENDPOINT_CHANGED",
                path=f"$.endpoints[id={endpoint_id}]",
                detail="changed fields: " + ",".join(changed_fields),
            )

    for route_id in route_removed:
        propagation.add(
            PropagationStep(
                stage="route",
                source=f"route:{route_id}",
                target=f"message:{route_id}",
                reason="ROUTE_REMOVED",
                path=f"$.routes[tunnel_id={route_id}]",
                detail="route and its existing message/fragments were removed",
            )
        )
    for route_id in route_added:
        propagation.add(
            PropagationStep(
                stage="route",
                source=f"route:{route_id}",
                target=f"message:{route_id}",
                reason="ROUTE_ADDED",
                path=f"$.routes[tunnel_id={route_id}]",
                detail="new route requires an initial fragment generation",
            )
        )

    for route_id, changed_fields in route_changed:
        propagation.add(
            PropagationStep(
                stage="route",
                source=f"route:{route_id}",
                target=f"message:{route_id}",
                reason="ROUTE_CHANGED",
                path=f"$.routes[tunnel_id={route_id}]",
                detail="changed fields: " + ",".join(changed_fields),
            )
        )
        invalidate(
            route_id,
            fragments=True,
            source=f"route:{route_id}",
            reason="ROUTE_CHANGED",
            path=f"$.routes[tunnel_id={route_id}]",
            detail="changed fields: " + ",".join(changed_fields),
        )

    old_policy = old_spec.fragment_policy
    new_policy = new_spec.fragment_policy
    policy_changes = [
        field
        for field in ("fragment_size", "max_retries", "reassembly_timeout_ms")
        if getattr(old_policy, field) != getattr(new_policy, field)
    ]
    for field in policy_changes:
        fragments_changed = field in {"fragment_size", "max_retries"}
        propagation.add(
            PropagationStep(
                stage="fragment_policy",
                source=f"fragment_policy.{field}",
                target="fragment_policy",
                reason="FRAGMENT_POLICY_CHANGED",
                path=f"$.fragment_policy.{field}",
                detail=f"old={getattr(old_policy, field)} new={getattr(new_policy, field)}",
            )
        )
        for message_key in sorted(new_keys & old_keys):
            invalidate(
                message_key,
                fragments=fragments_changed,
                source=f"fragment_policy.{field}",
                reason="FRAGMENT_POLICY_CHANGED",
                path=f"$.fragment_policy.{field}",
                detail=f"fragment-affecting={fragments_changed}",
            )

    if old_spec.samples.canonical() != new_spec.samples.canonical():
        propagation.add(
            PropagationStep(
                stage="sample",
                source="sample",
                target="all_messages",
                reason="SAMPLE_POLICY_CHANGED",
                path="$.sample",
                detail="deterministic inner payload size source changed",
            )
        )
        for message_key in sorted(new_keys & old_keys):
            invalidate(
                message_key,
                fragments=True,
                source="sample",
                reason="SAMPLE_POLICY_CHANGED",
                path="$.sample",
                detail="inner sample length may change",
            )

    if old_spec.runtime.fragment_format != new_spec.runtime.fragment_format:
        propagation.add(
            PropagationStep(
                stage="runtime",
                source="runtime.fragment_format",
                target="all_messages",
                reason="FRAGMENT_FORMAT_CHANGED",
                path="$.runtime.fragment_format",
                detail="wire format changed",
            )
        )
        for message_key in sorted(new_keys & old_keys):
            invalidate(
                message_key,
                fragments=True,
                source="runtime.fragment_format",
                reason="FRAGMENT_FORMAT_CHANGED",
                path="$.runtime.fragment_format",
                detail="target wire format changed",
            )

    old_route_list = tuple(old_routes.values())
    new_route_list = tuple(new_routes.values())
    for destination in sorted(
        {plan.route.network for plan in plans},
        key=lambda network: (network.version, int(network.network_address), network.prefixlen),
    ):
        old_selected = _select_route(old_route_list, destination)
        new_selected = _select_route(new_route_list, destination)
        old_id = old_selected.tunnel_id if old_selected else None
        new_id = new_selected.tunnel_id if new_selected else None
        if old_id != new_id:
            propagation.add(
                PropagationStep(
                    stage="route_selection",
                    source=f"destination:{destination}",
                    target=f"message:{new_id}",
                    reason="ROUTE_SELECTION_CHANGED",
                    path="$.routes",
                    detail=f"old={old_id} new={new_id}",
                )
            )
            if new_id is not None:
                invalidate(
                    new_id,
                    fragments=True,
                    source=f"destination:{destination}",
                    reason="ROUTE_SELECTION_CHANGED",
                    path="$.routes",
                    detail=f"old={old_id} new={new_id}",
                )
            if old_id is not None:
                invalidate(
                    old_id,
                    fragments=True,
                    source=f"destination:{destination}",
                    reason="ROUTE_SELECTION_CHANGED",
                    path="$.routes",
                    detail=f"old={old_id} new={new_id}",
                )

    for message_key in sorted(new_keys & old_keys):
        old_plan = old_bundle_map[message_key].plan
        new_plan = new_plan_map[message_key]
        differences: list[str] = []
        if old_plan.payload_sha256 != new_plan.payload_sha256:
            differences.append("payload_sha256")
        if old_plan.message_id != new_plan.message_id:
            differences.append("message_id")
        if old_plan.fragment_size != new_plan.fragment_size:
            differences.append("fragment_size")
        if old_plan.route.next_hop != new_plan.route.next_hop:
            differences.append("next_hop")
        if differences:
            propagation.add(
                PropagationStep(
                    stage="plan",
                    source=f"plan:{message_key}",
                    target=f"message:{message_key}",
                    reason="PLAN_DERIVATION_CHANGED",
                    path="$.routes",
                    detail="changed derived fields: " + ",".join(differences),
                )
            )
            invalidate(
                message_key,
                fragments=True,
                source=f"plan:{message_key}",
                reason="PLAN_DERIVATION_CHANGED",
                path="$.routes",
                detail="changed derived fields: " + ",".join(differences),
            )

    target_format = new_spec.runtime.fragment_format
    old_format = old_bundle.target_format
    result_messages: list[MessageBundle] = []
    recomputed: list[dict[str, Any]] = []
    preserved: list[dict[str, Any]] = []

    for message_key in sorted(new_keys):
        new_plan = new_plan_map[message_key]
        if message_key in added_messages:
            message_bundle = fragment_message(new_plan, target_format=target_format)
            recomputed.extend(
                {
                    "message_key": message_key,
                    "fragment_index": fragment.fragment_index,
                    "reason": "ADDED_MESSAGE",
                }
                for fragment in message_bundle.fragments
            )
        elif message_key in old_bundle_map and message_key not in invalid_fragments:
            old_message = old_bundle_map[message_key]
            if message_key in invalid_reassemblies:
                reassembly = reassemble_fragments(
                    [fragment.normalized for fragment in old_message.fragments],
                    expected_message_key=message_key,
                    expected_sha256=new_plan.payload_sha256,
                    expected_total=len(old_message.fragments),
                    reassembly_version=old_message.reassembly.reassembly_version + 1,
                )
                message_bundle = MessageBundle(
                    plan=new_plan,
                    fragments=old_message.fragments,
                    reassembly=reassembly,
                )
            else:
                message_bundle = MessageBundle(
                    plan=new_plan,
                    fragments=old_message.fragments,
                    reassembly=old_message.reassembly,
                )
            preserved.extend(
                {
                    "message_key": message_key,
                    "fragment_index": fragment.fragment_index,
                    "wire_sha256": fragment.wire_sha256,
                    "fragment_version": fragment.fragment_version,
                }
                for fragment in old_message.fragments
            )
        else:
            old_message = old_bundle_map.get(message_key)
            versions: dict[int, int] = {}
            if old_message is not None:
                for fragment in old_message.fragments:
                    versions[fragment.fragment_index] = fragment.fragment_version + 1
            reassembly_version = (
                old_message.reassembly.reassembly_version + 1 if old_message is not None else 1
            )
            message_bundle = fragment_message(
                new_plan,
                target_format=target_format,
                fragment_versions=versions,
                reassembly_version=reassembly_version,
            )
            recomputed.extend(
                {
                    "message_key": message_key,
                    "fragment_index": fragment.fragment_index,
                    "reason": "INVALIDATED",
                }
                for fragment in message_bundle.fragments
            )
        result_messages.append(message_bundle)

    new_bundle_map = {item.plan.message_key: item for item in result_messages}
    unchanged_wire = True
    unchanged_version = True
    for preserved_item in preserved:
        key = preserved_item["message_key"]
        if key not in new_bundle_map:
            unchanged_wire = False
            break
        matches = [
            fragment
            for fragment in new_bundle_map[key].fragments
            if fragment.fragment_index == preserved_item["fragment_index"]
        ]
        if not matches:
            unchanged_wire = False
            break
        fragment = matches[0]
        if fragment.wire_sha256 != preserved_item["wire_sha256"]:
            unchanged_wire = False
        if fragment.fragment_version != preserved_item["fragment_version"]:
            unchanged_version = False

    invalid_fragment_list = tuple(
        {"message_key": message_key, "fragment_index": fragment.fragment_index}
        for message_key in sorted(invalid_fragments)
        for fragment in old_bundle_map[message_key].fragments
    )
    invalid_reassembly_list = tuple(sorted(invalid_reassemblies))
    steps = tuple(propagation.steps)
    return InvalidationResult(
        old_versions=old_spec.versions.canonical(),
        new_versions=new_spec.versions.canonical(),
        invalid_fragments=invalid_fragment_list,
        invalid_reassemblies=invalid_reassembly_list,
        removed_messages=removed_messages,
        added_messages=added_messages,
        recomputed_fragments=tuple(recomputed),
        preserved_fragments=tuple(preserved),
        steps=steps,
        bundle=FragmentBundle(
            spec=new_spec,
            target_format=target_format,
            messages=tuple(result_messages),
            declaration_fingerprint=new_spec.declaration_fingerprint(),
        ),
        unchanged_checks={
            "unaffected_wire_sha256_unchanged": unchanged_wire,
            "unaffected_fragment_version_unchanged": unchanged_version,
        },
    )
