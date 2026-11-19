"""Deterministic HTTP/2 priority tree and fairness allocations."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable

from .errors import DeclarationError, FrameError


@dataclass
class PriorityNode:
    stream_id: int
    parent_id: int
    weight: int


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


class PriorityTree:
    """A small, ordered dependency tree rooted at stream 0."""

    def __init__(self, nodes: Iterable[PriorityNode]) -> None:
        self._nodes: dict[int, PriorityNode] = {}
        for node in sorted(nodes, key=lambda item: item.stream_id):
            if node.stream_id in self._nodes:
                raise ValueError(f"duplicate priority stream {node.stream_id}")
            self._nodes[node.stream_id] = PriorityNode(node.stream_id, node.parent_id, node.weight)
        for stream_id in sorted(self._nodes):
            node = self._nodes[stream_id]
            self._validate_parent(stream_id, node.parent_id)
        self._assert_acyclic()

    def copy(self) -> "PriorityTree":
        return PriorityTree(
            PriorityNode(node.stream_id, node.parent_id, node.weight)
            for node in self._nodes.values()
        )

    @property
    def stream_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._nodes))

    def node(self, stream_id: int) -> PriorityNode:
        return self._nodes[stream_id]

    def parent_of(self, stream_id: int) -> int:
        return self._nodes[stream_id].parent_id

    def weight_of(self, stream_id: int) -> int:
        return self._nodes[stream_id].weight

    def children_of(self, stream_id: int) -> list[int]:
        return sorted(
            candidate
            for candidate, node in self._nodes.items()
            if node.parent_id == stream_id and candidate != stream_id
        )

    def _validate_parent(self, stream_id: int, parent_id: int) -> None:
        if parent_id == stream_id:
            raise DeclarationError("stream cannot depend on itself", f"$.requests[{stream_id}].parent_id")
        if parent_id and parent_id not in self._nodes:
            raise DeclarationError(
                f"priority parent {parent_id} does not exist",
                f"$.requests[{stream_id}].parent_id",
            )

    def _is_descendant(self, candidate: int, ancestor: int) -> bool:
        current = candidate
        seen: set[int] = set()
        while current:
            if current == ancestor:
                return True
            if current in seen:
                return True
            seen.add(current)
            current = self._nodes[current].parent_id
        return False

    def _assert_acyclic(self) -> None:
        for stream_id in sorted(self._nodes):
            current = stream_id
            path: set[int] = set()
            while current:
                if current in path:
                    raise DeclarationError("priority dependency graph contains a cycle", "$.requests")
                path.add(current)
                current = self._nodes[current].parent_id

    def update(
        self,
        stream_id: int,
        *,
        parent_id: int | None = None,
        weight: int | None = None,
        exclusive: bool = False,
        active_streams: Iterable[int] | None = None,
    ) -> dict[str, object]:
        if stream_id not in self._nodes:
            raise FrameError(
                f"priority update targets unknown stream {stream_id}",
                frame_type="HEADERS",
                offset=0,
                stream_id=stream_id,
            )
        node = self._nodes[stream_id]
        old_parent = node.parent_id
        old_weight = node.weight
        new_parent = old_parent if parent_id is None else parent_id
        new_weight = old_weight if weight is None else weight
        if not 1 <= new_weight <= 256:
            raise FrameError(
                "priority weight must be 1..256",
                frame_type="HEADERS",
                offset=0,
                stream_id=stream_id,
            )
        rehang_reason = "explicit"
        active = set(active_streams or ())
        if new_parent == stream_id:
            raise FrameError(
                "priority update would make a stream depend on itself",
                frame_type="HEADERS",
                offset=0,
                stream_id=stream_id,
            )
        if new_parent and new_parent not in self._nodes:
            raise FrameError(
                f"priority parent {new_parent} does not exist",
                frame_type="HEADERS",
                offset=0,
                stream_id=stream_id,
            )
        if new_parent and self._is_descendant(new_parent, stream_id):
            raise FrameError(
                "priority update would create a dependency cycle",
                frame_type="HEADERS",
                offset=0,
                stream_id=stream_id,
            )
        if new_parent and active and new_parent not in active:
            new_parent = 0
            rehang_reason = "parent_inactive"
        exclusive_children = [
            child
            for child in self.children_of(new_parent)
            if child != stream_id
        ] if exclusive else []
        node.parent_id = new_parent
        node.weight = new_weight
        for child in exclusive_children:
            self._nodes[child].parent_id = stream_id
        return {
            "stream_id": stream_id,
            "old_parent_id": old_parent,
            "new_parent_id": new_parent,
            "old_weight": old_weight,
            "new_weight": new_weight,
            "exclusive": exclusive,
            "reparented_child_ids": exclusive_children,
            "reason": rehang_reason,
        }

    def reparent_children(self, closed_stream_id: int) -> list[dict[str, object]]:
        if closed_stream_id not in self._nodes:
            return []
        new_parent = self._nodes[closed_stream_id].parent_id
        changes: list[dict[str, object]] = []
        for child in self.children_of(closed_stream_id):
            old_parent = self._nodes[child].parent_id
            self._nodes[child].parent_id = new_parent
            changes.append(
                {
                    "stream_id": child,
                    "old_parent_id": old_parent,
                    "new_parent_id": new_parent,
                    "reason": "parent_rst_stream",
                }
            )
        return changes

    def shares(self) -> dict[int, Fraction]:
        """Return each stream's stable share of root bandwidth.

        A stream competes with its direct children using its own weight. This
        makes a parent's data rate and its children's combined rate follow the
        same sibling-weight rule, recursively.
        """

        shares: dict[int, Fraction] = {stream_id: Fraction(0) for stream_id in self._nodes}

        def distribute(parent_id: int, amount: Fraction) -> None:
            children = self.children_of(parent_id)
            if parent_id == 0:
                participants: list[tuple[int, int]] = [
                    (child, self._nodes[child].weight) for child in children
                ]
                self_share = Fraction(0)
            else:
                self_weight = self._nodes[parent_id].weight
                participants = [(parent_id, self_weight)] + [
                    (child, self._nodes[child].weight) for child in children
                ]
                total = sum(weight for _, weight in participants)
                self_share = amount * self_weight / total
                shares[parent_id] += self_share
                remaining = amount - self_share
                if not children:
                    return
                child_total = sum(weight for _, weight in participants[1:])
                for child, weight in participants[1:]:
                    distribute(child, remaining * weight / child_total)
                return
            if not participants:
                return
            total = sum(weight for _, weight in participants)
            for child, weight in participants:
                distribute(child, amount * weight / total)

        root_children = self.children_of(0)
        if root_children:
            distribute(0, Fraction(1))
        else:
            for stream_id in self._nodes:
                shares[stream_id] = Fraction(1, len(self._nodes))
        return shares

    def to_dict(self) -> dict[str, object]:
        return {
            "nodes": [
                {
                    "stream_id": stream_id,
                    "parent_id": self._nodes[stream_id].parent_id,
                    "weight": self._nodes[stream_id].weight,
                }
                for stream_id in sorted(self._nodes)
            ]
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "PriorityTree":
        nodes_value = value.get("nodes")
        if not isinstance(nodes_value, list):
            raise ValueError("priority nodes must be a list")
        nodes: list[PriorityNode] = []
        for item in nodes_value:
            if not isinstance(item, dict):
                raise ValueError("priority node must be an object")
            nodes.append(
                PriorityNode(
                    stream_id=int(item["stream_id"]),
                    parent_id=int(item["parent_id"]),
                    weight=int(item["weight"]),
                )
            )
        return cls(nodes)

    def rows(self) -> list[dict[str, object]]:
        shares = self.shares()
        rows: list[dict[str, object]] = []

        def walk(parent_id: int, depth: int) -> None:
            for child in self.children_of(parent_id):
                share = shares.get(child, Fraction(0))
                rows.append(
                    {
                        "stream_id": child,
                        "parent_id": parent_id,
                        "weight": self._nodes[child].weight,
                        "depth": depth,
                        "share": _fraction_text(share),
                        "share_numerator": share.numerator,
                        "share_denominator": share.denominator,
                    }
                )
                walk(child, depth + 1)

        walk(0, 0)
        return rows


def fair_quotas(
    body_bytes: dict[int, int],
    shares: dict[int, Fraction],
) -> dict[int, int]:
    """Weighted max-min allocation with per-stream body caps.

    Total expected bytes equal the sum of requested body bytes (except when all
    bodies are zero). Integer rounding uses largest fractional remainder with
    stream-id tie breakers, so the result is independent of mapping order.
    """

    stream_ids = sorted(body_bytes)
    total = sum(max(0, body_bytes[stream_id]) for stream_id in stream_ids)
    quotas = {stream_id: 0 for stream_id in stream_ids}
    if total == 0:
        return quotas
    active = [stream_id for stream_id in stream_ids if body_bytes[stream_id] > 0]
    remaining = total
    while remaining > 0 and active:
        weight_total = sum((shares.get(stream_id, Fraction(0)) for stream_id in active), Fraction(0))
        if weight_total <= 0:
            weights = {stream_id: Fraction(1, len(active)) for stream_id in active}
        else:
            weights = {stream_id: shares.get(stream_id, Fraction(0)) / weight_total for stream_id in active}
        proposals: dict[int, Fraction] = {
            stream_id: remaining * weights[stream_id] for stream_id in active
        }
        added = 0
        for stream_id in active:
            capacity = body_bytes[stream_id] - quotas[stream_id]
            addition = min(int(proposals[stream_id]), capacity)
            quotas[stream_id] += addition
            added += addition
        remaining -= added
        active = [
            stream_id
            for stream_id in active
            if quotas[stream_id] < body_bytes[stream_id]
        ]
        if remaining <= 0:
            break
        if added:
            continue
        # Hamilton remainder pass. Fractional ordering and stream id make this stable.
        ranked = sorted(
            active,
            key=lambda stream_id: (-(proposals[stream_id] - int(proposals[stream_id])), stream_id),
        )
        for stream_id in ranked:
            if remaining <= 0:
                break
            if quotas[stream_id] < body_bytes[stream_id]:
                quotas[stream_id] += 1
                remaining -= 1
        active = [
            stream_id
            for stream_id in active
            if quotas[stream_id] < body_bytes[stream_id]
        ]
    return quotas


def quota_rows(
    tree: PriorityTree,
    body_bytes: dict[int, int],
    actual_bytes: dict[int, int],
) -> list[dict[str, object]]:
    shares = tree.shares()
    quotas = fair_quotas(body_bytes, shares)
    rows: list[dict[str, object]] = []
    for stream_id in sorted(body_bytes):
        share = shares.get(stream_id, Fraction(0))
        actual = actual_bytes.get(stream_id, 0)
        expected = quotas.get(stream_id, 0)
        rows.append(
            {
                "stream_id": stream_id,
                "parent_id": tree.parent_of(stream_id),
                "weight": tree.weight_of(stream_id),
                "share": _fraction_text(share),
                "share_numerator": share.numerator,
                "share_denominator": share.denominator,
                "expected_bytes": expected,
                "actual_bytes": actual,
                "deviation": actual - expected,
            }
        )
    return rows
