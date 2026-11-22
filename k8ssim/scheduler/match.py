"""Pure matching primitives shared by Filter and Score."""
from __future__ import annotations

from typing import Mapping, Sequence

from ..models import LabelRequirement, Taint, Toleration


def matches_requirement(labels: Mapping[str, str], req: LabelRequirement) -> bool:
    if req.operator == "In":
        return labels.get(req.key) in set(req.values)
    if req.operator == "NotIn":
        value = labels.get(req.key)
        if value is None:
            return True
        return value not in set(req.values)
    if req.operator == "Exists":
        return req.key in labels
    if req.operator == "DoesNotExist":
        return req.key not in labels
    raise ValueError(f"unknown operator {req.operator!r}")


def all_match(labels: Mapping[str, str], requirements: Sequence[LabelRequirement]) -> bool:
    return all(matches_requirement(labels, req) for req in requirements)


def toleration_covers(toleration: Toleration, taint: Taint) -> bool:
    if toleration.effect and toleration.effect != taint.effect:
        return False
    if toleration.operator == "Exists":
        # Empty key means the wildcard toleration that covers every taint.
        return toleration.key == "" or toleration.key == taint.key
    # Equal
    if toleration.key != taint.key:
        return False
    return toleration.value == taint.value


def taints_tolerated(taints: Sequence[Taint], tolerations: Sequence[Toleration]) -> bool:
    for taint in taints:
        if not any(toleration_covers(tol, taint) for tol in tolerations):
            return False
    return True
