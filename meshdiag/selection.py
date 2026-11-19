from __future__ import annotations

import hashlib

from .errors import MeshError, diag
from .models import Instance, Service


def select_instance(
    service: Service,
    weights: dict[str, int],
    request_id: str,
    attempt_no: int,
) -> Instance:
    """Deterministic weighted selection over instance-id sorted candidates.

    The bucket is derived from SHA-256(request_id + attempt ordinal), so it is
    independent of Python's hash seed, dictionary iteration and thread order.
    Zero-weight instances occupy no bucket and can never be selected.
    """

    candidates: list[tuple[Instance, int]] = []
    for instance in sorted(service.instances, key=lambda item: item.instance_id):
        weight = weights.get(instance.instance_id, 0)
        if not isinstance(weight, int) or isinstance(weight, bool) or weight < 0:
            raise MeshError(
                "E_RULE",
                "实例权重非法",
                diagnostics=[
                    diag(
                        "E_RULE",
                        f"instance_weights.{instance.instance_id}",
                        "权重必须是非负整数",
                        weight,
                    )
                ],
            )
        if weight:
            candidates.append((instance, weight))
    total = sum(weight for _, weight in candidates)
    if total <= 0:
        raise MeshError(
            "E_RULE",
            "实例权重和为零，无法选择实例",
            diagnostics=[
                diag("E_RULE", f"services[{service.service_id}].instance_weights", "权重和为零", 0)
            ],
        )
    seed_material = f"{service.service_id}\0{request_id}\0{attempt_no}".encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big") % total
    cumulative = 0
    for instance, weight in candidates:
        cumulative += weight
        if bucket < cumulative:
            return instance
    raise AssertionError("weighted selection exhausted candidates")
