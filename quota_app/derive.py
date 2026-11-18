from __future__ import annotations

from typing import Any

from .errors import DeriveError
from .models import ConfigDocument, EffectiveVolume


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def resolve_effective(document: ConfigDocument) -> dict[str, EffectiveVolume]:
    """Resolve derived capacities and arbitrate explicit/derived conflicts.

    Arbitration rule: the effective capacity is the smaller of the explicit
    capacity and the value derived from the upstream chain.  This is the
    conservative intersection of both declared constraints.  The basis and
    arbitration winner are retained on every effective volume.
    """

    volumes = document.volume_map
    for volume in document.volumes:
        if not _is_int(volume.capacity_bytes) or volume.capacity_bytes < 0:
            raise DeriveError(
                "capacity_bytes must be a non-negative integer",
                volume_id=volume.volume_id,
                field_path=f"volume:{volume.volume_id}.capacity_bytes",
            )
        derivation = volume.derivation
        if derivation is None:
            continue
        if not _is_int(derivation.factor) or not _is_int(derivation.offset):
            raise DeriveError(
                "derivation factor and offset must be integers",
                volume_id=volume.volume_id,
                field_path=f"volume:{volume.volume_id}.derivation",
            )
        if derivation.factor < 0 or derivation.offset < 0:
            raise DeriveError(
                "derivation factor and offset must not be negative",
                volume_id=volume.volume_id,
                field_path=f"volume:{volume.volume_id}.derivation",
            )

    states: dict[str, str] = {}
    stack: list[str] = []
    resolved: dict[str, EffectiveVolume] = {}

    def resolve(volume_id: str) -> EffectiveVolume:
        state = states.get(volume_id)
        if state == "done":
            return resolved[volume_id]
        if state == "visiting":
            cycle_start = stack.index(volume_id)
            cycle = stack[cycle_start:] + [volume_id]
            raise DeriveError(
                f"derivation cycle detected: {' -> '.join(cycle)}",
                volume_id=volume_id,
                field_path=f"volume:{volume_id}.derivation.upstream_id",
                details={"cycle": cycle},
            )

        volume = volumes[volume_id]
        states[volume_id] = "visiting"
        stack.append(volume_id)
        derivation = volume.derivation
        derived_capacity: int | None = None
        chain: tuple[str, ...] = (volume_id,)
        if derivation is not None:
            upstream_id = derivation.upstream_id
            if upstream_id not in volumes:
                reference_chain = [volume_id, upstream_id]
                raise DeriveError(
                    f"derivation references missing volume {upstream_id!r}",
                    volume_id=volume_id,
                    field_path=f"volume:{volume_id}.derivation.upstream_id",
                    details={"reference_chain": reference_chain},
                )
            upstream = resolve(upstream_id)
            derived_capacity = upstream.capacity_bytes * derivation.factor + derivation.offset
            chain = upstream.derivation_chain + (volume_id,)

        if derived_capacity is None:
            effective_capacity = volume.capacity_bytes
            arbitration = "explicit"
            basis = f"explicit capacity_bytes={volume.capacity_bytes}"
        else:
            if derived_capacity < volume.capacity_bytes:
                arbitration = "derived"
                basis = (
                    f"min(explicit={volume.capacity_bytes}, derived={derived_capacity}); "
                    f"derived wins"
                )
            elif volume.capacity_bytes < derived_capacity:
                arbitration = "explicit"
                basis = (
                    f"min(explicit={volume.capacity_bytes}, derived={derived_capacity}); "
                    f"explicit wins"
                )
            else:
                arbitration = "equal"
                basis = f"explicit={volume.capacity_bytes} equals derived={derived_capacity}"
            effective_capacity = min(volume.capacity_bytes, derived_capacity)

        effective = EffectiveVolume(
            volume_id=volume_id,
            capacity_bytes=effective_capacity,
            explicit_capacity_bytes=volume.capacity_bytes,
            derived_capacity_bytes=derived_capacity,
            capacity_basis=basis,
            arbitration=arbitration,
            derivation_chain=chain,
            ops_per_sec=volume.ops_per_sec,
            burst_capacity=volume.burst_capacity,
            limit_dimension=volume.limit_dimension,
        )
        stack.pop()
        states[volume_id] = "done"
        resolved[volume_id] = effective
        return effective

    for volume in document.volumes:
        resolve(volume.volume_id)
    return resolved


def changed_fields(
    old_document: ConfigDocument | None,
    new_document: ConfigDocument,
) -> list[dict[str, Any]]:
    """Return a stable, volume-sorted description of configuration changes."""

    old_volumes = old_document.volume_map if old_document is not None else {}
    new_volumes = new_document.volume_map
    old_effective = resolve_effective(old_document) if old_document is not None else {}
    new_effective = resolve_effective(new_document)
    changes: list[dict[str, Any]] = []
    for volume_id in sorted(set(old_volumes) | set(new_volumes)):
        old = old_volumes.get(volume_id)
        new = new_volumes.get(volume_id)
        fields: list[str] = []
        if old is None:
            fields = ["added"]
        elif new is None:
            fields = ["removed"]
        else:
            for field_name in (
                "capacity_bytes",
                "ops_per_sec",
                "burst_capacity",
                "limit_dimension",
                "derivation",
            ):
                old_value = getattr(old, field_name)
                new_value = getattr(new, field_name)
                if old_value != new_value:
                    fields.append(field_name)
            old_eff = old_effective[volume_id]
            new_eff = new_effective[volume_id]
            if old_eff.capacity_bytes != new_eff.capacity_bytes:
                fields.append("effective_capacity_bytes")
            if old_eff.arbitration != new_eff.arbitration:
                fields.append("arbitration")
            if old_eff.derivation_chain != new_eff.derivation_chain:
                fields.append("derivation_chain")
        if fields:
            changes.append({"volume_id": volume_id, "fields": fields})
    return changes
