from __future__ import annotations

from collections.abc import Iterable

from .errors import GenerationChainError


def validate_generation_edge(lease_id: str, before: int, after: int) -> None:
    if after != before + 1:
        raise GenerationChainError(
            "generation edge must advance by exactly one",
            path=f"$.leases[?lease_id={lease_id}].generation",
            details={
                "lease_id": lease_id,
                "generation_before": before,
                "generation_after": after,
            },
        )


def audit_generation_chain(lease_id: str, generations: Iterable[int]) -> list[int]:
    chain = list(generations)
    for index in range(1, len(chain)):
        before = chain[index - 1]
        after = chain[index]
        if after <= before:
            raise GenerationChainError(
                "generation regression is forbidden",
                path=f"$.leases[?lease_id={lease_id}].generation[{index}]",
                details={
                    "lease_id": lease_id,
                    "field": "generation",
                    "generation": after,
                    "previous_generation": before,
                },
            )
        validate_generation_edge(lease_id, before, after)
    return chain
