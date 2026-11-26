from __future__ import annotations

import pytest

from lease_arbitrator.errors import GenerationChainError
from lease_arbitrator.generation import audit_generation_chain, validate_generation_edge


def test_generation_chain_accepts_exact_increments() -> None:
    assert audit_generation_chain("L1", [1, 2, 3, 4]) == [1, 2, 3, 4]


def test_generation_jump_is_egen() -> None:
    with pytest.raises(GenerationChainError) as exc:
        validate_generation_edge("L1", 1, 3)
    assert exc.value.code == "E_GEN"
    assert exc.value.details["lease_id"] == "L1"


def test_generation_regression_is_egen() -> None:
    with pytest.raises(GenerationChainError) as exc:
        audit_generation_chain("L1", [1, 2, 1])
    assert exc.value.code == "E_GEN"
    assert exc.value.details["field"] == "generation"
    assert exc.value.details["generation"] == 1
