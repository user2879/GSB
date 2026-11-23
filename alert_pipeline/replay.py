"""Post-recovery deterministic replay verification (E_REPLAY)."""
from __future__ import annotations

from .errors import ReplayError
from .util import canonical_digest


def compare_named(actual_records: list[dict], expected_records: list[dict],
                  id_field: str, label: str) -> None:
    """First-position mismatch between replayed and ledgered conclusions."""
    actual = {r[id_field]: r for r in actual_records}
    expected = {r[id_field]: r for r in expected_records}
    all_ids = sorted(set(actual) | set(expected))
    for aid in all_ids:
        if aid not in expected:
            raise ReplayError(
                f"replay produced unexpected {label}", 0, id_field, aid)
        if aid not in actual:
            raise ReplayError(
                f"replay missing expected {label}", 0, id_field, aid)
        if canonical_digest(actual[aid]) != canonical_digest(expected[aid]):
            field_name = _first_diverging_field(actual[aid], expected[aid])
            raise ReplayError(
                f"replay divergence for {label}", 0, field_name, aid)


def _first_diverging_field(left: dict, right: dict) -> str:
    for key in sorted(set(left) | set(right)):
        if left.get(key) != right.get(key):
            return key
    return "record"


def verify_replay(replayed_aggregates: list[dict],
                  ledger_aggregates: list[dict],
                  replayed_correlations: list[dict],
                  ledger_correlations: list[dict]) -> None:
    compare_named(replayed_aggregates, ledger_aggregates, "aggregate_id",
                  "aggregate")
    compare_named(replayed_correlations, ledger_correlations, "aggregate_id",
                  "correlation")
