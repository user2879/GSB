from __future__ import annotations

import pytest

from fcdiag.experiment import (
    assert_run_equal,
    full_rebuild,
    incremental_runs,
    result_signature,
)
from fcdiag.state import apply_commands


def test_incremental_chain_matches_full_rebuild(context):
    state = context["state"]
    commands = context["commands"]
    receipts = context["receipts"][3]
    incremental = incremental_runs(state, commands, receipts, max_workers=1)
    final_state = apply_commands(state, commands)[-1]
    rebuilt = full_rebuild(final_state, receipts, max_workers=1)
    assert_run_equal(incremental[-1], rebuilt)
    signature = result_signature(rebuilt)
    assert signature["artifacts"]
    assert signature["ledger"]
    assert signature["cache_keys"]


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_worker_count_does_not_change_results(context, workers):
    state = context["state"]
    commands = context["commands"]
    receipts = context["receipts"][3]
    incremental = incremental_runs(state, commands, receipts, max_workers=workers)
    final_state = apply_commands(state, commands)[-1]
    rebuilt = full_rebuild(final_state, receipts, max_workers=workers)
    assert_run_equal(incremental[-1], rebuilt)


def test_receipt_version_switching_is_covered_and_deterministic(context):
    ops = [command["op"] for command in context["commands"]]
    assert ops.count("set_receipt_version") == 3
    state = context["state"]
    commands = context["commands"]
    final_state = apply_commands(state, commands)[-1]
    assert final_state.receipt_version == 3
    left = full_rebuild(final_state, context["receipts"][3], 1)
    right = full_rebuild(final_state, context["receipts"][3], 4)
    assert result_signature(left) == result_signature(right)
