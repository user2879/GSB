from __future__ import annotations

from fcdiag.fixedpoint import FixedConfig
from fcdiag.loader import load_samples
from fcdiag.models import Config
from fcdiag.runner import Runner
from fcdiag.state import apply_command


def test_single_sample_replacement_only_invalidates_dependent_chain(context):
    state = context["state"]
    receipts = context["receipts"][3]
    command = {
        "op": "replace_sample",
        "sample": {
            "sample_id": "s002",
            "features": [1.25, -0.5, 0.75, 2.0],
            "label": 1,
        },
    }
    changed = apply_command(state, command)
    result = Runner(changed, receipts).run()
    decisions = {entry.artifact_id: entry.decision for entry in result.ledger}
    assert decisions["raw:s002"] == "RECOMPUTE"
    assert decisions["standardize:sample:s002"] == "RECOMPUTE"
    assert decisions["bucketize:sample:s002"] == "RECOMPUTE"
    assert decisions["interact:sample:s002"] == "RECOMPUTE"
    # Global standardization stats and the final aggregate necessarily see all rows.
    assert decisions["standardize:global"] == "RECOMPUTE"
    assert decisions["train_view:global"] == "RECOMPUTE"
    # Untouched raw slices remain directly reusable.
    assert decisions["raw:s000"] == "REUSE"
    assert decisions["raw:s001"] == "REUSE"


def test_parameter_edit_invalidates_only_its_stage_and_downstream(context):
    state = context["state"]
    receipts = context["receipts"][3]
    changed = apply_command(
        state,
        {"op": "set_interaction_degree", "stage": "interact", "value": 3},
    )
    result = Runner(changed, receipts).run()
    decisions = {entry.artifact_id: entry.decision for entry in result.ledger}
    assert all(decision == "REUSE" for artifact, decision in decisions.items() if artifact.startswith("standardize:"))
    assert all(decision == "REUSE" for artifact, decision in decisions.items() if artifact.startswith("bucketize:"))
    assert all(decision == "RECOMPUTE" for artifact, decision in decisions.items() if artifact.startswith("interact:"))
    assert decisions["train_view:global"] == "RECOMPUTE"


def test_degraded_receipt_uses_param_version_and_records_basis(context):
    state = context["state"]
    result = Runner(state, context["receipts"][2]).run()
    degraded = [entry for entry in result.ledger if entry.degraded]
    assert degraded
    assert all(entry.degradation_basis and "digest absent" in entry.degradation_basis for entry in degraded)
    strict_config = Config(
        state.config.fixed,
        state.config.budget,
        type(state.config.policy)(degraded_reuse="strict"),
        state.config.seed,
    )
    strict_state = state.copy()
    object.__setattr__(strict_state, "config", strict_config)
    strict_result = Runner(strict_state, context["receipts"][2]).run()
    assert all(entry.decision == "RECOMPUTE" for entry in strict_result.ledger if entry.degraded)


def test_v1_receipt_never_silently_reuses_without_evidence(context):
    result = Runner(context["state"], context["receipts"][1]).run()
    assert all(entry.decision != "REUSE" for entry in result.ledger if entry.receipt_version == 1)
