from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from haptic_diag.checker import (
    CAUSE_PROTOCOL,
    CAUSE_QUANTIZATION,
    CAUSE_TRANSPORT,
    reconcile_receipts,
)
from haptic_diag.engine import EngineState, initial_state
from haptic_diag.generator import load_config, generate_sample
from haptic_diag.transport import (
    DispatchConfig,
    LoopbackClient,
    LoopbackServer,
    ReceiptChannel,
    dispatch_concurrently,
)

ROOT = Path(__file__).resolve().parents[1]


def _state() -> EngineState:
    config = load_config(ROOT / "configs" / "samples.json")
    graph = generate_sample(config["samples"][2])
    return initial_state(graph, 2)


def _dispatch(state: EngineState, server: LoopbackServer, frames=None, **config):
    if frames is None:
        frames = state.frames[:100]
    defaults = dict(
        max_workers=4,
        batch_size=20,
        batch_memory_limit_bytes=262144,
        batch_time_budget_ms=250.0,
    )
    defaults.update(config)
    return dispatch_concurrently(
        frames,
        LoopbackClient(server.host, server.port),
        DispatchConfig(**defaults),
    )


def test_concurrent_dispatch_merges_by_frame_key_and_checks_cleanly() -> None:
    state = _state()
    with LoopbackServer() as server:
        result = _dispatch(state, server)
    assert [receipt.seq for receipt in result.receipts] == list(range(100))
    assert result.missing_sequences == ()
    assert result.duplicate_receipt_sequences == ()
    assert result.sequence_gaps == ()
    check = reconcile_receipts(
        state.frames[:100],
        result.receipts,
        expected_batch_digests=state.digests[:5],
        observed_batch_digests=result.observed_batch_digests,
        observed_batch_ranges=result.observed_batch_ranges,
    )
    assert check.status == "confirmed"
    assert check.first_mismatch is None
    assert check.digest_consistent is True


def test_missing_receipt_is_pending_not_success() -> None:
    state = _state()
    with LoopbackServer(drop_sequences=[7]) as server:
        result = _dispatch(state, server)
    check = reconcile_receipts(state.frames[:100], result.receipts)
    assert result.missing_sequences == (7,)
    assert check.pending_sequences == (7,)
    assert check.status == "pending_confirmation"
    assert check.first_mismatch is not None
    assert check.first_mismatch.seq == 7
    assert check.first_mismatch.cause == CAUSE_TRANSPORT
    assert 7 not in check.confirmed_sequences


def test_protocol_field_difference_is_identified_before_generic_failure() -> None:
    state = _state()
    frames = state.frames[720:740]
    with LoopbackServer(downgrade_protocol=1) as server:
        result = _dispatch(state, server, frames=frames)
    check = reconcile_receipts(frames, result.receipts)
    assert check.status == "mismatch"
    assert check.first_mismatch is not None
    assert check.first_mismatch.cause == CAUSE_PROTOCOL
    assert check.first_mismatch.seq == frames[0].seq
    assert any(issue.field == "protocol_version" for issue in check.issues)
    assert any(issue.field.endswith("frequency_hz") for issue in check.issues)


def test_quantization_difference_is_distinguished() -> None:
    state = _state()
    frames = state.frames[720:760]
    with LoopbackServer() as server:
        result = _dispatch(state, server, frames=frames)
    target = next(receipt for receipt in result.receipts if receipt.channels)
    modified_channels = tuple(
        replace(
            channel,
            amplitude_code=min(65535, channel.amplitude_code + 1),
            amplitude=channel.amplitude,
        )
        for channel in target.channels
    )
    modified = replace(target, channels=modified_channels)
    receipts = tuple(modified if receipt.seq == target.seq else receipt for receipt in result.receipts)
    check = reconcile_receipts(frames, receipts)
    assert check.status == "mismatch"
    assert check.first_mismatch is not None
    assert check.first_mismatch.seq == target.seq
    assert check.first_mismatch.cause == CAUSE_QUANTIZATION
    assert check.first_mismatch.within_tolerance


def test_digest_mismatch_alone_does_not_fail_frame_reconciliation() -> None:
    state = _state()
    with LoopbackServer() as server:
        result = _dispatch(state, server, frames=state.frames[:100])
    observed = list(result.observed_batch_digests)
    observed[0] = "0" * 64
    check = reconcile_receipts(
        state.frames[:100],
        result.receipts,
        expected_batch_digests=state.digests[:5],
        observed_batch_digests=observed,
        observed_batch_ranges=result.observed_batch_ranges,
    )
    assert check.status == "confirmed"
    assert check.first_mismatch is None
    assert check.digest_consistent is False
    assert check.digest_mismatch_only is True
    assert check.confirmed_sequences == tuple(range(100))


def test_budget_degradation_has_no_duplicate_or_sequence_gap() -> None:
    state = _state()
    with LoopbackServer() as server:
        result = _dispatch(
            state,
            server,
            frames=state.frames[:100],
            max_workers=4,
            batch_size=64,
            batch_memory_limit_bytes=2000,
            batch_time_budget_ms=2.0,
        )
    reasons = {item.reason for item in result.degradations}
    assert "memory_budget" in reasons
    assert "time_budget" in reasons or "time_budget_floor" in reasons
    assert result.effective_workers <= 4
    flattened = [frame.seq for batch in result.batches for frame in batch.frames]
    assert len(flattened) == len(set(flattened)) == 100
    assert sorted(flattened) == list(range(100))
    assert result.duplicate_receipt_sequences == ()
    assert result.sequence_gaps == ()
    assert result.missing_sequences == ()
    assert len(result.receipts) == 100


def test_physically_insufficient_memory_budget_records_floor() -> None:
    state = _state()
    with LoopbackServer() as server:
        result = _dispatch(
            state,
            server,
            frames=state.frames[:20],
            max_workers=2,
            batch_size=10,
            batch_memory_limit_bytes=100,
            batch_time_budget_ms=250.0,
        )
    assert any(item.reason == "memory_budget_floor" for item in result.degradations)
    assert result.missing_sequences == ()
    assert result.duplicate_receipt_sequences == ()
    assert sorted(receipt.seq for receipt in result.receipts) == list(range(20))
