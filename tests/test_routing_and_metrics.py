from __future__ import annotations

import dataclasses
from itertools import permutations

from graygate.metrics import (
    aggregate_full,
    aggregate_hits,
    aggregate_streaming,
    summarize_latency_full,
    summarize_latency_streaming,
)
from graygate.routing import decide_route
from graygate.util import canonical_json


def test_stable_hash_is_order_independent(sample_spec):
    route = sample_spec.routes_by_id["alpha-route"]
    clients = [f"client-{index}" for index in range(20)]
    expected = {
        client: decide_route(sample_spec, route, 150, client)
        for client in clients
    }
    for ordering in list(permutations(clients[:5])) + [clients[5:]]:
        for client in ordering:
            actual = decide_route(sample_spec, route, 150, client)
            assert actual == expected[client]
            assert actual.stage_id == "alpha-1"
            assert actual.side in {"stable", "canary"}


def test_full_and_streaming_snapshot_aggregation_are_byte_identical(
    sample_spec, tmp_path
):
    from graygate.ledger import LedgerStore

    ledger = LedgerStore(tmp_path / "ledger.db")
    try:
        for index in range(60):
            ledger.append_hit(
                request_seq=index + 1,
                rule_id="alpha-route" if index % 2 == 0 else "beta-route",
                stage_id="alpha-0" if index % 2 == 0 else "beta-0",
                stage_weight_bp=0,
                hash_bucket=index % 10000,
                side="stable",
                backend_version=(
                    "alpha-stable-v1" if index % 2 == 0 else "beta-stable-v1"
                ),
                batch_seq=index // 4 + 1,
                output_digest=f"sha256:{index:064x}",
                elapsed_us=index + 1,
                client_key=f"c-{index}",
                feature="alpha:x" if index % 2 == 0 else "beta:y",
                sample=[0.5, 1.0],
            )
        hits = ledger.all_hits()
        full = aggregate_full(hits, sample_spec)
        streaming = aggregate_streaming(hits, sample_spec)
        assert canonical_json(full) == canonical_json(streaming)
        streaming_spec = dataclasses.replace(sample_spec, stream_threshold=50)
        via_auto, path = aggregate_hits(hits, streaming_spec)
        assert path == "streaming"
        assert canonical_json(via_auto) == canonical_json(full)
        assert sum(row["stable_hits"] for row in full) == 60
        assert all(
            row["ratio_error_upper_bound_bp"] in (0, 1) for row in full
        )
    finally:
        ledger.close()


def test_fixed_point_latency_paths_match_and_report_error_bound():
    values = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    full = summarize_latency_full(values)
    streaming = summarize_latency_streaming(values)
    assert canonical_json(full.canonical()) == canonical_json(streaming.canonical())
    assert full.quantization_error_upper_bound_scaled_ms == 1
    assert full.p95_ms == 144
    assert full.max_ms == 144
    assert full.mean_scaled_ms == (sum(values) * 1000) // len(values)
