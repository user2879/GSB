from __future__ import annotations

import pytest

from topic_ledger.errors import BusinessError
from topic_ledger.samplegen import build_sample_bundle
from topic_ledger.validation import validate_command_stream


def test_command_stream_rejects_unknown_op_before_scheduling() -> None:
    bundle = build_sample_bundle()
    with pytest.raises(BusinessError) as exc_info:
        validate_command_stream(
            [{"command_id": "bad", "op": "not-a-real-op"}],
            shard_ids=bundle["topology"]["shards"],
            term_ids={item["term_id"] for item in bundle["vocab"]},
            initial_documents=bundle["documents"],
            node_ids={node["node_id"] for node in bundle["topology"]["nodes"]},
        )
    assert exc_info.value.code == "E_INVALID_INPUT"
    assert exc_info.value.field == "commands[0].op"


def test_generated_command_stream_cross_validates() -> None:
    bundle = build_sample_bundle()
    validate_command_stream(
        bundle["commands"],
        shard_ids=bundle["topology"]["shards"],
        term_ids={item["term_id"] for item in bundle["vocab"]},
        initial_documents=bundle["documents"],
        node_ids={node["node_id"] for node in bundle["topology"]["nodes"]},
        vocab_size=len(bundle["vocab"]),
    )
