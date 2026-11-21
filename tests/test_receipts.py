from __future__ import annotations

import random

from modeldash.receipts import (
    DEPENDENCY,
    MISMATCH,
    UNVERIFIED,
    VERIFIED,
    Receipt,
    ReceiptBook,
    build_node_untrust,
    receipt_key,
)


def make_receipt(
    node: str,
    task: str,
    version: int,
    shard: int,
    input_digest: str,
    output: str,
    item_keys: tuple[str, ...],
) -> Receipt:
    return Receipt(
        receipt_key=receipt_key(node, task, version, shard, input_digest),
        node_id=node,
        task_id=task,
        config_version=version,
        shard_index=shard,
        shard_count=3,
        item_keys=item_keys,
        input_digest=input_digest,
        output_summary=output,
        generated_version=version,
    )


def build_receipts() -> list[Receipt]:
    return [
        make_receipt("n1", "old0", 1, 0, "input-x", "old-x", ("alpha", "beta")),
        make_receipt("n1", "current0", 2, 0, "input-x", "new-x", ("alpha", "beta")),
        make_receipt("n1", "old1", 1, 1, "input-y", "old-y", ("alpha",)),
        make_receipt("n1", "old2", 1, 2, "input-z", "old-z", ("delta",)),
        make_receipt("n2", "old0", 1, 0, "input-x", "old-x", ("alpha",)),
        make_receipt("n3", "old0", 1, 0, "input-x", "stable-x", ("gamma",)),
        make_receipt("n3", "current0", 2, 0, "input-x", "stable-x", ("gamma",)),
    ]


def test_audit_classifies_and_quantifies_deterministically():
    outputs = []
    rankings = []
    for seed in range(5):
        values = build_receipts()
        random.Random(seed).shuffle(values)
        book = ReceiptBook()
        for receipt in values:
            book.add(receipt)
        audited = book.audit(current_version=2)
        by_key = {item.receipt_key: item for item in audited}
        n1_old0 = by_key[receipt_key("n1", "old0", 1, 0, "input-x")]
        n1_old1 = by_key[receipt_key("n1", "old1", 1, 1, "input-y")]
        n1_old2 = by_key[receipt_key("n1", "old2", 1, 2, "input-z")]
        n2_old0 = by_key[receipt_key("n2", "old0", 1, 0, "input-x")]
        n3_old0 = by_key[receipt_key("n3", "old0", 1, 0, "input-x")]
        assert n1_old0.status == MISMATCH
        assert n1_old1.status == DEPENDENCY
        assert n1_old2.status == UNVERIFIED
        assert n2_old0.status == UNVERIFIED
        assert n3_old0.status == VERIFIED
        assert n1_old0.untrust_score > n1_old1.untrust_score >= n1_old2.untrust_score > 0
        ranking = build_node_untrust(audited, ["n1", "n2", "n3"], 2)
        outputs.append([(item.receipt_key, item.status, item.decay_count, item.untrust_score) for item in audited])
        rankings.append(ranking)
    assert all(value == outputs[0] for value in outputs)
    assert all(value == rankings[0] for value in rankings)
    assert rankings[0][0]["node_id"] == "n1"
    assert rankings[0][-1]["node_id"] == "n3"
    assert rankings[0][-1]["score"] == 0


def test_current_receipts_with_disagreeing_outputs_are_inconsistent():
    book = ReceiptBook()
    book.add(make_receipt("n1", "a", 2, 0, "same-input", "out-a", ("alpha",)))
    book.add(make_receipt("n1", "b", 2, 0, "same-input", "out-b", ("alpha",)))
    audited = book.audit(2)
    assert {item.status for item in audited} == {"UNTRUSTED_INCONSISTENT"}
    assert all(item.untrust_score == 220 for item in audited)
