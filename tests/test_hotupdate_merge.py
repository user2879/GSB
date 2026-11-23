import pytest

from gsb.errors import GSBError
from gsb.hotupdate import ParamRegistry, compute_param_digest
from gsb.merge import MergeCandidate, merge_receipts
from gsb.receipts import NormalizedReceipt
from gsb.spec import BackendDecl

DECL = BackendDecl(
    backend_id="b1",
    host="127.0.0.1",
    port=18001,
    model_dim=4,
    hidden_dim=8,
    seed=42,
    quant_profile="int8",
    format_version="v3",
)


def _receipt(param_version=1, source_format="v3", batch_id="batch-1"):
    return NormalizedReceipt(
        batch_id=batch_id,
        route="gray",
        output_digest="a" * 64,
        param_digest="p" * 64,
        quant_profile="int8",
        elapsed_ms=1,
        sample_ids=("q1",),
        param_version=param_version,
        source_format=source_format,
    )


def test_hot_update_increments_and_freezes():
    registry = ParamRegistry({"b1": DECL})
    binding_v1 = registry.bind("b1")
    assert binding_v1.param_version == 1
    binding_v2 = registry.hot_update("b1", 43)
    assert binding_v2.param_version == 2
    # 旧版本摘要冻结可取，新版本摘要不同
    assert registry.digest_for("b1", 1) == binding_v1.param_digest
    assert binding_v2.param_digest != binding_v1.param_digest
    assert binding_v2.param_digest == compute_param_digest(43, 4, 8, "int8")
    # 新派发绑定新版本
    assert registry.bind("b1").param_version == 2


def test_merge_prefers_higher_param_version():
    older = MergeCandidate(receipt=_receipt(param_version=1, source_format="v3"), merge_seq=0)
    newer = MergeCandidate(receipt=_receipt(param_version=2, source_format="v1"), merge_seq=1)
    conclusion = merge_receipts([older, newer])
    assert conclusion.adopted_seq == 1
    assert conclusion.param_version == 2
    assert conclusion.rejected[0].merge_seq == 0
    assert "param_version" in conclusion.rejected[0].reason


def test_merge_prefers_higher_format_at_same_version():
    v1 = MergeCandidate(receipt=_receipt(source_format="v1"), merge_seq=0)
    v3 = MergeCandidate(receipt=_receipt(source_format="v3"), merge_seq=1)
    conclusion = merge_receipts([v1, v3])
    assert conclusion.adopted_seq == 1
    assert "format_version" in conclusion.rejected[0].reason


def test_merge_stable_tiebreak_by_seq_and_source_format_set():
    first = MergeCandidate(receipt=_receipt(source_format="v2"), merge_seq=0)
    second = MergeCandidate(receipt=_receipt(source_format="v2"), merge_seq=1)
    conclusion = merge_receipts([second, first])  # 输入顺序不影响结论
    assert conclusion.adopted_seq == 0
    assert "merge_seq" in conclusion.rejected[0].reason
    assert conclusion.source_formats == ("v2",)


def test_merge_conclusion_fields():
    candidates = [
        MergeCandidate(receipt=_receipt(source_format="v1"), merge_seq=0),
        MergeCandidate(receipt=_receipt(source_format="v3"), merge_seq=1),
    ]
    conclusion = merge_receipts(candidates)
    assert conclusion.route == "gray"
    assert conclusion.sample_ids == ("q1",)
    assert conclusion.param_digest == "p" * 64
    assert conclusion.source_formats == ("v1", "v3")
    assert conclusion.adopted_format == "v3"


def test_mixed_error_shape():
    err = GSBError("E_MIXED", "混装", batch_id="b1", versions=[1, 2])
    assert err.code == "E_MIXED"
    assert err.context["batch_id"] == "b1"
    assert err.context["versions"] == [1, 2]
