import numpy as np
import pytest

from fablock.config import OperatorConfig
from fablock.dataset import SampleDataset
from fablock.errors import Code, OperatorError
from fablock.tensor_pack import read_pack, validate_pack, write_pack


def test_roundtrip(tmp_path, dataset):
    path = tmp_path / "pack.fatp"
    dataset.write_pack(path)
    tensors, meta = read_pack(path)
    assert set(tensors) == {"Q", "K", "V"}
    np.testing.assert_array_equal(tensors["Q"], dataset.Q)
    assert meta["pad_rule"] == "zero"
    assert meta["orig_shape"]["Q"] == [20, 8]


def test_shape_data_mismatch(tmp_path, dataset):
    path = tmp_path / "pack.fatp"
    dataset.write_pack(path)
    blob = bytearray(path.read_bytes())
    blob[-5:] = b"\x00" * 5  # 截断语义：篡改尾部数据长度
    bad = tmp_path / "bad.fatp"
    bad.write_bytes(bytes(blob[:-3]))
    with pytest.raises(OperatorError) as exc:
        read_pack(bad)
    assert exc.value.code == Code.E_SHAPE_DATA_MISMATCH


def test_missing_tensor(dataset, cfg):
    tensors = {"Q": dataset.Q, "K": dataset.K}
    issues = validate_pack(tensors, dataset.meta, cfg)
    assert any(i.code == Code.E_TENSOR_MISSING for i in issues)


def test_nonfinite_maps_to_block(dataset, cfg):
    q = dataset.Q.copy()
    q[9, 2] = np.nan  # 查询块 1
    issues = validate_pack({"Q": q, "K": dataset.K, "V": dataset.V},
                           dataset.meta, cfg)
    nonfinite = [i for i in issues if i.code == Code.E_NONFINITE]
    assert nonfinite and nonfinite[0].context["query_blocks"] == [1]


def test_head_dim_mismatch(dataset, cfg):
    bad_q = dataset.Q[:, :6]
    issues = validate_pack({"Q": bad_q, "K": dataset.K, "V": dataset.V},
                           dataset.meta, cfg)
    assert any(i.code == Code.E_HEAD_DIM_MISMATCH for i in issues)


def test_head_dim_must_be_positive_even():
    assert any(i.code == Code.E_HEAD_DIM_MISMATCH
               for i in OperatorConfig(head_dim=7).validate())
    assert any(i.code == Code.E_HEAD_DIM_MISMATCH
               for i in OperatorConfig(head_dim=0).validate())


def test_pad_rule_violation(dataset, cfg):
    tensors = {"Q": dataset.Q[:-1], "K": dataset.K, "V": dataset.V}
    issues = validate_pack(tensors, dataset.meta, cfg)
    assert any(i.code == Code.E_PAD_RULE_VIOLATION for i in issues)
