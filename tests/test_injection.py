from fablock.errors import Code
from fablock.injection import Injection
from fablock.pipeline import Pipeline


def test_missing_tensor_injection(cfg, dataset):
    inj = [Injection(id="INJ-001", type="missing_tensor",
                     target={"tensor": "V"})]
    r = Pipeline(cfg, dataset, injections=inj).run(1)[0]
    assert r["injection_hits"] == [
        {"id": "INJ-001", "type": "missing_tensor",
         "code": Code.E_TENSOR_MISSING}]
    assert any(i["code"] == Code.E_TENSOR_MISSING for i in r["issues"])
    assert r["fatal"]


def test_head_dim_mismatch_injection(cfg, dataset):
    inj = [Injection(id="INJ-002", type="head_dim_mismatch",
                     target={"tensor": "Q", "head_dim": 6})]
    r = Pipeline(cfg, dataset, injections=inj).run(1)[0]
    assert r["injection_hits"][0]["code"] == Code.E_HEAD_DIM_MISMATCH
    assert any(i["code"] == Code.E_HEAD_DIM_MISMATCH for i in r["issues"])


def test_nonfinite_injection_isolates_block(cfg, dataset):
    inj = [Injection(id="INJ-003", type="nonfinite",
                     target={"tensor": "K", "row": 10, "col": 3})]
    r = Pipeline(cfg, dataset, injections=inj).run(1)[0]
    assert r["injection_hits"][0]["code"] == Code.E_NONFINITE
    # 键块 1 在加载阶段失败并被隔离，其余块继续处理
    assert any(f["code"] == Code.E_NONFINITE and f["key_block"] == 1
               and f["stage"] == "load_validate"
               and f["injection_id"] == "INJ-003" for f in r["failures"])
    # 键块 1 被隔离，所有查询块在剩余键块上继续完成
    assert r["completed_query_blocks"] == [0, 1, 2]
    assert not r.get("fatal")


def test_frac_bits_out_of_range_injection(cfg, dataset):
    inj = [Injection(id="INJ-005", type="frac_bits_out_of_range",
                     target={"value": 40})]
    r = Pipeline(cfg, dataset, injections=inj).run(1)[0]
    assert r["injection_hits"][0]["code"] == Code.E_FRAC_BITS_OUT_OF_RANGE
    assert any(i["code"] == Code.E_FRAC_BITS_OUT_OF_RANGE
               for i in r["issues"])


def test_matmul_block_fault_isolation(cfg, dataset):
    inj = [Injection(id="INJ-006", type="block_fault",
                     target={"stage": "matmul", "query_block": 2,
                             "key_block": 1})]
    clean = Pipeline(cfg, dataset).run(1)[0]
    r = Pipeline(cfg, dataset, injections=inj).run(1)[0]
    # 失败块记录块号、阶段名与业务码，且不计入输出汇总
    assert any(f["stage"] == "block_score" and f["query_block"] == 2
               and f["key_block"] == 1
               and f["code"] == Code.E_INJECTED_FAULT
               and f["injection_id"] == "INJ-006" for f in r["failures"])
    assert 2 not in r["completed_query_blocks"]
    assert 2 in r["excluded_query_blocks"]
    # 同批次其余块结果与无注入时逐元素一致
    for qi in (0, 1):
        assert r["outputs_fp"][str(qi)] == clean["outputs_fp"][str(qi)]


def test_softmax_block_fault_isolation(cfg, dataset):
    inj = [Injection(id="INJ-007", type="block_fault",
                     target={"stage": "softmax", "query_block": 0,
                             "key_block": 2})]
    clean = Pipeline(cfg, dataset).run(1)[0]
    r = Pipeline(cfg, dataset, injections=inj).run(1)[0]
    assert any(f["stage"] == "softmax_reduce" and f["query_block"] == 0
               and f["injection_id"] == "INJ-007" for f in r["failures"])
    assert 0 not in r["completed_query_blocks"]
    for qi in (1, 2):
        assert r["outputs_fp"][str(qi)] == clean["outputs_fp"][str(qi)]
