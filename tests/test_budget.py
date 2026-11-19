from fwaudit.errors import E_OPS, E_TIMEOUT
from fwaudit.engine import AuditEngine
from fwaudit.generator import build_image, build_patch
from fwaudit.patch import BudgetConfig


def test_degraded_chunk_coverage(image_blob, partitions):
    budget = BudgetConfig(
        image_byte_threshold=10, op_count_threshold=2,
        chunk_size=4, max_ops_per_patch=4096,
        timeout_seconds=5.0, concurrency=1,
    )
    ops = [
        {"name": "kernel", "write_offset": i, "write_length": 1}
        for i in range(10)
    ]
    patch_blob = build_patch(ops, base_image_id=100)
    engine = AuditEngine("/tmp/fwaudit-budget-1", budget=budget)
    out = engine.run(
        [("img", image_blob)], [("p", patch_blob)], lock=False
    )
    result = out["results"][0]
    assert result["degraded"] is True
    # 仅完整分块(每块 4 个)被校验：10 -> 覆盖 8 个
    assert result["coverage_ratio"] == 0.8
    assert result["uncovered_ops"] == [8, 9]
    # 未覆盖面不得报告为通过
    statuses = {op["seq"]: op["status"] for op in result["operations"]}
    assert statuses[8] == "skipped_degraded"
    assert statuses[9] == "skipped_degraded"


def test_ops_cap_aborts_but_batch_continues(image_blob):
    budget = BudgetConfig(
        image_byte_threshold=4096, op_count_threshold=16, chunk_size=8,
        max_ops_per_patch=3, timeout_seconds=5.0, concurrency=1,
    )
    big = build_patch(
        [
            {"name": "kernel", "write_offset": i, "write_length": 1}
            for i in range(5)
        ],
        base_image_id=100,
    )
    small = build_patch(
        [{"name": "kernel", "write_offset": 0, "write_length": 4}],
        base_image_id=100,
    )
    engine = AuditEngine("/tmp/fwaudit-budget-2", budget=budget)
    out = engine.run(
        [("img", image_blob)],
        [("big", big), ("small", small)],
        lock=False,
    )
    by_path = {r["path"]: r for r in out["results"]}
    assert E_OPS in by_path["big"]["codes"]
    assert by_path["big"]["coverage_ratio"] == 0.0
    assert by_path["small"]["status"] == "pass"


def test_real_timeout(image_blob):
    budget = BudgetConfig(
        image_byte_threshold=4096, op_count_threshold=16, chunk_size=8,
        max_ops_per_patch=4096, timeout_seconds=0.0, concurrency=1,
    )
    patch_blob = build_patch(
        [
            {"name": "kernel", "write_offset": 0, "write_length": 1}
        ],
        base_image_id=100,
    )
    engine = AuditEngine("/tmp/fwaudit-budget-3", budget=budget)
    out = engine.run(
        [("img", image_blob)], [("p", patch_blob)], lock=False
    )
    result = out["results"][0]
    assert E_TIMEOUT in result["codes"]
    assert result["status"] == "fail"
    assert result["coverage_ratio"] == 0.0
