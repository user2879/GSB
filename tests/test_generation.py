from fwaudit.errors import E_STALE_BASE
from fwaudit.engine import AuditEngine
from fwaudit.generator import build_image, build_patch
from fwaudit.patch import BudgetConfig


def _inputs(images, patches):
    return (
        [(name, blob) for name, blob in images],
        [(name, blob) for name, blob in patches],
    )


def test_stale_after_base_id_change(tmp_path, partitions, budget):
    v1 = build_image(partitions, version=1, base_image_id=100)
    v2 = build_image(partitions, version=1, base_image_id=200)
    old_patch = build_patch(
        [{"name": "kernel", "write_offset": 0, "write_length": 8}],
        base_image_id=100,
    )
    new_patch = build_patch(
        [{"name": "kernel", "write_offset": 0, "write_length": 8}],
        base_image_id=200,
    )
    engine = AuditEngine(str(tmp_path), budget=budget)
    out = engine.run(
        *_inputs(
            [("v1", v1), ("v2", v2)],
            [("old", old_patch), ("new", new_patch)],
        )
    )
    by_path = {r["path"]: r for r in out["results"]}
    assert E_STALE_BASE in by_path["old"]["codes"]
    assert by_path["new"]["codes"] == []
    assert engine.generation_watermark == 2


def test_stale_on_version_change(tmp_path, partitions, budget):
    v1 = build_image(partitions, version=1, base_image_id=100)
    patch = build_patch(
        [{"name": "kernel", "write_offset": 0, "write_length": 8}],
        base_image_id=100,
    )
    engine_a = AuditEngine(str(tmp_path / "a"), budget=budget)
    engine_a.run(*_inputs([("v1", v1)], [("p", patch)]))

    v2 = build_image(partitions, version=2, base_image_id=100)
    engine_b = AuditEngine(str(tmp_path / "b"), budget=budget)
    out = engine_b.run(
        *_inputs([("v1", v1), ("v2", v2)], [("p", patch)])
    )
    assert engine_b.images[0]["version"] != engine_b.images[1]["version"]
    assert E_STALE_BASE in out["results"][0]["codes"]
    assert out["results"][0]["stale"] is True


def test_interleaved_imports_give_same_stale_set(tmp_path, partitions, budget):
    v1 = build_image(partitions, version=1, base_image_id=100)
    v2 = build_image(partitions, version=1, base_image_id=200)
    old_patch = build_patch(
        [{"name": "kernel", "write_offset": 0, "write_length": 8}],
        base_image_id=100,
    )
    new_patch = build_patch(
        [{"name": "kernel", "write_offset": 0, "write_length": 8}],
        base_image_id=200,
    )

    def run(order):
        engine = AuditEngine(str(tmp_path / order), budget=budget)
        images = [("v1", v1), ("v2", v2)]
        patches = [("old", old_patch), ("new", new_patch)]
        out = engine.run(*_inputs(images, patches))
        return {
            r["path"]: sorted(r["codes"]) for r in out["results"]
        }

    first = run("a")
    second = run("b")
    assert first == second
    assert first["old"] == [E_STALE_BASE]
    assert first["new"] == []


def test_patches_accumulate_without_image_update(tmp_path, partitions, budget):
    v1 = build_image(partitions, version=1, base_image_id=100)
    p1 = build_patch(
        [{"name": "kernel", "write_offset": 0, "write_length": 500}],
        base_image_id=100,
    )
    p2 = build_patch(
        [{"name": "kernel", "write_offset": 500, "write_length": 40}],
        base_image_id=100,
    )
    engine = AuditEngine(str(tmp_path), budget=budget)
    out = engine.run(*_inputs([("v1", v1)], [("p1", p1), ("p2", p2)]))
    # p2 单独越界；两份补丁互不影响、均被独立审计
    statuses = {r["path"]: r["status"] for r in out["results"]}
    assert statuses["p1"] == "pass"
    assert statuses["p2"] == "fail"
