"""离线验收脚本：确定性生成样本并断言业务码、覆盖率与退出码语义。"""

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fwaudit.cli import main, EXIT_PARTIAL_FAIL
from fwaudit.engine import AuditEngine
from fwaudit.errors import AuditError, E_LOCK
from fwaudit.generator import build_image, build_patch
from fwaudit.patch import BudgetConfig
from fwaudit.storage import file_lock
from fwaudit.verify import verify_image_truth


def main_demo():
    work = tempfile.mkdtemp(prefix="fwaudit-acceptance-")
    samples = os.path.join(work, "samples")
    reports = os.path.join(work, "reports")
    state = os.path.join(work, "state")
    os.makedirs(samples)

    partitions = [
        {"name": "bootloader", "part_type": 0, "slot": 0,
         "payload_length": 256},
        {"name": "kernel", "part_type": 1, "slot": 0,
         "payload_length": 512},
        {"name": "appfs", "part_type": 2, "slot": 0,
         "payload_length": 768},
    ]
    image = build_image(partitions, base_image_id=100)
    next_image = build_image(partitions, version=2, base_image_id=200)
    good = build_patch(
        [{"name": "kernel", "write_offset": 0, "write_length": 32}],
        base_image_id=100,
    )
    bad = build_patch(
        [{"name": "kernel", "write_offset": 500, "write_length": 64}],
        base_image_id=100,
    )
    stale = build_patch(
        [{"name": "kernel", "write_offset": 0, "write_length": 8}],
        base_image_id=100,
    )
    paths = {}
    for name, blob in [
        ("v1.fwim", image), ("v2.fwim", next_image),
        ("good.fwdf", good), ("bad.fwdf", bad), ("stale.fwdf", stale),
    ]:
        path = os.path.join(samples, name)
        with open(path, "wb") as handle:
            handle.write(blob)
        paths[name] = path

    ok, truth = verify_image_truth(image)
    assert ok and not truth["mismatches"]

    code = main([
        "audit", "--images", paths["v1.fwim"],
        "--patches", paths["good.fwdf"], paths["bad.fwdf"],
        paths["stale.fwdf"],
        "--reports-dir", reports, "--workdir", state,
    ])
    assert code == EXIT_PARTIAL_FAIL
    report = json.load(open(os.path.join(reports, "audit_report.json")))
    by_name = {
        os.path.basename(p["path"]): p for p in report["patches"]
    }
    assert by_name["good.fwdf"]["status"] == "pass"
    assert "E_PATCH_BOUNDS" in by_name["bad.fwdf"]["codes"]
    assert "E_WRITE_CHAIN" in by_name["bad.fwdf"]["codes"]
    assert by_name["stale.fwdf"]["status"] == "pass"
    assert by_name["good.fwdf"]["coverage_ratio"] == 1.0

    state2 = os.path.join(work, "state2")
    engine = AuditEngine(state2, budget=BudgetConfig(concurrency=2))
    out = engine.run(
        [("v1", image), ("v2", next_image)],
        [("old", stale), ("good", good), ("newgen", build_patch(
            [{"name": "kernel", "write_offset": 0, "write_length": 8}],
            base_image_id=200,
        ))],
        lock=False,
    )
    stale_paths = {r["path"]: r["codes"] for r in out["results"]}
    assert "E_STALE_BASE" in stale_paths["old"]
    assert "E_STALE_BASE" in stale_paths["good"]
    assert stale_paths["newgen"] == []

    try:
        with file_lock(state):
            with file_lock(state):
                raise AssertionError("expected E_LOCK")
    except AuditError as exc:
        assert exc.code == E_LOCK

    shutil.rmtree(work, ignore_errors=True)
    print("ACCEPTANCE OK: 业务码/覆盖率/退出码/失效传播/文件锁全部符合预期")


if __name__ == "__main__":
    main_demo()
