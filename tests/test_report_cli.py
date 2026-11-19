import json
import os
import sys

from fwaudit.cli import main, EXIT_ALL_PASS, EXIT_PARTIAL_FAIL, EXIT_ARG_ERROR
from fwaudit.generator import build_image, build_patch
from fwaudit.image import parse_image
from fwaudit.engine import AuditEngine
from fwaudit.patch import BudgetConfig
from fwaudit.report import build_report, render_graph_text
from fwaudit.verify import verify_image_truth


def test_report_structure_and_graph(image_blob, budget):
    patch_blob = build_patch(
        [{"name": "kernel", "write_offset": 0, "write_length": 8}],
        base_image_id=100,
    )
    engine = AuditEngine("/tmp/fwaudit-report-1", budget=budget)
    out = engine.run(
        [("img", image_blob)], [("p", patch_blob)], lock=False
    )
    report = build_report(parse_image(image_blob), out, budget)
    # JSON 可解析且键确定
    encoded = json.dumps(report, sort_keys=True)
    assert json.loads(encoded)["summary"]["overall_status"] == "pass"
    assert report["image"]["partitions"][0]["mapping_status"] == "mapped"
    graph = report["graph"]
    assert {n["id"] for n in graph["nodes"]} == {
        "bootloader", "kernel", "appfs"
    }
    assert all("slot" in n for n in graph["nodes"])
    text = render_graph_text(graph)
    assert "bootloader" in text and "slot=" in text


def test_truth_marker_roundtrip(image_blob, partitions):
    ok, detail = verify_image_truth(image_blob)
    assert ok is True
    assert len(detail["checked"]) == 3
    assert detail["mismatches"] == []
    # 篡改载荷后真值摘要应失配
    tampered = bytearray(image_blob)
    first = parse_image(image_blob).partitions[0]
    from fwaudit.generator import MARKER_SIZE
    tampered[first.payload_offset + MARKER_SIZE + 10] ^= 0xFF
    ok2, detail2 = verify_image_truth(bytes(tampered))
    assert ok2 is False
    assert detail2["mismatches"][0]["name"] == "bootloader"


def test_cli_exit_codes(tmp_path, partitions):
    samples = tmp_path / "samples"
    reports = tmp_path / "reports"
    state = tmp_path / "state"
    samples.mkdir()
    image = build_image(partitions, base_image_id=100)
    good = build_patch(
        [{"name": "kernel", "write_offset": 0, "write_length": 8}],
        base_image_id=100,
    )
    bad = build_patch(
        [{"name": "kernel", "write_offset": 9000, "write_length": 8}],
        base_image_id=100,
    )
    img_path = str(samples / "a.fwim")
    good_path = str(samples / "good.fwdf")
    bad_path = str(samples / "bad.fwdf")
    for path, blob in [
        (img_path, image), (good_path, good), (bad_path, bad)
    ]:
        with open(path, "wb") as handle:
            handle.write(blob)

    code = main([
        "audit", "--images", img_path, "--patches", good_path,
        "--reports-dir", str(reports / "r1"),
        "--workdir", str(state / "s1"),
    ])
    assert code == EXIT_ALL_PASS

    code = main([
        "audit", "--images", img_path,
        "--patches", good_path, bad_path,
        "--reports-dir", str(reports / "r2"),
        "--workdir", str(state / "s2"),
    ])
    assert code == EXIT_PARTIAL_FAIL
    report = json.loads(
        (reports / "r2" / "audit_report.json").read_text()
    )
    assert report["summary"]["patch_failed"] == 1
    assert (reports / "r2" / "partition_map.json").exists()


def test_cli_arg_error_on_missing_file(tmp_path):
    code = main([
        "audit", "--images", str(tmp_path / "nope.fwim"),
        "--reports-dir", str(tmp_path / "r"),
        "--workdir", str(tmp_path / "s"),
    ])
    assert code == EXIT_ARG_ERROR


def test_malformed_image_does_not_kill_batch(tmp_path, partitions, budget):
    good_image = build_image(partitions, base_image_id=100)
    bad_image = build_image(partitions, base_image_id=100,
                            corrupt={"magic": True})
    patch = build_patch(
        [{"name": "kernel", "write_offset": 0, "write_length": 8}],
        base_image_id=100,
    )
    engine = AuditEngine(str(tmp_path / "state"), budget=budget)
    out = engine.run(
        [("good", good_image), ("bad", bad_image)],
        [("p", patch)],
        lock=False,
    )
    assert out["parse_errors"][0]["error"]["code"] == "E_HEADER"
    # 补丁仍基于最后一个可解析镜像完成审计
    assert out["results"][0]["status"] == "pass"
