"""样本生成器确定性与同步标记完备性测试。"""

from __future__ import annotations

import json
import os

from audit_engine.generator import SYNC_EPOCH, generate_all


def test_generator_output_byte_stable(tmp_path):
    first = tmp_path / "a"
    second = tmp_path / "b"
    paths_a = generate_all(str(first))
    paths_b = generate_all(str(second))
    assert len(paths_a) == len(paths_b)
    for path_a, path_b in zip(sorted(paths_a), sorted(paths_b)):
        with open(path_a, "rb") as handle:
            bytes_a = handle.read()
        with open(os.path.join(second, os.path.basename(path_b)), "rb") as handle:
            bytes_b = handle.read()
        assert bytes_a == bytes_b


def test_sync_markers_are_explicit_and_fixed(tmp_path):
    paths = generate_all(str(tmp_path))
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        for record in manifest["records"]:
            if "truth" not in record:
                continue  # 解析错误样本中的非法记录故意不带同步标记
            truth = record["truth"]
            assert truth["sync_marker"].startswith(SYNC_EPOCH)
            assert truth["raw_path_bytes"] == record["target_path"].encode("utf-8").hex()
            lo, hi = truth["expected_epsilon"]
            assert 0 <= lo <= hi


def test_no_wallclock_dependency_in_truth(tmp_path):
    paths = generate_all(str(tmp_path))
    first_mtime = {}
    for path in paths:
        with open(path, "rb") as handle:
            first_mtime[os.path.basename(path)] = handle.read()
    # 再生成一次，内容不随运行次序/时间变化
    again = tmp_path / "again"
    for path in generate_all(str(again)):
        with open(path, "rb") as handle:
            assert handle.read() == first_mtime[os.path.basename(path)]
