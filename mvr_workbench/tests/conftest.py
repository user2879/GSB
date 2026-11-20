"""测试共享夹具：固定种子样本、工作副本与重建辅助。"""

from __future__ import annotations

import json
import os
import shutil
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mvrwb.pipeline import ReconstructionPipeline  # noqa: E402
from mvrwb.sampler import generate_sample  # noqa: E402
from mvrwb.util import read_json, write_json_atomic  # noqa: E402


@pytest.fixture(scope="session")
def sample_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("sample")
    generate_sample(str(path), view_count=6)
    return str(path)


@pytest.fixture()
def work_dirs(tmp_path):
    return {
        "work": str(tmp_path / "work"),
        "snapshot": str(tmp_path / "snapshot"),
        "report": str(tmp_path / "report.json"),
    }


def run_pipeline(sample_dir, dirs, *, edits=None, injections=None, env=None):
    from mvrwb.editor import apply_edit_stream

    apply_edit_stream(sample_dir, dirs["work"], edits)
    pipeline = ReconstructionPipeline(
        dirs["work"], dirs["snapshot"],
        injection_config=injections, crash_env=env or {},
    )
    pipeline.stop_flag = {"stop": False}
    outcome = pipeline.run()
    write_json_atomic(dirs["report"], outcome.report)
    return outcome.report


def deterministic_core(report: dict) -> dict:
    """剔除耗时等非确定性字段后的逐字段可比核心结果。"""
    return {
        "views": report["views"],
        "pairs": report["pairs"],
        "cloud": report["cloud"],
        "invalidation": report["invalidation"],
        "budget_ledger": report["budget_ledger"],
        "failures": report["failures"],
        "injections": report["injections"],
        "recovery": report["recovery"],
    }


def write_config(work_dir: str, updates: dict) -> None:
    path = os.path.join(work_dir, "reconstruction_config.json")
    raw = read_json(path)
    raw.update(updates)
    write_json_atomic(path, raw)


def write_injection(path: str, items: list[dict]) -> str:
    write_json_atomic(path, {"injections": items})
    return path


def reconstruction_artifacts(report: dict) -> dict:
    """跨增量 / 全量必须逐字段一致的重建产物（不含运行路径相关的失效清单）。"""
    return {
        "views": report["views"],
        "pairs": report["pairs"],
        "cloud": report["cloud"],
        "budget_ledger": report["budget_ledger"],
        "failures": report["failures"],
        "injections": report["injections"],
    }
