"""实验台端到端与单元测试。"""

import json
import os
import shutil
import signal
import subprocess
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fusion_bench.config import load_config  # noqa: E402
from fusion_bench.errors import (  # noqa: E402
    E_BUDGET_EXHAUSTED, E_CONFIG_MISMATCH, E_IMAGE_FORMAT,
    E_SAMPLE_INVALID)
from fusion_bench.fixedpoint import div_round, exp_fixed, scale_of  # noqa: E402
from fusion_bench.injection import InjectionPlan  # noqa: E402
from fusion_bench.pipeline import Pipeline  # noqa: E402

CONFIG = os.path.join(ROOT, "configs", "fusion_config.yaml")
INJECT = os.path.join(ROOT, "configs", "injection.yaml")
ARTIFACTS = ("results.jsonl", "failures.jsonl", "budget_ledger.jsonl",
             "cache_evictions.jsonl", "cache_recomputes.jsonl",
             "pending_queue.json", "summary.json")


@pytest.fixture(scope="session")
def samples_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("samples")
    subprocess.check_call(
        [sys.executable, os.path.join(ROOT, "scripts", "generate_samples.py"),
         "--output", str(out), "--count", "32"])
    return str(out)


def run_pipeline(samples_dir, output_dir, config=CONFIG, inject=INJECT,
                 budget=None, workers=None):
    cfg = load_config(config)
    plan = InjectionPlan.load(inject)
    pipe = Pipeline(cfg, plan, samples_dir, str(output_dir),
                    budget_override=budget, workers_override=workers)
    return pipe.run()


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------- 定点运算 ----------

def test_div_round_half_even():
    assert div_round(5, 2) == 2      # 2.5 -> 2（偶）
    assert div_round(7, 2) == 4      # 3.5 -> 4（偶）
    assert div_round(6, 4) == 2      # 1.5 -> 2
    assert div_round(-5, 2) == -2
    assert div_round(10, 4) == 2     # 2.5 -> 2
    assert div_round(11, 4) == 3


def test_exp_fixed_deterministic():
    scale = scale_of(16)
    assert exp_fixed(0, 16) == scale
    assert exp_fixed(-scale, 16) == exp_fixed(-scale, 16)
    assert abs(exp_fixed(-scale, 16) - 24109) <= 2  # e^-1 ≈ 0.3679
    assert exp_fixed(-20 * scale, 16) == 0          # 深度下溢 -> 0


# ---------- 串并行一致性 ----------

def test_serial_parallel_identical(samples_dir, tmp_path):
    out_s = tmp_path / "serial"
    out_p = tmp_path / "parallel"
    assert run_pipeline(samples_dir, out_s, workers=1) == 0
    assert run_pipeline(samples_dir, out_p, workers=8) == 0
    for name in ARTIFACTS:
        with open(out_s / name, "rb") as fa, open(out_p / name, "rb") as fb:
            assert fa.read() == fb.read(), f"{name} differs"


def test_repeat_run_identical(samples_dir, tmp_path):
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    run_pipeline(samples_dir, out_a)
    run_pipeline(samples_dir, out_b)
    for name in ARTIFACTS:
        with open(out_a / name, "rb") as fa, open(out_b / name, "rb") as fb:
            assert fa.read() == fb.read(), f"{name} differs"


# ---------- 特征与产物 ----------

def test_results_and_metadata(samples_dir, tmp_path):
    out = tmp_path / "run"
    assert run_pipeline(samples_dir, out) == 0
    results = read_jsonl(out / "results.jsonl")
    assert len(results) == 27  # 32 - 5 个注入失败样本
    assert [r["sample_id"] for r in results] == sorted(
        r["sample_id"] for r in results)
    for rec in results:
        meta = rec["feature_meta"]
        assert meta["frac_bits"] == 16
        assert meta["rounding"] == "half_even"
        assert meta["quant_levels"] == 256
        assert rec["commit_verified"] is True
    # 默认补齐样本：sample_000 断言为空
    rec0 = next(r for r in results if r["sample_id"] == "sample_000")
    assert rec0["text_info"]["used_default"] is True
    assert rec0["text_info"]["filled_positions"] == list(range(16))
    # 冲突断言样本：sample_000 无，sample_005 有（idx 5 % 5 == 0）
    rec5 = next(r for r in results if r["sample_id"] == "sample_005")
    assert rec5["text_info"]["conflicts"], "expected conflict reduction"
    # 排名同分按样本 ID 字典序
    ranking = read_json(out / "summary.json")["ranking"]
    keys = [(-r["score"], r["sample_id"]) for r in ranking]
    assert keys == sorted(keys)


def test_cache_eviction_and_recompute(samples_dir, tmp_path):
    out = tmp_path / "run"
    run_pipeline(samples_dir, out)
    evictions = read_jsonl(out / "cache_evictions.jsonl")
    recomputes = read_jsonl(out / "cache_recomputes.jsonl")
    assert evictions, "expected eviction audit entries"
    assert recomputes, "expected recompute audit entries"
    assert all(r["verified_elementwise"] for r in recomputes)
    seqs = [e["seq"] for e in evictions]
    assert seqs == sorted(seqs)


def test_injections_all_hit(samples_dir, tmp_path):
    out = tmp_path / "run"
    run_pipeline(samples_dir, out)
    diag = read_json(out / "diagnostics.json")
    hits = {inj["id"]: inj["hit"] for inj in diag["injections"]}
    assert all(hits.values()), f"unhit injections: {hits}"
    failures = read_jsonl(out / "failures.jsonl")
    by_sample = {f["sample_id"]: f for f in failures}
    assert by_sample["sample_007"]["code"] == E_SAMPLE_INVALID
    assert by_sample["sample_011"]["code"] == E_IMAGE_FORMAT
    assert by_sample["sample_013"]["code"] == E_BUDGET_EXHAUSTED
    assert by_sample["sample_017"]["code"] == E_IMAGE_FORMAT
    assert by_sample["sample_019"]["code"] == E_SAMPLE_INVALID
    assert by_sample["sample_017"]["injection_id"] == "INJ-TRUNCBLOCK-004"
    assert by_sample["sample_019"]["injection_id"] == "INJ-CORRUPT-005"


# ---------- 预算与续跑 ----------

def test_budget_exhaustion_and_resume(samples_dir, tmp_path):
    out = tmp_path / "budget"
    fresh = tmp_path / "fresh"
    run_pipeline(samples_dir, out, budget=100000)
    pending = read_json(out / "pending_queue.json")["pending_samples"]
    assert pending, "expected pending samples after budget exhaustion"
    failures = read_jsonl(out / "failures.jsonl")
    assert any(f["code"] == E_BUDGET_EXHAUSTED for f in failures)
    # 续跑：余量继续消耗，最终产物与全新全量运行一致（台账除外，
    # 台账忠实记录两轮不同预算上限下的扣减历史）
    run_pipeline(samples_dir, out)
    run_pipeline(samples_dir, fresh)
    for name in ("results.jsonl", "failures.jsonl", "cache_evictions.jsonl",
                 "cache_recomputes.jsonl", "pending_queue.json",
                 "summary.json"):
        with open(out / name, "rb") as fa, open(fresh / name, "rb") as fb:
            assert fa.read() == fb.read(), f"{name} differs after resume"
    ledger = read_jsonl(out / "budget_ledger.jsonl")
    assert [e["seq"] for e in ledger] == list(range(len(ledger)))
    fresh_costs = [e["cost"] for e in read_jsonl(fresh / "budget_ledger.jsonl")]
    assert [e["cost"] for e in ledger] == fresh_costs


def test_config_mismatch_refuses_resume(samples_dir, tmp_path):
    out = tmp_path / "mm"
    run_pipeline(samples_dir, out)
    alt = tmp_path / "alt.yaml"
    with open(CONFIG, encoding="utf-8") as fh:
        alt.write_text(fh.read().replace("visual_dim: 12",
                                         "visual_dim: 14"),
                       encoding="utf-8")
    code = run_pipeline(samples_dir, out, config=str(alt))
    assert code == 3
    failures = read_jsonl(out / "failures.jsonl")
    assert failures[0]["code"] == E_CONFIG_MISMATCH


# ---------- 非法输入 ----------

def test_invalid_inputs(samples_dir, tmp_path):
    bad = tmp_path / "bad_samples"
    shutil.copytree(samples_dir, bad)
    # 坏 PNG 头
    with open(bad / "images" / "sample_002.png", "wb") as fh:
        fh.write(b"NOTAPNG" + b"\x00" * 64)
    # 重复样本 ID
    with open(bad / "descriptions.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"sample_id": "sample_003",
                             "assertions": []}) + "\n")
    # 无文本描述的图像
    shutil.copy(bad / "images" / "sample_004.png",
                bad / "images" / "sample_900.png")
    out = tmp_path / "bad_out"
    assert run_pipeline(str(bad), out) == 0
    failures = read_jsonl(out / "failures.jsonl")
    codes = {(f["sample_id"], f["code"]) for f in failures}
    assert ("sample_002", E_IMAGE_FORMAT) in codes
    assert ("sample_003", E_SAMPLE_INVALID) in codes
    assert ("sample_900", E_SAMPLE_INVALID) in codes
    # 其余样本不受影响
    results = read_jsonl(out / "results.jsonl")
    assert len(results) == 27 - 2


def test_config_invalid(tmp_path, samples_dir):
    bad_cfg = tmp_path / "bad.yaml"
    with open(CONFIG, encoding="utf-8") as fh:
        bad_cfg.write_text(fh.read().replace("max_workers: 4",
                                             "max_workers: 0"),
                           encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "main.py"),
         "--config", str(bad_cfg), "--samples", samples_dir,
         "run", "--output", str(tmp_path / "out")],
        capture_output=True, text=True)
    assert proc.returncode == 2
    assert "E_CONFIG_INVALID" in proc.stderr


# ---------- 边界判定 ----------

def test_borderline_rule():
    from fusion_bench.fusion import fuse
    import numpy as np
    cfg = load_config(CONFIG)
    cfg.decision_threshold = 0  # 全零特征 -> 得分 0 == 阈值
    zeros_v = np.zeros(cfg.visual_dim, dtype=np.int64)
    zeros_t = np.zeros(cfg.text_dim, dtype=np.int64)
    fused = fuse(zeros_v, zeros_t, cfg)
    assert fused["score"] == 0
    assert fused["borderline"] is True
    assert fused["decision"] == "accept"  # 恰等于阈值 -> accept 并标注边界


# ---------- 信号退出 ----------

def test_signal_exit_zero_and_resume(samples_dir, tmp_path):
    out = tmp_path / "sig"
    proc = subprocess.Popen(
        [sys.executable, os.path.join(ROOT, "main.py"),
         "--samples", samples_dir, "--stage-delay-ms", "60",
         "run", "--output", str(out)])
    time.sleep(1.2)
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=30) == 0
    manifest = read_json(out / "snapshots" / "manifest.json")
    assert manifest["completed_samples"], "expected partial progress"
    # 续跑完成，与全新全量运行一致
    fresh = tmp_path / "fresh"
    assert run_pipeline(samples_dir, out) == 0
    run_pipeline(samples_dir, fresh)
    for name in ARTIFACTS:
        with open(out / name, "rb") as fa, open(fresh / name, "rb") as fb:
            assert fa.read() == fb.read(), f"{name} differs after resume"
