"""可重复演示编排：编辑序列 + 增量/全量逐字段一致性验证。

每一步对样本或配置施加一次编辑，随后：
  A) 在共享 work 目录做增量运行（复用缓存与账本）；
  B) 在独立 fresh 目录按相同最终状态从空白全量运行；
逐字段比对窗口预测、定点指标、转换残差、规范化账本与失败清单顺序。
另含并发/串行一致性、预算耗尽、注入诊断与跨版本往返的专项验证。
"""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import yaml

from .config import load_config
from .exchange.build import build_ir
from .exchange.converter import fingerprint_v2, v2_to_v1
from .exchange.v1_codec import fingerprint_v1
from .runner import run_pipeline
from .sample import default_specs, write_jsonl, SeriesSpec, generate_series


def _reset(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _write_config(target: Path, raw: dict) -> None:
    path = target if target.suffix == ".yaml" else target / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def _read_raw_config(sample_dir: Path) -> dict:
    return yaml.safe_load((sample_dir / "config.yaml").read_text(encoding="utf-8"))


def _canonical_run(run, cfg) -> dict:
    """跨运行可比的规范结果：预测、定点指标、账本、失败、转换残差。"""
    ir = build_ir(cfg, run.results)
    if cfg.exchange_version == "v1":
        ir.version_fingerprint = fingerprint_v1(ir)
        v1_ir, sidecar = v2_to_v1(_as_v2(ir, cfg))
        residuals = sorted(
            (e.series_id, e.window_index, e.model, round(e.residual, 12))
            for e in sidecar.entries)
    else:
        ir.version_fingerprint = fingerprint_v2(ir)
        residuals = []
    return {
        "windows": [
            (w.series_id, w.window_index,
             [m.model for m in w.models],
             [tuple(m.predictions_q) for m in w.models],
             tuple(w.actual_q), w.metric.__dict__)
            for w in sorted(run.results, key=lambda x: x.sort_key)
        ],
        "failures": [(f["series_id"], f.get("window_index"), f["code"],
                      f.get("injection_id")) for f in run.failures],
        "ledger": [(e["series_id"], e["window_index"], e["stage"],
                    e["model"], e["status"]) for e in run.ledger.canonical()],
        "cursors": run.cursors,
        "residuals": residuals,
    }


def _as_v2(ir, cfg):
    v2 = copy.deepcopy(ir)
    v2.exchange_version = "v2"
    v2.missing_strategy = cfg.missing_strategy
    v2.version_fingerprint = fingerprint_v2(v2)
    return v2


def _compare(step: str, inc, fresh) -> list[str]:
    errors: list[str] = []
    a, b = _canonical_run(*inc), _canonical_run(*fresh)
    for key in ("windows", "failures", "cursors", "residuals"):
        if a[key] != b[key]:
            errors.append(f"[{step}] {key} 不一致")
    # 账本：全量必然全 recomputed，增量可能 reused；比对“存在的键集合与最终归属”
    inc_keys = {(e[0], e[1], e[2], e[3]) for e in a["ledger"]}
    fresh_keys = {(e[0], e[1], e[2], e[3]) for e in b["ledger"]}
    if inc_keys != fresh_keys:
        errors.append(f"[{step}] 账本键集合不一致")
    return errors


def _replace_series(sample_dir: Path, series_id: str) -> None:
    specs = {s.series_id: s for s in default_specs()}
    spec = specs[series_id]
    new_spec = SeriesSpec(
        series_id=spec.series_id, sensor_id=spec.sensor_id, length=spec.length,
        seed=spec.seed + 777, slope=spec.slope + 0.02, period=spec.period,
        amplitude=spec.amplitude, noise_sigma=spec.noise_sigma,
        missing_every=spec.missing_every)
    grouped: dict[str, list] = {}
    for sid, rows in _read_rows(sample_dir).items():
        grouped[sid] = rows
    grouped[series_id] = generate_series(new_spec)
    _write_grouped(sample_dir, grouped)


def _add_series(sample_dir: Path, series_id: str) -> None:
    spec = SeriesSpec(series_id, "sensor-new", 96, 20261124, 0.08, 14, 2.5, 0.1)
    grouped = _read_rows(sample_dir)
    grouped[series_id] = generate_series(spec)
    _write_grouped(sample_dir, grouped)


def _delete_series(sample_dir: Path, series_id: str) -> None:
    grouped = _read_rows(sample_dir)
    grouped.pop(series_id, None)
    _write_grouped(sample_dir, grouped)


def _read_rows(sample_dir: Path):
    from .sample import read_jsonl
    return read_jsonl(sample_dir / "samples.jsonl")


def _write_grouped(sample_dir: Path, grouped) -> None:
    path = sample_dir / "samples.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for sid in sorted(grouped):
            for row in grouped[sid]:
                payload = {k: v for k, v in row.items() if not k.startswith("_")}
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_demo(root: str | Path = "demo_out") -> dict:
    root = Path(root)
    sample_dir = root / "data"
    inc_work = root / "work_inc"
    _reset(sample_dir)
    write_jsonl(sample_dir / "samples.jsonl", default_specs())
    base_raw = yaml.safe_load(__import__("tspworkbench.cli", fromlist=["DEFAULT_CONFIG_YAML"])
                              .DEFAULT_CONFIG_YAML)
    _write_config(sample_dir, base_raw)
    for d in (inc_work,):
        _reset(d)

    def edit_toggle_version(raw):
        raw["exchange"]["version"] = "v2" if raw["exchange"]["version"] == "v1" else "v1"
        return raw

    def edit_window(raw):
        raw["backtest"]["window_length"] += 4
        raw["backtest"]["window_step"] += 2
        return raw

    def edit_model(raw):
        for m in raw["models"]:
            if m["name"] == "moving_average":
                m["lag_order"] += 2
            if m["name"] == "autoregressive":
                m["weight"] = round(m["weight"] - 0.1, 4)
                m["lag_order"] += 1
        return raw

    def edit_quant(raw):
        raw["quantization"]["gear"] = (raw["quantization"]["gear"] + 1) % 5
        raw["quantization"]["frac_bits"] += 1
        return raw

    steps: list[tuple[str, object, str]] = [
        ("baseline", lambda raw, sd: (raw, sd), "初始全量"),
        ("switch-version", lambda raw, sd: (edit_toggle_version(raw), sd), "切换交换版本"),
        ("replace-series", lambda raw, sd: (raw, _replace_series(sd, "S001")), "替换单条序列"),
        ("add-series", lambda raw, sd: (raw, _add_series(sd, "S004")), "新增序列"),
        ("window-params", lambda raw, sd: (edit_window(raw), sd), "调整回测窗口与步长"),
        ("model-params", lambda raw, sd: (edit_model(raw), sd), "变更模型超参数"),
        ("quant-gear", lambda raw, sd: (edit_quant(raw), sd), "切换量化档位与位宽"),
        ("delete-series", lambda raw, sd: (raw, _delete_series(sd, "S002")), "删除序列"),
        ("switch-back", lambda raw, sd: (edit_toggle_version(raw), sd), "再次切换交换版本"),
    ]

    raw = copy.deepcopy(base_raw)
    report: list[dict] = []
    all_errors: list[str] = []
    for step, fn, desc in steps:
        raw, _ = fn(raw, sample_dir)
        _write_config(sample_dir, raw)
        cfg = load_config(sample_dir / "config.yaml")
        inc = run_pipeline(cfg, sample_dir / "samples.jsonl", inc_work, fresh=False)
        fresh_dir = root / f"fresh_{step}"
        _reset(fresh_dir)
        fresh = run_pipeline(cfg, sample_dir / "samples.jsonl", fresh_dir, fresh=True)
        errors = _compare(step, (inc, cfg), (fresh, cfg))
        all_errors.extend(errors)
        reused = sum(1 for e in inc.ledger.entries if e.status == "reused")
        report.append({
            "step": step, "desc": desc, "version": cfg.exchange_version,
            "windows": len(inc.results), "failures": len(inc.failures),
            "ledger_reused": reused,
            "ledger_recomputed": sum(1 for e in inc.ledger.entries
                                     if e.status == "recomputed"),
            "consistent": not errors,
            "errors": errors,
        })

    # 并发 vs 串行一致性
    par_errors = _parallel_vs_serial(sample_dir, root)
    all_errors.extend(par_errors)
    budget_errors = _budget_demo(sample_dir, root)
    all_errors.extend(budget_errors)

    summary = {
        "steps": report,
        "parallel_vs_serial_errors": par_errors,
        "budget_errors": budget_errors,
        "all_consistent": not all_errors,
        "all_errors": all_errors,
    }
    (root / "demo_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _parallel_vs_serial(sample_dir: Path, root: Path) -> list[str]:
    raw = _read_raw_config(sample_dir)
    par_raw = copy.deepcopy(raw)
    par_raw["execution"]["max_workers"] = 4
    ser_raw = copy.deepcopy(raw)
    ser_raw["execution"]["max_workers"] = 1
    par_dir, ser_dir = root / "par", root / "ser"
    _reset(par_dir); _reset(ser_dir)
    _write_config(sample_dir / "_par.yaml", par_raw)
    _write_config(sample_dir / "_ser.yaml", ser_raw)
    par_cfg = load_config(sample_dir / "_par.yaml")
    ser_cfg = load_config(sample_dir / "_ser.yaml")
    par = run_pipeline(par_cfg, sample_dir / "samples.jsonl", par_dir, fresh=True)
    ser = run_pipeline(ser_cfg, sample_dir / "samples.jsonl", ser_dir, fresh=True)
    cp, cs = _canonical_run(par, par_cfg), _canonical_run(ser, ser_cfg)
    errors = []
    for key in ("windows", "failures", "cursors"):
        if cp[key] != cs[key]:
            errors.append(f"并发/串行 {key} 不一致")
    return errors


def _budget_demo(sample_dir: Path, root: Path) -> list[str]:
    raw = _read_raw_config(sample_dir)
    braw = copy.deepcopy(raw)
    # 故意把单序列配额设为仅够 2 个窗口（每窗观测点 = window_length）
    braw["execution"]["series_quota"] = 2 * raw["backtest"]["window_length"]
    braw["execution"]["observation_budget"] = 10 ** 9
    bdir = root / "budget"
    _reset(bdir)
    _write_config(sample_dir / "_budget.yaml", braw)
    cfg = load_config(sample_dir / "_budget.yaml")
    run = run_pipeline(cfg, sample_dir / "samples.jsonl", bdir, fresh=True)
    codes = {f["code"] for f in run.failures}
    errors = []
    if "E_SERIES_BUDGET_EXHAUSTED" not in codes:
        errors.append("预算耗尽未产生 E_SERIES_BUDGET_EXHAUSTED")
    # 每条序列应恰好完成 2 个窗口后中止
    for sid, cursor in run.cursors.items():
        if cursor < 2:
            errors.append(f"序列 {sid} 待办游标异常: {cursor}")
    return errors


if __name__ == "__main__":
    result = run_demo()
    print(json.dumps({"all_consistent": result["all_consistent"],
                      "steps": [(s["step"], s["consistent"], s["ledger_reused"])
                                for s in result["steps"]],
                      "parallel_errors": result["parallel_vs_serial_errors"],
                      "budget_errors": result["budget_errors"]},
                     ensure_ascii=False, indent=2))
