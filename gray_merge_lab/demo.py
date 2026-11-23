from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import socket
from pathlib import Path
from typing import Any

from .canonical import atomic_write_json, atomic_write_text, canonical_json, sha256_value
from .deterministic import build_backend_parameters, build_input_tensor, stable_route_bucket
from .diagnostics import anomaly_suite, environment_report
from .engine import GatewayEngine, ModelBank
from .errors import LabError
from .fixedpoint import (
    BatchFixedStats,
    aggregate_batch_stats,
    compare_candidates,
    q_to_original_interval,
)
from .gateway import (
    GatewayClient,
    GatewayTCPServer,
    TcpBackendTransport,
    WorkerManager,
)
from .ledger import Ledger
from .receipts import decode_receipt
from .replay import authoritative_recompute, replay_remaining
from .routing import batch_partition_digest, plan_batches, route_sample
from .snapshot import SnapshotStore, snapshot_document
from .specs import LoadedSpec, load_spec


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def default_spec() -> LoadedSpec:
    return load_spec(DATA / "backends.json", DATA / "rules.json", DATA / "samples.ndjson")


def _check_all_ports_bindable(spec: LoadedSpec) -> None:
    failures = []
    for backend in spec.backends:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((backend.host, backend.port))
        except OSError as exc:
            failures.append(f"{backend.backend_id}:{backend.port} ({exc})")
        finally:
            probe.close()
    if failures:
        raise LabError(
            "E_RUNTIME",
            "declared loopback backend ports are not bindable: " + "; ".join(failures),
        )


def _determinism_report(spec: LoadedSpec, scale_exp: int) -> dict[str, Any]:
    params_a = {item.backend_id: build_backend_parameters(item, 1) for item in spec.backends}
    params_b = {item.backend_id: build_backend_parameters(item, 1) for item in spec.backends}
    param_equal = all(
        params_a[key].param_digest == params_b[key].param_digest
        and params_a[key].weights_digest == params_b[key].weights_digest
        for key in params_a
    )
    anchors = [spec.samples[0], spec.samples[-1]]
    input_rows = []
    for sample in anchors:
        backend = spec.backends[0]
        first = build_input_tensor(sample, backend.model_dim)
        second = build_input_tensor(sample, backend.model_dim)
        input_rows.append(
            {
                "sample_id": sample.sample_id,
                "digest": first.digest,
                "repeat_equal": first.digest == second.digest,
            }
        )
    plans = plan_batches(spec.samples, spec.rules)
    canonical_digest = batch_partition_digest(plans)
    shuffled = list(spec.samples)
    random.Random(20261123).shuffle(shuffled)
    shuffled_digest = batch_partition_digest(plan_batches(shuffled, spec.rules))
    route_rows = []
    for rule in spec.rules:
        sample = next(item for item in spec.samples if item.rule_id == rule.rule_id)
        repeated = [route_sample(sample, rule) for _ in range(64)]
        route_rows.append(
            {
                "rule_id": rule.rule_id,
                "sample_id": sample.sample_id,
                "bucket": repeated[0].bucket,
                "destination": repeated[0].destination,
                "stable_over_64_repeats": len({item.destination for item in repeated}) == 1
                and len({item.bucket for item in repeated}) == 1,
            }
        )
    return {
        "parameter_summaries": {key: params_a[key].summary() for key in sorted(params_a)},
        "parameters_bitwise_equal_on_repeat": param_equal,
        "inputs": input_rows,
        "inputs_bitwise_equal_on_repeat": all(item["repeat_equal"] for item in input_rows),
        "canonical_batch_partition_digest": canonical_digest,
        "shuffled_batch_partition_digest": shuffled_digest,
        "partition_order_independent": canonical_digest == shuffled_digest,
        "route_stability": route_rows,
        "scale_exp": scale_exp,
    }


def _fixedpoint_report(engine: GatewayEngine) -> dict[str, Any]:
    records = engine.ledger.batches()
    batch_stats = [
        BatchFixedStats(
            batch_id=item["batch_id"],
            raw_duration_ms=int(item["raw_duration_ms"]),
            raw_samples=int(item["raw_sample_count"]),
            scale_exp=engine.scale_exp,
        )
        for item in records
    ]
    aggregate = aggregate_batch_stats(batch_stats)
    lower = compare_candidates("candidate-batch-size-2", 12345, "candidate-batch-size-4", 12346)
    higher = compare_candidates("candidate-batch-size-2", 12000, "candidate-batch-size-4", 12010)
    interval = q_to_original_interval(aggregate.avg_duration_q, aggregate.scale_exp)
    return {
        "truncation": "toward_zero",
        "scale_exp": aggregate.scale_exp,
        "scale": aggregate.scale,
        "single_value_error_bound_original": aggregate.single_value_error_original,
        "single_value_error_bound_q": 1,
        "batch_aggregate_error_bound_original": aggregate.batch_aggregate_error_original,
        "canonical_aggregate_error_bound_original": aggregate.canonical_aggregate_error_original,
        "raw_duration_sum_ms": aggregate.raw_duration_sum_ms,
        "raw_sample_sum": aggregate.raw_sample_sum,
        "batch_count": aggregate.batch_count,
        "avg_duration_q": aggregate.avg_duration_q,
        "avg_samples_q": aggregate.avg_samples_q,
        "avg_duration_original_interval": interval,
        "within_combined_bound": lower.to_dict(),
        "outside_combined_bound": higher.to_dict(),
        "raw_fixed_round_trip_ok": interval[0]
        <= (aggregate.raw_duration_sum_ms / aggregate.batch_count)
        <= interval[1],
    }


def _receipt_counts(engine: GatewayEngine) -> dict[str, int]:
    counts = {"h1": 0, "h2": 0, "h3": 0}
    for raw in engine.ledger.receipt_rows():
        version = decode_receipt(raw).format
        counts[version] += 1
    return counts


def _write_report(out_dir: Path, report: dict[str, Any]) -> None:
    report["report_digest"] = sha256_value(report)
    atomic_write_json(out_dir / "report.json", report)
    acceptance = report["acceptance"]
    lines = [
        "# 推理网关灰度回执归并诊断台 — 实际验证报告",
        "",
        f"- 报告摘要：`{report['report_digest']}`",
        f"- Python：`{report['environment']['python']}`",
        f"- NumPy：`{report['environment']['numpy']}` / SQLite：`{report['environment']['sqlite3']}`",
        f"- Tkinter：`{report['environment']['tkinter']}`",
        f"- 回环端口绑定：`{report['environment']['loopback_bind']}`",
        f"- 规格版本：后端 `{report['spec']['versions']['backends']}`；规则 `{report['spec']['versions']['rules']}`；样本 `{report['spec']['versions']['samples']}`",
        "",
        "## 验收结果",
        "",
        f"- 三版本回执归并：**{'通过' if acceptance['three_format_merge'] else '失败'}**；账本回执计数 `{report['receipt_counts']}`；热更新批次采纳 merge_seq `{report['merge_showcase']['adopted_merge_seq']}`，参数版本 `{report['merge_showcase']['backend_param_version']}`，否决 `{report['merge_showcase']['rejected_count']}` 条。",
        f"- 灰度去向稳定性：**{'通过' if acceptance['route_stability'] else '失败'}**；64 次重复桶值与去向一致；批次划分摘要 `{report['determinism']['canonical_batch_partition_digest']}`。",
        f"- 整数定点与误差界：**{'通过' if acceptance['fixedpoint'] else '失败'}**；缩放因子 `10^{report['fixedpoint']['scale_exp']}`，单值误差 `< {report['fixedpoint']['single_value_error_bound_original']}`，候选合成界 `{report['fixedpoint']['within_combined_bound']['combined_error_bound_q']}` 个定点单位，误差内稳定决胜为 `{report['fixedpoint']['within_combined_bound']['winner']}`。",
        f"- 恢复重放：**{'通过' if acceptance['recovery_replay'] else '失败'}**；从快照进度 `{report['recovery']['resumed_after_batch_index']}` 后重放 `{report['recovery']['replayed_batches']}` 批。",
        f"- 权威重算：**{'通过' if acceptance['authoritative_recompute'] else '失败'}**；账本 `{report['ledger']['completed_batches']}` 批、内容摘要 `{report['ledger']['content_digest']}`。",
        f"- 异常注入：**{'通过' if acceptance['anomaly_injection'] else '失败'}**；三类主异常与预览/重放异常均被稳定分类。",
        "",
        "## 关键口径",
        "",
        "- 分流桶：`SHA-256(rule_id, client_key)` 前 64 位模 10000 后加 1，得到 1..10000；权重比较为 `bucket <= weight_bp`。",
        "- 归并采纳：最高 `merge_seq`，同序号取最高参数版本，最终按回执摘要字典序稳定决胜。",
        "- 定点除法：正整数一律向零截断；原始整数和与定点值可在账本中互相复核。",
        "- 快照：后端集合/规则/样本/缩放口径任一变化即整体替换，并抛出 `E_SNAPSHOT` 差异列表。",
        "",
        "## 复现命令",
        "",
        "```bat",
        "py -3 -m gray_merge_lab doctor",
        "py -3 -m gray_merge_lab demo --out artifacts\\demo",
        "py -3 -m unittest discover -s tests -v",
        "py -3 -m gray_merge_lab gui",
        "```",
        "",
        "详细结构化结果见 `report.json`，SQLite 账本见 `ledger.sqlite3`，快速视图快照见 `snapshot.json`。",
    ]
    atomic_write_text(out_dir / "REPORT.md", "\n".join(lines) + "\n")


def run_demo(out_dir: str | Path = "artifacts/demo", scale_exp: int = 3) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in (
        "ledger.sqlite3",
        "ledger.sqlite3-wal",
        "ledger.sqlite3-shm",
        "snapshot.json",
        "snapshot.json.rejected.json",
        "checkpoint.json",
    ):
        path = out / name
        if path.exists():
            path.unlink()

    spec = default_spec()
    _check_all_ports_bindable(spec)
    env = environment_report(spec)
    determinism = _determinism_report(spec, scale_exp)

    worker_manager = WorkerManager(spec.backends)
    transport: TcpBackendTransport | None = None
    ledger: Ledger | None = None
    server: GatewayTCPServer | None = None
    epoch_rows: list[dict[str, Any]] = []
    reinfer: dict[str, Any] | None = None
    checkpoint_payload: dict[str, Any] | None = None
    engine: GatewayEngine | None = None
    try:
        worker_manager.start()
        bank = ModelBank(spec)
        transport = TcpBackendTransport(bank)
        ledger = Ledger(out / "ledger.sqlite3")
        engine = GatewayEngine(
            spec,
            ledger,
            scale_exp=scale_exp,
            snapshot_path=out / "snapshot.json",
            transport=transport,
        )
        server = GatewayTCPServer(engine, host="127.0.0.1", port=0)
        server.start_background()

        with GatewayClient(server.host, server.port) as client:
            for sample in spec.samples[:8]:
                response = client.accept(sample.declaration())
                if not response.get("ok"):
                    raise LabError("E_RUNTIME", f"gateway rejected sample: {response}")
            first_flush = client.flush()
            epoch_rows.append(
                {
                    "epoch": 1,
                    "samples": first_flush["samples"],
                    "batches": first_flush["batches"],
                    "batch_rows": first_flush["batch_rows"],
                }
            )
            checkpoint_payload = engine.payload()
            atomic_write_json(
                out / "checkpoint.json",
                snapshot_document(engine.manifest(), checkpoint_payload),
            )

            canary_batch = next(
                item for item in first_flush["batch_rows"] if item["rule_id"] == "search" and item["destination"] == "canary"
            )
            hot = json.loads((DATA / "hot_update.json").read_text(encoding="utf-8"))
            hot_summary = engine.hot_update_backend(hot["backend_id"], hot["changes"])
            reinfer = engine.reinfer_existing_batch(canary_batch["batch_id"])

            for epoch, batch in ((2, spec.samples[8:16]), (3, spec.samples[16:])):
                for sample in batch:
                    response = client.accept(sample.declaration())
                    if not response.get("ok"):
                        raise LabError("E_RUNTIME", f"gateway rejected sample: {response}")
                flushed = client.flush()
                epoch_rows.append(
                    {
                        "epoch": epoch,
                        "samples": flushed["samples"],
                        "batches": flushed["batches"],
                        "batch_rows": flushed["batch_rows"],
                    }
                )
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if transport is not None:
            transport.close()
        worker_manager.stop()

    if engine is None or ledger is None or reinfer is None or checkpoint_payload is None:
        raise LabError("E_RUNTIME", "demo did not complete")

    snapshot_store = SnapshotStore(out / "snapshot.json")
    snapshot_load = snapshot_store.load_or_replace(engine.manifest(), engine.payload())
    snapshot_payload = snapshot_load.payload
    ledger_info = ledger.verify_integrity()
    authoritative = authoritative_recompute(ledger, snapshot_payload)
    recovery = replay_remaining(ledger, checkpoint_payload)
    fixedpoint = _fixedpoint_report(engine)
    anomalies = anomaly_suite(engine)
    conclusion = reinfer["conclusion"]

    all_receipts = ledger.receipt_rows()
    with (out / "receipts.ndjson").open("w", encoding="utf-8", newline="\n") as handle:
        for raw in all_receipts:
            handle.write(canonical_json(raw) + "\n")

    merge_showcase = {
        "batch_id": reinfer["batch_id"],
        "adopted_merge_seq": reinfer["adopted_merge_seq"],
        "backend_param_version": reinfer["backend_param_version"],
        "receipt_count": reinfer["receipt_count"],
        "rejected_count": reinfer["rejected_count"],
        "comparison_order": [
            {
                "format": item["format"],
                "merge_seq": item["merge_seq"],
                "param_version": item["param_version"],
                "receipt_id": item["receipt_id"],
            }
            for item in conclusion["comparison_order"]
        ],
        "rejected": [
            {
                "format": item["format"],
                "merge_seq": item["merge_seq"],
                "param_version": item["param_version"],
                "reason": item["rejection_reason"],
                "sample_discrepancies": item["sample_discrepancies"],
            }
            for item in conclusion["rejected"]
        ],
        "decision_rule": conclusion["decision_rule"],
        "arrival_order_independent": conclusion["arrival_order_independent"],
    }
    acceptance = {
        "three_format_merge": {"h1", "h2", "h3"}.issubset(
            {item["format"] for item in conclusion["comparison_order"]}
        )
            and conclusion["backend_param_version"] == 2
            and conclusion["merge_seq"] == 3,
        "route_stability": all(item["stable_over_64_repeats"] for item in determinism["route_stability"])
        and determinism["partition_order_independent"]
        and determinism["parameters_bitwise_equal_on_repeat"]
        and determinism["inputs_bitwise_equal_on_repeat"],
        "fixedpoint": fixedpoint["raw_fixed_round_trip_ok"]
        and fixedpoint["within_combined_bound"]["error_affected"]
        and not fixedpoint["outside_combined_bound"]["indistinguishable"],
        "recovery_replay": recovery["ok"] and recovery["replayed_batches"] == 6,
        "authoritative_recompute": authoritative["ok"],
        "anomaly_injection": all(item["ok"] for item in anomalies),
    }
    report = {
        "generated_by": "gray_merge_lab.demo",
        "environment": env,
        "spec": {
            "versions": {
                "backends": spec.backend_version(),
                "rules": spec.rules_version(),
                "samples": spec.samples_version(),
            },
            "backend_ids": [item.backend_id for item in spec.backends],
            "rule_ids": [item.rule_id for item in spec.rules],
            "sample_count": len(spec.samples),
        },
        "gateway": {
            "host": server.host if server is not None else "127.0.0.1",
            "port": server.port if server is not None else None,
            "epochs": epoch_rows,
        },
        "determinism": determinism,
        "merge_showcase": merge_showcase,
        "receipt_counts": _receipt_counts(engine),
        "fixedpoint": fixedpoint,
        "ledger": ledger_info,
        "snapshot": {
            "replaced": snapshot_load.replaced,
            "differences": snapshot_load.differences,
            "rule_stats": snapshot_payload.get("rule_stats", {}),
            "progress": snapshot_payload.get("progress", {}),
            "content_digest": json.loads((out / "snapshot.json").read_text(encoding="utf-8"))[
                "content_digest"
            ],
        },
        "authoritative_recompute": {
            "ok": authoritative["ok"],
            "completed_batches": authoritative["completed_batches"],
            "ledger_content_digest": authoritative["ledger_content_digest"],
        },
        "recovery": {
            "ok": recovery["ok"],
            "resumed_after_batch_index": recovery["resumed_after_batch_index"],
            "replayed_batches": recovery["replayed_batches"],
            "batch_ids": recovery["batch_ids"],
        },
        "anomalies": anomalies,
        "acceptance": acceptance,
    }
    _write_report(out, report)
    ledger.close()
    return report


def print_report(report: dict[str, Any], out_dir: str | Path) -> None:
    print("DEMO OK" if all(report["acceptance"].values()) else "DEMO FAILED")
    print(f"  python: {report['environment']['python']}")
    print(f"  sqlite3: {report['environment']['sqlite3']}  numpy: {report['environment']['numpy']}")
    print(f"  tkinter: {report['environment']['tkinter']}")
    print(f"  loopback_bind: {report['environment']['loopback_bind']}")
    for key, value in report["acceptance"].items():
        print(f"  {key}: {'PASS' if value else 'FAIL'}")
    print(f"  receipt_counts: {report['receipt_counts']}")
    print(f"  batch_partition_digest: {report['determinism']['canonical_batch_partition_digest']}")
    print(f"  fixed_scale_exp: {report['fixedpoint']['scale_exp']}")
    print(f"  bounds: single<{report['fixedpoint']['single_value_error_bound_original']} "
          f"combined_q={report['fixedpoint']['within_combined_bound']['combined_error_bound_q']}")
    print(f"  recovery: after {report['recovery']['resumed_after_batch_index']} replayed "
          f"{report['recovery']['replayed_batches']} batches")
    print(f"  ledger: {report['ledger']['content_digest']}")
    print(f"  report: {Path(out_dir) / 'report.json'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic loopback gray-merge demo")
    parser.add_argument("--out", default="artifacts/demo")
    parser.add_argument("--scale-exp", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        report = run_demo(args.out, args.scale_exp)
    except LabError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False, indent=2))
        return 2
    print_report(report, args.out)
    return 0 if all(report["acceptance"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
