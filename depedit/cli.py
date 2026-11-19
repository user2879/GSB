"""命令行入口。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .codecs import write_json_atomic, write_text_atomic
from .config import AppConfig, Injection, load_concurrency
from .diagnostics import roundtrip_compare
from .fingerprints import content_hash, file_sha256
from .injection import InjectionEngine
from .pipeline import Pipeline, fingerprints_of_inputs, commit_snapshot
from .relations import load_mapping
from .replay import apply_edit_sequence, clone_corpus, load_sequence
from .validation import validate_sentence
from .samplegen import generate_samples
from .snapshot import SnapshotStore
from .analysis import model_weight_stats


def _report(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, payload)


def cmd_generate(args: argparse.Namespace) -> int:
    paths = generate_samples(args.data_dir, seed=args.seed, n_sentences=args.n)
    print(json.dumps(paths, ensure_ascii=False, indent=2))
    return 0


def _build_pipeline(args: argparse.Namespace) -> tuple[Pipeline, AppConfig, dict[str, Any]]:
    cfg = AppConfig.load(args.config if args.config else None)
    cfg = load_concurrency(args.concurrency if args.concurrency else None, cfg)
    mapping = load_mapping(Path(args.data_dir) / "relation_map.json")
    injections = Injection.load_list(args.injections)
    engine = InjectionEngine(injections)
    return Pipeline(cfg, mapping, engine), cfg, mapping


def cmd_parse(args: argparse.Namespace) -> int:
    pipeline, cfg, mapping = _build_pipeline(args)
    data_dir = Path(args.data_dir)
    model_fp, mapping_fp = fingerprints_of_inputs(
        data_dir / "model_params.json", data_dir / "relation_map.json"
    )
    if args.source == "v1":
        corpus, failures = pipeline.parse_v1(data_dir / "parse_v1.txt")
    else:
        corpus, failures = pipeline.parse_v2(data_dir / "parse_v2.json")

    result = pipeline.fingerprint_and_serialize(
        corpus, model_fp, mapping_fp, version=args.source
    )
    result.failures = failures + result.failures

    store = SnapshotStore(Path(args.out_dir) / "snapshot")
    diagnostics = roundtrip_compare(
        [corpus.sentences[sid] for sid in corpus.sorted_ids()], cfg, mapping
    )
    manifest = commit_snapshot(
        store,
        result,
        global_version=1,
        model_fp=model_fp,
        mapping_fp=mapping_fp,
        diagnostics=diagnostics,
    )
    report = {
        "parsed_sentences": corpus.sorted_ids(),
        "failure_list": result.failures,
        "invalidation_listing": result.invalidation,
        "reusable_sentences": result.reusable,
        "degradation_count": len(result.degradation),
        "degradation_sample": result.degradation[:10],
        "diagnostics": diagnostics,
        "manifest_global_version": manifest["global_version"],
        "model_fp": model_fp,
        "mapping_fp": mapping_fp,
    }
    _report(Path(args.out_dir) / "parse_report.json", report)
    print(json.dumps({"ok": True, "sentences": len(corpus.sentences),
                      "failures": len(result.failures)}, ensure_ascii=False))
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    pipeline, cfg, mapping = _build_pipeline(args)
    data_dir = Path(args.data_dir)
    if args.direction == "up":
        corpus, failures = pipeline.parse_v1(data_dir / "parse_v1.txt")
        doc, sidecar = pipeline.upgrade_v1_to_v2(corpus)
        write_json_atomic(Path(args.out_dir) / "upgraded_v2.json", doc)
        write_json_atomic(Path(args.out_dir) / "upgraded_v2_sidecar.json", sidecar)
    else:
        corpus, failures = pipeline.parse_v2(data_dir / "parse_v2.json")
        text, losses = pipeline.downgrade_v2_to_v1(corpus)
        write_text_atomic(Path(args.out_dir) / "downgraded_v1.txt", text)
        write_json_atomic(Path(args.out_dir) / "downgrade_losses.json", losses)
    _report(Path(args.out_dir) / "convert_failures.json", failures)
    print(json.dumps({"ok": True, "direction": args.direction,
                      "failures": len(failures)}, ensure_ascii=False))
    return 0


def cmd_edit(args: argparse.Namespace) -> int:
    pipeline, cfg, mapping = _build_pipeline(args)
    data_dir = Path(args.data_dir)
    model_fp, mapping_fp = fingerprints_of_inputs(
        data_dir / "model_params.json", data_dir / "relation_map.json"
    )
    corpus, parse_failures = pipeline.parse_v1(data_dir / "parse_v1.txt")
    edit_failures: list[dict[str, Any]] = []

    # 编辑阶段环注入：对目标句制造真实环并执行真实结构校验。
    for sid in list(corpus.sentences):
        inj = pipeline.injections.inject_edit_cycle(corpus.sentences[sid])
        if inj is not None:
            try:
                validate_sentence(corpus.sentences[sid])
            except Exception as exc:
                edit_failures.append(
                    {
                        "sent_id": sid,
                        "stage": "edit",
                        "business_code": getattr(exc, "code", "E_UNKNOWN"),
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                        "inject_id": inj.inject_id,
                    }
                )
            # 环注入仅用于证明命中，不写入有效状态：重新解析恢复该句。
            repaired, _ = pipeline.parse_v1(data_dir / "parse_v1.txt")
            if sid in repaired.sentences:
                corpus.sentences[sid] = repaired.sentences[sid]

    sequence = load_sequence(args.sequence)
    edit_summary = apply_edit_sequence(corpus, sequence)
    edit_failures.extend(
        {**item, "stage": "edit", "exception_type": "DepEditError"} for item in edit_summary["rejected"]
    )

    result = pipeline.fingerprint_and_serialize(corpus, model_fp, mapping_fp, version="v1-edit")
    diagnostics = roundtrip_compare(
        [corpus.sentences[sid] for sid in corpus.sorted_ids()], cfg, mapping
    )
    store = SnapshotStore(Path(args.out_dir) / "snapshot_edit")
    manifest = commit_snapshot(
        store, result, global_version=2, model_fp=model_fp,
        mapping_fp=mapping_fp, diagnostics=diagnostics,
    )
    report = {
        "parse_failures": parse_failures,
        "edit_failures": edit_failures,
        "edit_summary": edit_summary,
        "invalidation_listing": result.invalidation,
        "reusable_sentences": result.reusable,
        "degradation_count": len(result.degradation),
        "diagnostics": diagnostics,
        "manifest_global_version": manifest["global_version"],
    }
    _report(Path(args.out_dir) / "edit_report.json", report)
    injection_hits = sum(1 for f in edit_failures if f.get("inject_id"))
    print(json.dumps({"ok": True, "applied": len(edit_summary["applied"]),
                      "rejected": len(edit_summary["rejected"]),
                      "injection_hits": injection_hits}, ensure_ascii=False))
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    pipeline, cfg, mapping = _build_pipeline(args)
    data_dir = Path(args.data_dir)
    model_fp, mapping_fp = fingerprints_of_inputs(
        data_dir / "model_params.json", data_dir / "relation_map.json"
    )
    corpus, parse_failures = pipeline.parse_v1(data_dir / "parse_v1.txt")
    sequence = load_sequence(args.sequence)

    # 逐步编辑基线
    baseline = clone_corpus(corpus)
    baseline_summary = apply_edit_sequence(baseline, sequence)

    # 从空白工作区重放：先重新解析再套同一序列
    replay_corpus, replay_failures = pipeline.parse_v1(data_dir / "parse_v1.txt")
    replay_summary = apply_edit_sequence(replay_corpus, sequence)

    identical = content_hash(
        {sid: s.to_dict() for sid, s in baseline.sentences.items()}
    ) == content_hash({sid: s.to_dict() for sid, s in replay_corpus.sentences.items()})

    result = pipeline.fingerprint_and_serialize(
        replay_corpus, model_fp, mapping_fp, version="v1-replay"
    )
    diagnostics = roundtrip_compare(
        [replay_corpus.sentences[sid] for sid in replay_corpus.sorted_ids()], cfg, mapping
    )
    store = SnapshotStore(Path(args.out_dir) / "snapshot_replay")
    commit_snapshot(
        store, result, global_version=1 + len(sequence),
        model_fp=model_fp, mapping_fp=mapping_fp, diagnostics=diagnostics,
    )
    report = {
        "sequence_length": len(sequence),
        "baseline_equals_replay": identical,
        "baseline_summary": baseline_summary,
        "replay_summary": replay_summary,
        "parse_failures": parse_failures + replay_failures,
        "degradation_count": len(result.degradation),
        "diagnostics": diagnostics,
    }
    _report(Path(args.out_dir) / "replay_report.json", report)
    print(json.dumps({"ok": True, "identical": identical,
                      "rejected": len(replay_summary["rejected"])}, ensure_ascii=False))
    return 0


def cmd_fingerprints(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    model_params = json.loads((data_dir / "model_params.json").read_text(encoding="utf-8"))
    out = {
        "utf8_ok": _check_utf8(data_dir),
        "model_params_sha256": file_sha256(data_dir / "model_params.json"),
        "relation_map_sha256": file_sha256(data_dir / "relation_map.json"),
        "model_weight_stats": model_weight_stats(model_params),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def _check_utf8(data_dir: Path) -> bool:
    for path in data_dir.iterdir():
        if path.is_file():
            path.read_text(encoding="utf-8")
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="depedit", description="中文依存解析结果双版本编辑应用")
    parser.add_argument("--data-dir", default="data/sample")
    parser.add_argument("--out-dir", default="out")
    parser.add_argument("--config", default=None)
    parser.add_argument("--concurrency", default="config/concurrency.json")
    parser.add_argument("--injections", default=None,
                        help="注入配置 JSON；缺省不启用注入")

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("generate")
    p_gen.add_argument("--seed", type=int, default=20241120)
    p_gen.add_argument("--n", type=int, default=8)
    p_gen.set_defaults(func=cmd_generate)

    p_parse = sub.add_parser("parse")
    p_parse.add_argument("--source", choices=["v1", "v2"], default="v1")
    p_parse.set_defaults(func=cmd_parse)

    p_conv = sub.add_parser("convert")
    p_conv.add_argument("--direction", choices=["up", "down"], required=True)
    p_conv.set_defaults(func=cmd_convert)

    p_edit = sub.add_parser("edit")
    p_edit.add_argument("--sequence", default="config/edit_sequence.json")
    p_edit.set_defaults(func=cmd_edit)

    p_replay = sub.add_parser("replay")
    p_replay.add_argument("--sequence", default="config/edit_sequence.json")
    p_replay.set_defaults(func=cmd_replay)

    p_fp = sub.add_parser("fingerprints")
    p_fp.set_defaults(func=cmd_fingerprints)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
