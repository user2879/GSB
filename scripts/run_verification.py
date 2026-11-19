"""端到端确定性验证：并发一致、增量失效、快照过期、双版本转换、回放。

报告以临时文件 + 原子替换写盘。
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from depedit.analysis import model_weight_stats
from depedit.cache import IncrementalCache
from depedit.codecs import (
    load_json,
    v1_parse_document,
    v1_serialize_document,
    v2_parse_document,
    v2_serialize_document,
    load_text,
    write_json_atomic,
)
from depedit.config import AppConfig, Injection
from depedit.diagnostics import roundtrip_compare
from depedit.edits import set_head
from depedit.errors import ESnapshotStale
from depedit.fingerprints import content_hash, file_sha256
from depedit.injection import InjectionEngine
from depedit.models import Corpus
from depedit.pipeline import Pipeline, fingerprints_of_inputs
from depedit.relations import load_mapping
from depedit.replay import apply_edit_sequence, clone_corpus, load_sequence
from depedit.samplegen import generate_samples
from depedit.snapshot import SnapshotStore


def make_pipeline(cfg: AppConfig, mapping: dict, injections=None) -> Pipeline:
    return Pipeline(cfg, mapping, InjectionEngine(injections or []))


def main() -> int:
    work = ROOT / "out" / "verify"
    import shutil

    shutil.rmtree(work, ignore_errors=True)
    sample = work / "sample"
    generate_samples(sample, seed=20241120, n_sentences=8)
    mapping = load_mapping(sample / "relation_map.json")
    model_fp, mapping_fp = fingerprints_of_inputs(
        sample / "model_params.json", sample / "relation_map.json"
    )
    injections = Injection.load_list(ROOT / "config" / "injections.json")
    report: dict = {"checks": []}

    def record(name: str, ok: bool, detail=None) -> None:
        report["checks"].append({"name": name, "ok": bool(ok), "detail": detail})

    # 1) 多并发度逐字段一致（v1 解析 + 序列化 + 失败清单顺序）
    runs = {}
    for workers in (1, 2, 4, 8, 16):
        cfg = AppConfig()
        cfg.max_workers = workers
        pipe = make_pipeline(cfg, mapping, injections)
        corpus, failures = pipe.parse_v1(sample / "parse_v1.txt")
        result = pipe.fingerprint_and_serialize(corpus, model_fp, mapping_fp, "v1")
        runs[workers] = {
            "order": corpus.sorted_ids(),
            "artifacts": {sid: result.artifacts[sid] for sid in sorted(result.artifacts)},
            "failures": failures + result.failures,
            "invalidation": result.invalidation,
        }
    base = runs[1]
    concurrency_ok = all(runs[w] == base for w in (2, 4, 8, 16))
    record("concurrency_field_identical_v1", concurrency_ok,
           {"workers_tested": [1, 2, 4, 8, 16], "failure_count": len(base["failures"])})

    # 2) v2 解析并发一致
    v2_runs = {}
    for workers in (1, 8):
        cfg = AppConfig()
        cfg.max_workers = workers
        pipe = make_pipeline(cfg, mapping, injections)
        corpus, failures = pipe.parse_v2(sample / "parse_v2.json")
        v2_runs[workers] = {"order": corpus.sorted_ids(), "failures": failures}
    record("concurrency_field_identical_v2", v2_runs[1] == v2_runs[8],
           {"v2_failures": v2_runs[1]["failures"]})

    # 3) 六类注入命中并映射业务码
    hit_codes = {f["business_code"] for f in base["failures"]}
    expected = {
        "E_FIXED_WIDTH_FIELD", "E_HEAD_OUT_OF_RANGE", "E_SCORE_RANGE",
        "E_RELATION_UNKNOWN", "E_SCORE_NON_FINITE",
    }
    record("injection_business_codes", expected.issubset(hit_codes),
           {"hit": sorted(hit_codes), "with_inject_id": all(f["inject_id"] for f in base["failures"])})

    # 编辑阶段环注入
    cfg = AppConfig()
    pipe = make_pipeline(cfg, mapping, injections)
    corpus, _ = pipe.parse_v1(sample / "parse_v1.txt")
    inj = pipe.injections.inject_edit_cycle(corpus.sentences["S003"])
    from depedit.validation import validate_sentence

    cycle_code = None
    try:
        validate_sentence(corpus.sentences["S003"])
    except Exception as exc:
        cycle_code = getattr(exc, "code", None)
    record("edit_cycle_injection", cycle_code == "E_CYCLE_DETECTED",
           {"inject_id": inj.inject_id if inj else None})

    # 4) 增量失效精确范围
    clean_cfg = AppConfig()
    clean_pipe = make_pipeline(clean_cfg, mapping)
    clean_corpus, _ = clean_pipe.parse_v1(sample / "parse_v1.txt")
    cache = IncrementalCache(model_fp, mapping_fp, "v1")
    clean_pipe.fingerprint_and_serialize(clean_corpus, model_fp, mapping_fp, "v1", cache)

    cache.invalidate_sample_replaced(clean_corpus, {"S002"})
    reusable_after_sample = set(cache.reusable_sentences())
    record("invalidation_sample_scoped",
           "S002" not in reusable_after_sample and "S001" in reusable_after_sample,
           {"reusable": sorted(reusable_after_sample)})

    target_rel = next(iter(mapping["named_relations"]))
    mapping_report = cache.invalidate_mapping_change(clean_corpus, {target_rel})
    scoped_sents = {item["sent_id"] for item in mapping_report}
    expected_sents = {
        sid for sid in clean_corpus.sentences
        if any(a.relation == target_rel for a in clean_corpus.sentences[sid].arcs)
    }
    record("invalidation_mapping_scoped_arcs", scoped_sents == expected_sents,
           {"relation": target_rel, "sentences": sorted(scoped_sents)})

    cache.invalidate_model_change(clean_corpus, "new-model-fp")
    record("invalidation_model_all", cache.reusable_sentences() == [], {})

    # 5) 快照提交与过期恢复
    snap_corpus, _ = make_pipeline(AppConfig(), mapping).parse_v1(sample / "parse_v1.txt")
    pipe = make_pipeline(AppConfig(), mapping)
    pres = pipe.fingerprint_and_serialize(snap_corpus, model_fp, mapping_fp, "v1")
    store = SnapshotStore(work / "snap")
    store.commit(snap_corpus, global_version=5, model_fp=model_fp,
                 mapping_fp=mapping_fp, artifacts=pres.artifacts,
                 degradation=pres.degradation)
    store.verify_recovery(snap_corpus, expected_version=5,
                          model_fp=model_fp, mapping_fp=mapping_fp)
    stale = False
    try:
        store.verify_recovery(snap_corpus, expected_version=6,
                              model_fp=model_fp, mapping_fp=mapping_fp)
    except ESnapshotStale:
        stale = True
    preserved = store.load_manifest()["global_version"] == 5
    record("snapshot_stale_preserved", stale and preserved,
           {"manifest_version_after_reject": store.load_manifest()["global_version"]})

    # 6) 双版本转换与信息保留
    up_corpus, _ = make_pipeline(AppConfig(), mapping).parse_v1(sample / "parse_v1.txt")
    v2_doc, sidecar = v2_serialize_document(
        [up_corpus.sentences[s] for s in up_corpus.sorted_ids()], AppConfig()
    )
    v1_text, losses = v1_serialize_document(
        [up_corpus.sentences[s] for s in up_corpus.sorted_ids()],
        AppConfig(), mapping["v2_to_v1"],
    )
    record("upgrade_null_heads_sidecar",
           len({x["sent_id"] for x in sidecar["null_heads"]}) == len(up_corpus.sentences),
           {"null_head_arcs": len(sidecar["null_heads"])})
    record("downgrade_origin_loss_recorded",
           any(x["type"] == "origin_adjustment" for x in losses),
           {"loss_entries": len(losses)})

    diag = roundtrip_compare(
        [up_corpus.sentences[s] for s in up_corpus.sorted_ids()], AppConfig(), mapping
    )
    record("roundtrip_zero_head_deviation",
           set(diag["head_deviation_histogram"]) == {"0"},
           {"histogram": diag["head_deviation_histogram"],
            "max_score_deviation": diag["max_score_deviation"],
            "relation_failures": diag["relation_restore_failures"]})

    # 7) 编辑序列回放一致
    base_corpus, _ = make_pipeline(AppConfig(), mapping).parse_v1(sample / "parse_v1.txt")
    sequence = load_sequence(ROOT / "config" / "edit_sequence.json")
    baseline = clone_corpus(base_corpus)
    apply_edit_sequence(baseline, sequence)
    replayed = clone_corpus(base_corpus)
    replay_summary = apply_edit_sequence(replayed, sequence)
    identical = content_hash(
        {sid: s.to_dict() for sid, s in baseline.sentences.items()}
    ) == content_hash({sid: s.to_dict() for sid, s in replayed.sentences.items()})
    record("replay_identical_state", identical,
           {"applied": len(replay_summary["applied"]),
            "rejected": replay_summary["rejected"]})

    # 8) NumPy 模型统计可用
    stats = model_weight_stats(load_json(sample / "model_params.json"))
    record("numpy_model_stats", stats["shape"] == [4, 64],
           {"global_std": stats["global_std"]})

    report["summary"] = {
        "total": len(report["checks"]),
        "passed": sum(1 for c in report["checks"] if c["ok"]),
        "model_sha256": file_sha256(sample / "model_params.json"),
        "mapping_sha256": file_sha256(sample / "relation_map.json"),
    }
    out_path = ROOT / "out" / "verification_report.json"
    write_json_atomic(out_path, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    failed = [c["name"] for c in report["checks"] if not c["ok"]]
    if failed:
        print("FAILED:", failed)
        return 1
    print("ALL CHECKS PASSED ->", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
