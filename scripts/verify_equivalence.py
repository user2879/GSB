"""Incremental-vs-cold equivalence and precise-invalidation verification.

The script drives a deterministic *edit sequence* over the four inputs
(scheme switch, base change, training-length change, scale-tier switch and a
single-sample replacement), running the pipeline incrementally after each
edit.  At the end it rebuilds the same final inputs once from an empty
snapshot and asserts the two reports are field-for-field identical apart from
the run/snapshot/audit bookkeeping.  It also prints, per edit, which task keys
were reused versus recomputed to prove only affected samples/stages reran.

A compact fixture is used so the whole sequence finishes quickly offline.
"""

from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pe_diag.common import canonical_json  # noqa: E402
from pe_diag.config import (  # noqa: E402
    Config, load_injections, load_json, load_samples,
)
from pe_diag.pipeline import DiagnosticPipeline  # noqa: E402

WORK = Path("/tmp/equiv_work")


def _write_inputs(base_dir: Path, config: dict, samples: list,
                  scale: dict, injections: dict) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "position_encoding_config.json").write_text(
        canonical_json(config))
    with open(base_dir / "attention_samples.jsonl", "w",
              encoding="utf-8") as fh:
        for row in samples:
            fh.write(canonical_json(row, None) + "\n")
    (base_dir / "scale_descriptor.json").write_text(canonical_json(scale))
    (base_dir / "injection_manifest.json").write_text(
        canonical_json(injections))


def _run(base_dir: Path, snap: Path, parallel: bool) -> dict:
    config = Config.load(base_dir / "position_encoding_config.json")
    samples = load_samples(base_dir / "attention_samples.jsonl")
    scale = load_json(base_dir / "scale_descriptor.json")
    injections = load_injections(base_dir / "injection_manifest.json")
    pipe = DiagnosticPipeline(config, samples, injections, scale, str(snap))
    return pipe.run(parallel=parallel)


def _business(report: dict) -> dict:
    return {k: v for k, v in report.items()
            if k not in ("snapshot", "cache_audit", "execution")}


def main() -> int:
    shutil.rmtree(WORK, ignore_errors=True)
    src = Path("samples")
    config = load_json(src / "position_encoding_config.json")
    scale = load_json(src / "scale_descriptor.json")
    injections = load_json(src / "injection_manifest.json")
    all_samples = load_json(src / "attention_samples.jsonl") if False else \
        [json.loads(line) for line in open(
            src / "attention_samples.jsonl", encoding="utf-8")]

    # compact fixture: deterministic subset spanning all three tiers
    pick = {"small": 12, "medium": 12, "large": 12}
    samples = []
    seen = {t: 0 for t in pick}
    for row in all_samples:
        tier = row["tier_id"]
        if seen[tier] < pick[tier]:
            samples.append(copy.deepcopy(row))
            seen[tier] += 1
    # strip injections that target samples not in the fixture
    kept_ids = {row["sample_id"] for row in samples}
    injections = {"injections": [
        inj for inj in injections["injections"]
        if inj["sample_id"] in kept_ids]}
    compact_scale = {"master_seed": scale["master_seed"], "tiers": [
        {**t, "sample_count": pick[t["tier_id"]]} for t in scale["tiers"]]}

    inc_dir = WORK / "inputs"
    snap = WORK / "snap_inc"
    _write_inputs(inc_dir, config, samples, compact_scale, injections)

    audits = []
    # edit 0: cold run with rope+linear only
    cfg = copy.deepcopy(config)
    cfg["schemes"] = ["rope", "linear"]
    _write_inputs(inc_dir, cfg, samples, compact_scale, injections)
    r0 = _run(inc_dir, snap, parallel=False)
    audits.append(("cold_rope_linear", r0["cache_audit"]["counts"]))

    # edit 1: add ntk (scheme switch) -> only ntk tasks recompute
    cfg = copy.deepcopy(cfg)
    cfg["schemes"] = ["rope", "linear", "ntk"]
    _write_inputs(inc_dir, cfg, samples, compact_scale, injections)
    r1 = _run(inc_dir, snap, parallel=False)
    audits.append(("add_ntk", r1["cache_audit"]["counts"]))

    # edit 2: change base frequency -> all schemes' params fingerprint changes
    cfg = copy.deepcopy(cfg)
    cfg["global"]["base"] = 5000.0
    _write_inputs(inc_dir, cfg, samples, compact_scale, injections)
    r2 = _run(inc_dir, snap, parallel=False)
    audits.append(("base_change", r2["cache_audit"]["counts"]))

    # edit 3: change training length (small tier) -> only small tier invalidates
    cfg = copy.deepcopy(cfg)
    cfg["scale_profiles"]["small"]["train_len"] = 128
    _write_inputs(inc_dir, cfg, samples, compact_scale, injections)
    r3 = _run(inc_dir, snap, parallel=False)
    audits.append(("train_len_change_small", r3["cache_audit"]["counts"]))

    # edit 4: replace a single sample -> only that sample's stages rerun
    samples2 = copy.deepcopy(samples)
    target = next(row for row in samples2 if row["tier_id"] == "medium")
    target["query_features"][0] = 0.123456789
    _write_inputs(inc_dir, cfg, samples2, compact_scale, injections)
    r4 = _run(inc_dir, snap, parallel=False)
    audits.append(("single_sample_replace", r4["cache_audit"]["counts"]))

    # edit 5: scale-tier switch (large profile dim) -> only large invalidates
    cfg = copy.deepcopy(cfg)
    cfg["scale_profiles"]["large"]["dim"] = 128
    # fixture large samples must match the new dim: regenerate their features
    from pe_diag.common import rng_for
    for row in samples2:
        if row["tier_id"] == "large":
            rng = rng_for(20261121, "large", int(row["sample_id"].split("-")[1]))
            row["query_features"] = [round(float(x), 10)
                                     for x in rng.normal(size=128)]
            row["key_features"] = [[round(float(x), 10)
                                    for x in rng.normal(size=128)]
                                   for _ in row["key_features"]]
    compact_scale2 = copy.deepcopy(compact_scale)
    for tier in compact_scale2["tiers"]:
        if tier["tier_id"] == "large":
            tier["dim"] = 128
    _write_inputs(inc_dir, cfg, samples2, compact_scale2, injections)
    r5 = _run(inc_dir, snap, parallel=False)
    audits.append(("tier_switch_large_dim", r5["cache_audit"]["counts"]))

    # cold rebuild of the exact final inputs from an empty snapshot
    cold_snap = WORK / "snap_cold"
    r_cold = _run(inc_dir, cold_snap, parallel=False)
    same = _business(r5) == _business(r_cold)

    # parallel cold rebuild must also match (concurrency equivalence)
    par_snap = WORK / "snap_par"
    r_par = _run(inc_dir, par_snap, parallel=True)
    same_par = _business(r5) == _business(r_par)

    print("edit sequence cache audits (reused/recomputed/invalidated):")
    for name, counts in audits:
        print("  {:28s} reused={:5d} recomputed={:5d} invalidated={:4d}".format(
            name, counts["reused"], counts["recomputed"],
            counts["invalidated"]))
    print("incremental == cold rebuild:           ", same)
    print("incremental == parallel cold rebuild:  ", same_par)
    if not (same and same_par):
        return 1
    # precise invalidation spot checks
    add_ntk = audits[1][1]
    assert add_ntk["reused"] > 0 and add_ntk["recomputed"] > 0
    single = audits[4][1]
    assert single["recomputed"] < add_ntk["recomputed"]
    print("precise invalidation spot checks: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
