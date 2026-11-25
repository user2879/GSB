"""Acceptance demo: the four required local verification scenarios.

1. three-format (r1/r2/r3) receipt normalization
2. hot-update parameter version isolation
3. invalidation propagation along worker-version -> batch -> request
4. forced-termination recovery and byte-identical replay
"""
from __future__ import annotations

import json
import os
import tempfile

from grayroute.batching import build_batch_context, plan_batches
from grayroute.merger import merge_receipts, propagate_invalidation
from grayroute.receipt import normalize_raw
from grayroute.recovery import classify_batches, replay_unfinished
from grayroute.ledger import Ledger
from grayroute.receipt import encode_receipt
from grayroute.sharding import attribute_all
from grayroute.spec import load_spec

ROOT = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    spec = load_spec(os.path.join(ROOT, "config"),
                     os.path.join(ROOT, "data", "requests.ndjson"))
    hits = attribute_all(spec)
    batches = plan_batches(spec, hits)
    requests_by_id = {q.request_id: q for q in spec.requests}
    known = set(requests_by_id)
    report: dict = {}

    # 1. three-format normalization and stable merge
    target = next(b for b in batches if len(b.request_ids) >= 2)
    ctx = build_batch_context(spec, target, requests_by_id)
    rid = target.request_ids[0]
    normalized = {}
    for fv in (1, 2, 3):
        raw = encode_receipt(fv, ctx, rid,
                             ctx.output_summary_by_request[rid], 12)
        normalized[f"r{fv}"] = normalize_raw(raw, ctx, known).to_dict()
    decision = merge_receipts(
        [(i, normalize_raw(encode_receipt(fv, ctx, rid,
                                          ctx.output_summary_by_request[rid],
                                          12), ctx, known))
         for i, fv in enumerate((1, 2, 3))], spec.merge_rule)
    report["1_three_format_normalization"] = {
        "r1_filled_fields": normalized["r1"]["filled_fields"],
        "r2_filled_fields": normalized["r2"]["filled_fields"],
        "r3_filled_fields": normalized["r3"]["filled_fields"],
        "adopted_format": f"r{decision.adopted.format_version}",
        "rejected_count": len(decision.rejected),
    }

    # 2. hot-update isolation: no batch mixes parameter versions
    mixed = [b.batch_id for b in batches
             if len({hits[r].param_version_seq for r in b.request_ids}) > 1]
    report["2_hot_update_isolation"] = {
        "batches": len(batches),
        "batches_mixing_versions": mixed,
        "versions_per_worker": {
            worker: sorted({b.param_version_seq for b in batches
                            if b.worker_id == worker})
            for worker in sorted({b.worker_id for b in batches})
        },
    }

    # 3. invalidation propagation
    invalid_versions = {("worker-gray-b", 2)}
    propagation = propagate_invalidation(batches, {}, invalid_versions)
    report["3_invalidation_propagation"] = {
        "invalid_worker_versions": sorted(map(list, invalid_versions)),
        "invalid_batches": sorted({r["batch_id"] for r in propagation}),
        "invalid_requests": sorted({r["request_id"] for r in propagation}),
    }

    # 4. forced termination + replay
    with tempfile.TemporaryDirectory() as tmp:
        ledger = Ledger(os.path.join(tmp, "ledger.sqlite"))
        first = batches[0]
        fctx = build_batch_context(spec, first, requests_by_id)
        adopted, raws = [], []
        for request_id in first.request_ids:
            raw = encode_receipt(3, fctx, request_id,
                                 fctx.output_summary_by_request[request_id], 12)
            raws.append(raw)
            norm = normalize_raw(raw, fctx, known)
            adopted.append({
                "batch_id": first.batch_id, "request_id": request_id,
                "source_format": 3, "content_summary": norm.content_summary(),
                "normalized": norm.to_dict()})
        ledger.commit_batch(batch=first, hits=hits, adopted=adopted,
                            rejected=[], invalidations=[],
                            wait_start_ms=0, wait_end_ms=1)
        committed, unknown = classify_batches(ledger, batches)
        replay = replay_unfinished(
            spec, ledger, batches,
            {first.batch_id: raws})
        ledger.close()
        report["4_force_termination_replay"] = {
            "committed_batches": [b.batch_id for b in committed],
            "unknown_requires_rerun": [b.batch_id for b in unknown],
            "exit_report_generated": False,
            "replay_verified": replay["verified"],
        }

    out_path = os.path.join(ROOT, "data", "acceptance_report.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nreport written to", out_path)


if __name__ == "__main__":
    main()
