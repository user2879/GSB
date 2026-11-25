"""Generate mixed r1/r2/r3 receipts for the same batch (cross-version merge).

Produces three historical receipts per request of one chosen batch, each valid
under its own format, plus deliberately invalid variants used by the E_FIELD
tests.  Output: data/receipts.ndjson and data/receipts_invalid.ndjson.
"""
from __future__ import annotations

import json
import os

from grayroute.batching import build_batch_context, plan_batches
from grayroute.receipt import encode_receipt
from grayroute.sharding import attribute_all
from grayroute.spec import load_spec

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")


def main() -> None:
    spec = load_spec(os.path.join(ROOT, "config"),
                     os.path.join(DATA, "requests.ndjson"))
    hits = attribute_all(spec)
    batches = plan_batches(spec, hits)
    target = next(b for b in batches if len(b.request_ids) >= 2)
    requests_by_id = {q.request_id: q for q in spec.requests}
    context = build_batch_context(spec, target, requests_by_id)

    lines: list[dict] = []
    for request_id in target.request_ids:
        summary = context.output_summary_by_request[request_id]
        for fv in (1, 2, 3):
            lines.append(encode_receipt(fv, context, request_id, summary,
                                        elapsed_ms=12, superseded_ref=None))
    out_path = os.path.join(DATA, "receipts.ndjson")
    with open(out_path, "w", encoding="utf-8") as handle:
        for row in lines:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    invalid = []
    base = encode_receipt(3, context, target.request_ids[0],
                          context.output_summary_by_request[target.request_ids[0]],
                          12)
    bad_elapsed = dict(base); bad_elapsed["elapsed_ms"] = -5
    invalid.append(bad_elapsed)
    bad_rank = dict(encode_receipt(3, context, target.request_ids[0],
                                   context.output_summary_by_request[
                                       target.request_ids[0]], 12))
    bad_rank["rank_in_batch"] = 999
    invalid.append(bad_rank)
    bad_ref = dict(encode_receipt(3, context, target.request_ids[0],
                                  context.output_summary_by_request[
                                      target.request_ids[0]], 12))
    bad_ref["superseded_ref"] = "does-not-exist"
    invalid.append(bad_ref)
    bad_snap = dict(encode_receipt(3, context, target.request_ids[0],
                                   context.output_summary_by_request[
                                       target.request_ids[0]], 12))
    bad_snap["param_summary_snapshot"] = "0"
    invalid.append(bad_snap)
    with open(os.path.join(DATA, "receipts_invalid.ndjson"), "w",
              encoding="utf-8") as handle:
        for row in invalid:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta = {"batch_id": target.batch_id,
            "request_ids": list(target.request_ids)}
    with open(os.path.join(DATA, "receipts_meta.json"), "w",
              encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    print("receipt fixtures written for", target.batch_id)


if __name__ == "__main__":
    main()
