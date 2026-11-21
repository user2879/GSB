"""Repeatable end-to-end demo: routing -> hot swap -> rebuild -> verify."""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hotgray.gateway import Gateway
from hotgray.samples import build_sample_declaration
from hotgray.snapshot import authoritative_recompute, compare_snapshots, snapshot_digest
from hotgray.spec import validate_declaration

DECL_PATH = os.path.join(ROOT, "samples", "gateway_declaration.json")
DB_PATH = os.path.join(ROOT, "runtime", "demo_e2e.sqlite3")


def main():
    os.makedirs(os.path.join(ROOT, "runtime"), exist_ok=True)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    declaration = build_sample_declaration()
    validate_declaration(declaration)
    with open(DECL_PATH, "w", encoding="utf-8") as handle:
        json.dump(declaration, handle, ensure_ascii=False, indent=2)

    gateway = Gateway(DECL_PATH, DB_PATH)
    gateway.start()
    stream = declaration["request_stream"]
    split_point = 10
    for request in stream[:split_point]:
        gateway.admit_request(request)
    gateway.flush_all()

    swaps = gateway.hotswap_gray_backends()
    for request in stream[split_point:]:
        gateway.admit_request(request)
    gateway.flush_all()

    event = gateway.rebuild_snapshot()
    snapshot = event["snapshot"]
    recomputed = authoritative_recompute(declaration, gateway.store)
    compare_snapshots(snapshot["body"], recomputed["groups"])
    digest = snapshot_digest(
        recomputed["groups"], snapshot["param_versions"],
        snapshot["ledger_seq_from"], snapshot["ledger_seq_to"],
    )
    assert digest == snapshot["digest"]

    split_count = sum(
        len(group.get("splits", []))
        for rule in snapshot["body"].values()
        for key, group in rule.items() if isinstance(group, dict)
    )
    result = {
        "committed_hits": gateway.store.count_hits(),
        "hotswaps": swaps,
        "snapshot_id": snapshot["snapshot_id"],
        "ledger_range": [snapshot["ledger_seq_from"],
                         snapshot["ledger_seq_to"]],
        "incoming_during_rebuild": snapshot["incoming_during_rebuild"],
        "recorded_splits": split_count,
        "authoritative_verdict": "byte_identical",
        "rebuilds": gateway.store.rebuilds(),
    }
    gateway.stop()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
