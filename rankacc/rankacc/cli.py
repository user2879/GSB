"""Command line interface: gen-samples / run / apply-edits / convert / verify / report."""
from __future__ import annotations

import argparse
import json
import os
import sys

from .core import BizError, CrashFault, parse_yaml
from .formats import parse_v1_line, v1_lines_to_v2
from .runner import Runner
from .samplegen import write_samples
from .store import Store, canon


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def _load_data(data_dir):
    requests = _read_jsonl(os.path.join(data_dir, "candidates.jsonl"))
    with open(os.path.join(data_dir, "config.yaml"), "r", encoding="utf-8") as fh:
        config = parse_yaml(fh.read())
    return requests, config


def cmd_gen_samples(args):
    write_samples(args.out)
    print("samples written to %s" % args.out)


def cmd_run(args):
    requests, config = _load_data(args.data)
    runner = Runner(args.store)
    runner.set_inputs(requests, config)
    faults = _read_jsonl(args.faults) if args.faults else None
    try:
        summary = runner.full_run(faults=faults, note="cli:run")
    except CrashFault as exc:
        print("FORCED TERMINATION (injected): %s; store left at last commit" % exc)
        return 3
    print(json.dumps(summary, sort_keys=True))
    return 0


def cmd_apply_edits(args):
    runner = Runner(args.store)
    for op in _read_jsonl(args.edits):
        try:
            summary = runner.apply_edit(op)
        except BizError as exc:
            print("edit rejected: %s" % exc)
            continue
        print("%s -> %s" % (op.get("op"), json.dumps(summary, sort_keys=True)))
    return 0


def cmd_convert(args):
    """Round-trip the stored outputs between v1 and v2 via the sidecar."""
    store = Store(args.store)
    outputs = store.outputs
    if outputs is None:
        print("no outputs in store")
        return 1
    ledger = []
    if outputs["version"] == "v2":
        from .formats import export_v1
        lines, sidecar = [], {}
        for rid in sorted(outputs["conclusions"]):
            concl = outputs["conclusions"][rid]
            new_lines, sc = export_v1(concl)
            lines.extend(new_lines)
            sidecar[rid] = sc
            ledger.append({"type": "downgrade", "direction": "v2->v1",
                           "request_id": rid})
        restored = v1_lines_to_v2(lines, sidecar, ledger)
    else:
        lines = outputs["artifacts"]["lines"]
        for line in lines:
            parse_v1_line(line)
        restored = v1_lines_to_v2(lines, outputs["sidecar"], ledger)
    ok = True
    orig = {r["request_id"]: r for r in
            (outputs["artifacts"].get("records") or restored)}
    for rec in restored:
        ref = orig.get(rec["request_id"])
        if ref is None:
            continue
        for got, want in zip(rec["candidates"], ref["candidates"]):
            if got["boundary"] != want["boundary"] \
                    or got["boundary_diff_ref"] != want["boundary_diff_ref"] \
                    or got["score"] != want["score"]:
                ok = False
    store.new_ledger = ledger
    store.commit()
    print(json.dumps({"roundtrip_restore_ok": ok, "requests": len(restored),
                      "downgrade_entries": len(ledger)}, sort_keys=True))
    return 0 if ok else 2


def cmd_verify(args):
    """Incremental edits vs from-scratch full rebuild, field by field."""
    requests, config = _load_data(args.data)
    edits = _read_jsonl(os.path.join(args.data, "edits.jsonl"))
    store_a = os.path.join(args.work, "store_inc")
    store_b = os.path.join(args.work, "store_full")
    runner_a = Runner(store_a)
    runner_a.set_inputs(requests, config)
    runner_a.full_run(note="verify:base")
    for op in edits:
        runner_a.apply_edit(op)
    final_state = runner_a.store.state
    runner_b = Runner(store_b)
    runner_b.set_inputs(final_state["requests"], final_state["config"])
    runner_b.full_run(note="verify:full")
    out_a, out_b = runner_a.store.outputs, runner_b.store.outputs
    same_outputs = canon(out_a) == canon(out_b)
    keys_a = runner_a.store.report["cache_keys"]
    keys_b = runner_b.store.report["cache_keys"]
    same_keys = canon(keys_a) == canon(keys_b)
    reused = 0
    for rid in keys_a:
        v = runner_a.store.report["reuse_map"][rid]
        stages = list(v["norm"].values()) + [v["trunc"]] + \
            list(v["layer"].values()) + [v["ser"]]
        reused += sum(1 for s in stages if s == "reuse")
    result = {"outputs_identical": same_outputs, "cache_keys_identical": same_keys,
              "reused_stages_in_incremental": reused}
    print(json.dumps(result, sort_keys=True))
    return 0 if (same_outputs and same_keys) else 2


def cmd_report(args):
    store = Store(args.store)
    if store.report is None:
        print("no report in store")
        return 1
    print(json.dumps(store.report, ensure_ascii=False, indent=1, sort_keys=True))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rankacc")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("gen-samples")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_gen_samples)
    p = sub.add_parser("run")
    p.add_argument("--data", required=True)
    p.add_argument("--store", required=True)
    p.add_argument("--faults", default=None)
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("apply-edits")
    p.add_argument("--store", required=True)
    p.add_argument("--edits", required=True)
    p.set_defaults(fn=cmd_apply_edits)
    p = sub.add_parser("convert")
    p.add_argument("--store", required=True)
    p.set_defaults(fn=cmd_convert)
    p = sub.add_parser("verify")
    p.add_argument("--data", required=True)
    p.add_argument("--work", required=True)
    p.set_defaults(fn=cmd_verify)
    p = sub.add_parser("report")
    p.add_argument("--store", required=True)
    p.set_defaults(fn=cmd_report)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
