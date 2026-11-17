"""qdemo CLI entry point.

Subcommands:
  gen-weights   generate a deterministic local weight pack
  gen-calib     generate a deterministic calibration set
  validate      validate a config against a weight pack
  full          one-shot full run (quant + eval + distill) with report/record
  interactive   interactive command session (stdin or --script)
  replay        recompute from an experiment record and compare field-by-field
"""
from __future__ import annotations

import argparse
import json
import sys

from .common import (
    EXIT_CONFIG_INVALID,
    EXIT_OK,
    EXIT_REPLAY_MISMATCH,
    EXIT_WEIGHT_MISMATCH,
    BusinessError,
    E_CONFIG_INVALID,
    E_REPLAY_MISMATCH,
    E_WEIGHT_SHAPE_MISMATCH,
    OK,
    atomic_write_json,
    read_json,
)
from .config import load_config, validate_config
from .engine import Engine
from .report import build_record, build_report
from .weights_io import (
    generate_calibration,
    generate_weights,
    load_calibration,
    load_weights,
)

DEFAULT_SPEC = "fc1:16:32:normal,fc2:8:16:uniform,head:4:8:normal"


def _load_pack_and_cfg(config_path: str):
    cfg = load_config(config_path)
    weights = load_weights(cfg["weights_path"])
    validate_config(cfg, weights)
    calib = load_calibration(cfg["calib_path"])
    return cfg, weights, calib


def cmd_gen_weights(args) -> int:
    meta = generate_weights(args.out, args.seed, args.layers)
    print(json.dumps({"business_code": OK, "layers": len(meta["layers"]), "out": args.out}))
    return EXIT_OK


def cmd_gen_calib(args) -> int:
    meta = generate_calibration(args.weights, args.out, args.seed, args.samples)
    print(json.dumps({"business_code": OK, "samples": meta["samples"], "out": args.out}))
    return EXIT_OK


def cmd_validate(args) -> int:
    _load_pack_and_cfg(args.config)
    print(json.dumps({"business_code": OK}))
    return EXIT_OK


def cmd_full(args) -> int:
    cfg, weights, calib = _load_pack_and_cfg(args.config)
    engine = Engine(cfg, weights, calib)
    results = engine.compute()
    atomic_write_json(args.record, build_record(cfg, weights, calib, results))
    report = build_report(cfg, weights, calib, results)
    atomic_write_json(args.report, report)
    print(json.dumps({
        "business_code": report["business_code"],
        "summary": report["summary"],
        "failures": report["failures"],
        "report": args.report,
        "record": args.record,
    }, ensure_ascii=False))
    return EXIT_OK


def cmd_interactive(args) -> int:
    from .interactive import Session

    cfg, weights, calib = _load_pack_and_cfg(args.config)
    session = Session(cfg, weights, calib, args.workdir)
    if args.script:
        with open(args.script, "r", encoding="utf-8") as fh:
            commands = fh.read().splitlines()
        rc = session.run_script(commands)
    else:
        rc = session.run_stdin()
    print(json.dumps({"business_code": OK, "exit_code": rc, "workdir": args.workdir}))
    return rc


def _strip_for_compare(record: dict) -> dict:
    return {
        "config_fingerprint": record["config_fingerprint"],
        "weights_fingerprint": record["weights_fingerprint"],
        "calib_fingerprint": record["calib_fingerprint"],
        "rounding": record["rounding"],
        "layers": record["layers"],
        "distill": record["distill"],
        "failures": record["failures"],
    }


def cmd_replay(args) -> int:
    record = read_json(args.record)
    cfg = record["config"]
    weights = load_weights(cfg["weights_path"])
    calib = load_calibration(cfg["calib_path"])
    validate_config(cfg, weights)
    engine = Engine(cfg, weights, calib)
    results = engine.compute()
    fresh = _strip_for_compare(build_record(cfg, weights, calib, results))
    old = _strip_for_compare(record)
    if fresh == old:
        print(json.dumps({"business_code": OK, "replay": "identical"}))
        return EXIT_OK
    diff_keys = [k for k in fresh if fresh[k] != old.get(k)]
    print(json.dumps({"business_code": E_REPLAY_MISMATCH, "mismatched_sections": diff_keys}))
    return EXIT_REPLAY_MISMATCH


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qdemo")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen-weights")
    g.add_argument("--out", required=True)
    g.add_argument("--seed", type=int, required=True)
    g.add_argument("--layers", default=DEFAULT_SPEC)
    g.set_defaults(func=cmd_gen_weights)

    c = sub.add_parser("gen-calib")
    c.add_argument("--weights", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--seed", type=int, required=True)
    c.add_argument("--samples", type=int, default=32)
    c.set_defaults(func=cmd_gen_calib)

    v = sub.add_parser("validate")
    v.add_argument("--config", required=True)
    v.set_defaults(func=cmd_validate)

    f = sub.add_parser("full")
    f.add_argument("--config", required=True)
    f.add_argument("--report", required=True)
    f.add_argument("--record", required=True)
    f.set_defaults(func=cmd_full)

    i = sub.add_parser("interactive")
    i.add_argument("--config", required=True)
    i.add_argument("--workdir", required=True)
    i.add_argument("--script", default=None)
    i.set_defaults(func=cmd_interactive)

    r = sub.add_parser("replay")
    r.add_argument("--record", required=True)
    r.set_defaults(func=cmd_replay)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except BusinessError as exc:
        code = EXIT_CONFIG_INVALID
        if exc.code == E_WEIGHT_SHAPE_MISMATCH:
            code = EXIT_WEIGHT_MISMATCH
        print(json.dumps({"business_code": exc.code, "error": exc.message}))
        return code


if __name__ == "__main__":
    sys.exit(main())
