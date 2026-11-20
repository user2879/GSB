"""Command line interface for graph editing, conversion and runs."""
import argparse
import json
import os
import sys

from .config import load_config
from .converter import (downgrade_v2_to_v1, upgrade_v1_to_v2,
                        write_sidecar, read_sidecar)
from .editor import (edit_bitwidth, edit_scale, edit_rounding, add_node,
                     remove_node)
from .engine import Engine
from .injection import load_injections
from .v1_codec import parse_v1, dump_v1
from .v2_codec import parse_v2, dump_v2
from .validator import validate


def _load_graph(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    if path.endswith(".v1.txt") or text.startswith("QCGV1"):
        return parse_v1(text)
    return parse_v2(text)


def _write_graph(graph, path: str) -> None:
    text = dump_v1(graph) if path.endswith(".v1.txt") else dump_v2(graph)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def cmd_convert(args) -> None:
    graph = _load_graph(args.input)
    sidecar_path = args.sidecar
    if args.target == "v2":
        sidecar = read_sidecar(sidecar_path) if sidecar_path and \
            os.path.exists(sidecar_path) else None
        converted, losses = upgrade_v1_to_v2(graph, sidecar)
        validate(converted)
        _write_graph(converted, args.output)
        print(json.dumps({"direction": "v1->v2",
                          "loss_records": [l.to_dict() for l in losses]},
                         indent=2))
        return
    converted, sidecar, losses = downgrade_v2_to_v1(graph)
    validate(converted)
    _write_graph(converted, args.output)
    if sidecar_path:
        write_sidecar(sidecar_path, sidecar)
    print(json.dumps({"direction": "v2->v1",
                      "sidecar": sidecar_path,
                      "loss_records": [l.to_dict() for l in losses]},
                     indent=2))


def cmd_edit(args) -> None:
    graph = _load_graph(args.graph)
    if args.op == "bitwidth":
        edit_bitwidth(graph, args.node, int(args.value))
    elif args.op == "scale":
        edit_scale(graph, args.node, scale_permille=int(args.value))
    elif args.op == "rounding":
        edit_rounding(graph, args.node, args.value)
    elif args.op == "add-conv":
        add_node(graph, "conv", int(args.bitwidth), int(args.scale),
                 args.inputs.split(","), kernel=int(args.kernel),
                 stride=int(args.stride), padding=int(args.padding))
    elif args.op == "remove":
        remove_node(graph, args.node)
    validate(graph)
    _write_graph(graph, args.graph)
    print("edited %s on %s" % (args.op, args.node))


def cmd_run(args) -> None:
    cfg = load_config(args.config)
    graph = _load_graph(args.graph)
    injections = load_injections(args.injection)
    engine = Engine(cfg, args.snapshot_root, injections,
                    global_version=args.global_version)
    report = engine.run(graph, args.pack, resume=args.resume)
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, default=str)
    print(json.dumps({"report": args.report,
                      "recomputed": report["recomputed"],
                      "reused": report["reused"],
                      "pending": report["pending"],
                      "faults": report["faults"]}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qcg")
    sub = parser.add_subparsers(dest="command", required=True)

    conv = sub.add_parser("convert")
    conv.add_argument("--input", required=True)
    conv.add_argument("--output", required=True)
    conv.add_argument("--target", choices=["v1", "v2"], required=True)
    conv.add_argument("--sidecar")
    conv.set_defaults(func=cmd_convert)

    edit = sub.add_parser("edit")
    edit.add_argument("op", choices=["bitwidth", "scale", "rounding",
                                     "add-conv", "remove"])
    edit.add_argument("--graph", required=True)
    edit.add_argument("--node")
    edit.add_argument("--value")
    edit.add_argument("--bitwidth", default=8)
    edit.add_argument("--scale", default=1000)
    edit.add_argument("--kernel", type=int, default=3)
    edit.add_argument("--stride", type=int, default=1)
    edit.add_argument("--padding", type=int, default=1)
    edit.add_argument("--inputs", default="")
    edit.set_defaults(func=cmd_edit)

    run = sub.add_parser("run")
    run.add_argument("--graph", required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--pack", required=True)
    run.add_argument("--snapshot-root", required=True)
    run.add_argument("--report", required=True)
    run.add_argument("--injection")
    run.add_argument("--global-version", type=int, default=1)
    run.add_argument("--resume", action="store_true")
    run.set_defaults(func=cmd_run)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
