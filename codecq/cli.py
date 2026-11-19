"""Command line entry: synth / run / view."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from . import report as report_mod
from . import runner, synth, util


def load_config(path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def cmd_synth(args) -> int:
    cfg = load_config(args.config)
    frames, pcm, truth = synth.generate_source(cfg, Path(args.out) / "source")
    from . import colors, pipeline

    planes = [colors.rgb_to_planes(rgb, "yuv420p") for rgb in frames]
    pipeline.mux_source_reference(planes, pcm, cfg, Path(args.out) / "source" / "source.mkv")
    print(f"source: {len(frames)} frames, {pcm.shape[1]} samples -> {args.out}/source")
    print(f"video content sha256: {truth['video']['content_sha256']}")
    print(f"audio sha256: {truth['audio']['sha256']}")
    return 0


def cmd_run(args) -> int:
    cfg = load_config(args.config)
    result = runner.run_pipeline(cfg, args.out)
    ledger = result["ledger"]
    summary = ledger["summary"]
    print(f"ledger version: {ledger['version']}")
    print(f"commit: {result['commit']}")
    print(
        "encodes: {encodes_ok}/{encodes_total} ok, findings: {findings_total}, "
        "execution_complete: {execution_complete}".format(**summary)
    )
    for enc in ledger["encodes"]:
        if enc["status"] != "ok":
            print(f"  [{enc['id']}] FAILED after {enc['retries']} retries: {enc['error']}")
            continue
        codes = [f["code"] for f in enc["findings"]]
        print(f"  [{enc['id']}] frames={enc['frame_count']} findings={codes or 'none'}")
    print(f"report: {result['view']['path']}")
    return 0


def cmd_view(args) -> int:
    info = report_mod.render_report(args.ledger, args.out)
    print(f"report written: {info['path']} (ledger {info['version']})")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="codecq", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("synth", help="generate deterministic source material only")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_synth)

    p = sub.add_parser("run", help="full pipeline: synth+encode+decode+metrics+inject+detect+ledger")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("view", help="render interactive HTML diagnostics from the committed ledger")
    p.add_argument("--ledger", required=True, help="ledger directory containing ledger.json + HEAD")
    p.add_argument("--out", required=True, help="output report.html path")
    p.set_defaults(fn=cmd_view)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
