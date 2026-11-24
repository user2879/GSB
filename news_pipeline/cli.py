"""Command line entry points for the deterministic news pipeline."""

import argparse
import json
import os
import sys

from .config import load_config, load_rewrite_table
from .documents import load_documents
from .edits import apply_all, load_edit_stream
from .pipeline import GracefulStop, run_pipeline


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)


def _load_inputs(sample_dir):
    config = load_config(os.path.join(sample_dir, "config.yaml"))
    failures = []
    table = load_rewrite_table(os.path.join(sample_dir, "rewrite_table.yaml"), config, failures)
    documents, doc_failures = load_documents(os.path.join(sample_dir, "documents.jsonl"))
    failures.extend(doc_failures)
    return config, table, documents, failures


def run(sample_dir, snapshot_dir, report_path, edit_stream=None, stop_after_docs=None):
    config, table, documents, failures = _load_inputs(sample_dir)
    if edit_stream:
        commands, edit_failures = load_edit_stream(edit_stream)
        failures.extend(edit_failures)
        documents, config, table = apply_all(documents, config, table, commands, failures)

    code_counter = {}
    stop_controller = GracefulStop()
    stop_controller.install()
    def after_doc(completed, _doc_id):
        if stop_after_docs is not None and completed >= stop_after_docs:
            stop_controller.requested = True
    report, _results, store, _cache = run_pipeline(
        documents, config, table, snapshot_dir, stop_controller, code_counter,
        after_doc_callback=after_doc,
    )
    report["failures"] = failures
    report["interrupted"] = stop_controller.requested
    _write_json(report_path, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Deterministic Chinese news summarization pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate-samples", help="create deterministic local samples")
    gen.add_argument("--out", required=True)

    run_parser = sub.add_parser("run", help="run the pipeline")
    run_parser.add_argument("--samples", required=True)
    run_parser.add_argument("--snapshot", required=True)
    run_parser.add_argument("--report", required=True)
    run_parser.add_argument("--edits")
    run_parser.add_argument("--stop-after-docs", type=int)

    args = parser.parse_args(argv)
    if args.command == "generate-samples":
        from .sample_generator import generate_all
        generate_all(args.out)
        print("samples generated in %s" % args.out)
        return 0
    report = run(
        args.samples, args.snapshot, args.report,
        edit_stream=args.edits, stop_after_docs=args.stop_after_docs,
    )
    print(json.dumps(report["totals"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
