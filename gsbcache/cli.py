"""Command line interface for validation, execution and verification."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from .content import artifact_digest
from .converge import check_convergence
from .errors import AppError, LedgerError
from .ledger import Ledger, replace_ledger_from_snapshot
from .runner import PipelineRun
from .sample import create_sample
from .spec import load_spec
from .util import write_json_file
from .verify import verify_ledger


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def _validate_command(args) -> dict[str, Any]:
    spec = load_spec(args.spec)
    matrix = []
    for node in spec.nodes:
        for artifact in spec.ordered_artifacts:
            actual = artifact_digest(node, artifact.key)
            refs_ok = all(
                artifact_digest(node, ref) == spec.artifacts[ref].expected_digest
                for ref in artifact.refs
            )
            reusable = actual == artifact.expected_digest and refs_ok
            matrix.append(
                {
                    "node_id": node.node_id,
                    "artifact_key": artifact.key,
                    "expected_digest": artifact.expected_digest,
                    "actual_digest": actual,
                    "references_reusable": refs_ok,
                    "reusable": reusable,
                    "reason": (
                        "digest_and_references_match"
                        if reusable
                        else ("digest_mismatch" if actual != artifact.expected_digest else "reference_digest_mismatch")
                    ),
                }
            )
    return {
        "ok": True,
        "code": "OK",
        "spec": str(Path(args.spec).resolve()),
        "spec_digest": spec.digest,
        "format_version": spec.format_version,
        "pipeline_id": spec.pipeline.pipeline_id,
        "nodes": [node.__dict__ for node in spec.nodes],
        "stages": [
            {
                "stage_id": stage.stage_id,
                "concurrency": stage.concurrency,
                "artifact_keys": list(stage.artifact_keys),
            }
            for stage in spec.pipeline.stages
        ],
        "topological_stage_order": [stage.stage_id for stage in spec.topological_stages()],
        "reusability": matrix,
        "first_invalid": None,
    }


def _run_command(args) -> dict[str, Any]:
    run = PipelineRun(
        args.spec,
        args.work,
        lease_ms=args.lease_ms,
        queue_timeout_ms=args.queue_timeout_ms,
        wait_for_expiry=not args.no_wait_expiry,
    )
    result = run.execute()
    if args.result_out:
        write_json_file(args.result_out, result)
    return result


def _ledger_info_command(args) -> dict[str, Any]:
    spec = load_spec(args.spec) if args.spec else None
    with Ledger(args.ledger, spec, read_only=True) as ledger:
        info = ledger.info()
        info["ok"] = True
        info["sessions"] = ledger.get_sessions()
        info["attempts"] = ledger.get_attempts()
        return info


def _ledger_replace_command(args) -> dict[str, Any]:
    result = replace_ledger_from_snapshot(args.ledger, args.snapshot)
    if args.spec:
        with Ledger(args.ledger, load_spec(args.spec), read_only=True) as ledger:
            result["post_replace_info"] = ledger.info()
    return result


def _digest_command(args) -> dict[str, Any]:
    spec = load_spec(args.spec)
    rows = []
    for node in spec.nodes:
        for artifact in spec.ordered_artifacts:
            actual = artifact_digest(node, artifact.key)
            rows.append(
                {
                    "node_id": node.node_id,
                    "artifact_key": artifact.key,
                    "expected_digest": artifact.expected_digest,
                    "actual_digest": actual,
                    "digest_match": actual == artifact.expected_digest,
                    "reusable": actual == artifact.expected_digest
                    and all(
                        artifact_digest(node, ref) == spec.artifacts[ref].expected_digest
                        for ref in artifact.refs
                    ),
                }
            )
    return {"ok": True, "spec_digest": spec.digest, "digests": rows}


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m gsbcache",
        description="Deterministic loopback artifact-cache lease arbitration ledger",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sample = sub.add_parser("sample", help="generate the local deterministic sample")
    sample.add_argument("--out", required=True)
    sample.add_argument("--base-port", type=int, default=None)
    sample.set_defaults(func=lambda args: create_sample(args.out, args.base_port))

    validate = sub.add_parser("validate", help="strongly validate declarations and show reusability")
    validate.add_argument("--spec", required=True)
    validate.set_defaults(func=_validate_command)

    run = sub.add_parser("run", help="start nodes, run the pipeline and verify the result")
    run.add_argument("--spec", required=True)
    run.add_argument("--work", required=True)
    run.add_argument("--lease-ms", type=int, default=700)
    run.add_argument("--queue-timeout-ms", type=int, default=15000)
    run.add_argument("--no-wait-expiry", action="store_true")
    run.add_argument("--result-out", default=None)
    run.set_defaults(func=_run_command)

    verify = sub.add_parser("verify", help="independently replay the ledger")
    verify.add_argument("--spec", required=True)
    verify.add_argument("--ledger", required=True)
    verify.set_defaults(func=lambda args: verify_ledger(args.spec, args.ledger))

    converge = sub.add_parser("converge", help="compare persisted node replicas with ledger digests")
    converge.add_argument("--spec", required=True)
    converge.add_argument("--ledger", required=True)
    converge.add_argument("--nodes-root", required=True)
    converge.set_defaults(
        func=lambda args: check_convergence(args.spec, args.ledger, args.nodes_root)
    )

    info = sub.add_parser("ledger-info", help="show validated ledger summary")
    info.add_argument("--ledger", required=True)
    info.add_argument("--spec", default=None)
    info.set_defaults(func=_ledger_info_command)

    export = sub.add_parser("ledger-export", help="export a whole-state snapshot")
    export.add_argument("--ledger", required=True)
    export.add_argument("--spec", required=True)
    export.add_argument("--out", required=True)
    export.set_defaults(
        func=lambda args: _export_command(args.ledger, args.spec, args.out)
    )

    replace = sub.add_parser("ledger-replace", help="whole-snapshot replacement with stable differences")
    replace.add_argument("--ledger", required=True)
    replace.add_argument("--snapshot", required=True)
    replace.add_argument("--spec", default=None)
    replace.set_defaults(func=_ledger_replace_command)

    digest = sub.add_parser("digest", help="show per-node deterministic digests")
    digest.add_argument("--spec", required=True)
    digest.set_defaults(func=_digest_command)
    return parser


def _export_command(ledger_path: str, spec_path: str, out: str) -> dict[str, Any]:
    spec = load_spec(spec_path)
    with Ledger(ledger_path, spec, read_only=True) as ledger:
        return ledger.export_snapshot(out)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.func(args)
        _print(result)
        return 0 if result.get("ok", True) else 1
    except AppError as exc:
        _print(exc.to_dict())
        return exc.exit_code
    except BrokenPipeError:
        return 1
    except Exception as exc:  # Stable CLI boundary for unexpected failures.
        _print(
            {
                "ok": False,
                "code": "E_INTERNAL",
                "message": "unexpected internal failure",
                "error": repr(exc),
            }
        )
        return 11
