from __future__ import annotations

import argparse
import sys
from typing import Any

from .canonical import canonical_text
from .errors import DomainError
from .gui_model import build_diagnostics_model, model_digest
from .runner import execute, reset_ledger_files
from .session import build_full_session, compare_incremental_with_full, recompute_session
from .spec import load_spec
from .termination import termination_support


def _print(value: Any) -> None:
    sys.stdout.write(canonical_text(value) + "\n")
    sys.stdout.flush()


def _spec_from_args(args: argparse.Namespace):
    return load_spec(args.certs, args.handshake, getattr(args, "anomaly", "none"))


def command_validate(args: argparse.Namespace) -> int:
    spec = _spec_from_args(args)
    session = build_full_session(spec)
    result = {
        "ok": True,
        "certificate_count": len(spec.certificates),
        "message_count": len(spec.messages),
        "trust_anchors": list(spec.trust_anchors),
        "fixed": spec.fixed.as_dict(),
        "decision_counts": session.decision_counts(),
        "affected_by_error_count": session.affected_count(),
        "expected_mismatch_seqs": session.expected_mismatch_seqs(),
        "message_decisions": {
            str(seq): {
                "decision": record["decision"],
                "affected_by_error": record["affected_by_error"],
                "selection_reason": record["selection_reason"],
                "selected_cert_ids": record["selected_cert_ids"],
                "error_bound": record["selected_chain"]["error_bound"],
            }
            for seq, record in sorted(session.message_records.items())
        },
    }
    _print(result)
    return 0 if not session.expected_mismatch_seqs() else 1


def command_run(args: argparse.Namespace, force_resume: bool = False) -> int:
    if args.reset:
        reset_ledger_files(args.ledger)
    spec = _spec_from_args(args)
    result = execute(
        cert_path=args.certs,
        handshake_path=args.handshake,
        spec=spec,
        ledger_path=args.ledger,
        port=args.port,
        resume=force_resume or args.resume,
        terminate_after=args.terminate_after,
        termination=args.termination,
        drop_client_seq=args.drop_client_seq,
        omit_reply_field=args.omit_reply_field,
    )
    _print({"ok": True, **result})
    return 0


def command_incremental(args: argparse.Namespace) -> int:
    base_spec = load_spec(args.base_certs, args.base_handshake)
    changed_spec = load_spec(args.changed_certs, args.changed_handshake or args.base_handshake)
    base_session = build_full_session(base_spec)
    incremental = recompute_session(base_session, changed_spec)
    full = build_full_session(changed_spec)
    comparison = compare_incremental_with_full(incremental, full)
    retained_certs = sorted(set(base_session.cert_bases) - set(incremental.invalidated_cert_ids))
    retained_messages = sorted(set(base_session.message_records) - set(incremental.invalidated_message_seqs))
    retained_evidence = {
        "certificate_bytes_unchanged": all(
            canonical_text(base_session.cert_bases[cert_id]) == canonical_text(incremental.cert_bases[cert_id])
            for cert_id in retained_certs
        ),
        "message_bytes_unchanged": all(
            canonical_text(base_session.message_records[seq])
            == canonical_text(incremental.message_records[seq])
            for seq in retained_messages
        ),
        "retained_certificate_ids": retained_certs,
        "retained_message_seqs": retained_messages,
    }
    _print(
        {
            "ok": comparison["equal"] and all(
                value for key, value in retained_evidence.items() if key.endswith("_unchanged")
            ),
            "comparison": comparison,
            "retained_evidence": retained_evidence,
            "invalidated_certificate_ids": incremental.invalidated_cert_ids,
            "invalidated_message_seqs": incremental.invalidated_message_seqs,
            "certificate_invalidation": incremental.cert_invalidation,
            "message_invalidation": incremental.message_invalidation,
            "decision_counts": incremental.decision_counts(),
        }
    )
    return 0


def command_diagnostics(args: argparse.Namespace) -> int:
    spec = load_spec(args.certs, args.handshake)
    session = build_full_session(spec)
    model = build_diagnostics_model(spec, session, args.ledger)
    _print(
        {
            "ok": True,
            "model_digest": model_digest(model),
            "model": model,
        }
    )
    return 0


def command_gui(args: argparse.Namespace) -> int:
    from .gui import run_gui

    return run_gui(args.certs, args.handshake, args.ledger)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tlschain", description="Fixed-point TLS chain and handshake diagnostics")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="validate and verify a local JSON specification")
    validate_parser.add_argument("--certs", required=True)
    validate_parser.add_argument("--handshake", required=True)
    validate_parser.add_argument("--anomaly", choices=["none", "corrupted-json", "self-cycle"], default="none")
    validate_parser.set_defaults(func=command_validate)

    run_parser = subparsers.add_parser("run", help="run the loopback handshake and commit the ledger")
    run_parser.add_argument("--certs", required=True)
    run_parser.add_argument("--handshake", required=True)
    run_parser.add_argument("--ledger", required=True)
    run_parser.add_argument("--port", type=int, default=23771)
    run_parser.add_argument("--reset", action="store_true")
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--terminate-after", type=int)
    run_parser.add_argument("--termination", choices=["graceful", "hard"], default="graceful")
    run_parser.add_argument("--drop-client-seq", type=int)
    run_parser.add_argument("--omit-reply-field")
    run_parser.add_argument("--anomaly", choices=["none", "corrupted-json", "self-cycle"], default="none")
    run_parser.set_defaults(func=command_run)

    recover_parser = subparsers.add_parser("recover", help="replay committed records then rerun unknown batches")
    recover_parser.add_argument("--certs", required=True)
    recover_parser.add_argument("--handshake", required=True)
    recover_parser.add_argument("--ledger", required=True)
    recover_parser.add_argument("--port", type=int, default=23771)
    recover_parser.add_argument("--anomaly", choices=["none", "corrupted-json", "self-cycle"], default="none")
    recover_parser.add_argument("--drop-client-seq", type=int)
    recover_parser.add_argument("--omit-reply-field")
    recover_parser.set_defaults(func=lambda args: command_run(args, force_resume=True), reset=False, resume=True, terminate_after=None, termination="graceful")

    incremental_parser = subparsers.add_parser("incremental", help="compare incremental and full rebuilds")
    incremental_parser.add_argument("--base-certs", required=True)
    incremental_parser.add_argument("--base-handshake", required=True)
    incremental_parser.add_argument("--changed-certs", required=True)
    incremental_parser.add_argument("--changed-handshake")
    incremental_parser.set_defaults(func=command_incremental)

    diagnostics_parser = subparsers.add_parser("diagnostics", help="emit a deterministic headless GUI model")
    diagnostics_parser.add_argument("--certs", required=True)
    diagnostics_parser.add_argument("--handshake", required=True)
    diagnostics_parser.add_argument("--ledger")
    diagnostics_parser.set_defaults(func=command_diagnostics)

    gui_parser = subparsers.add_parser("gui", help="open the Tkinter diagnostics UI")
    gui_parser.add_argument("--certs", required=True)
    gui_parser.add_argument("--handshake", required=True)
    gui_parser.add_argument("--ledger")
    gui_parser.set_defaults(func=command_gui)

    info_parser = subparsers.add_parser("termination-info", help="print termination support details")
    info_parser.set_defaults(func=lambda args: _print({"ok": True, **termination_support()}) or 0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except DomainError as exc:
        _print({"ok": False, **exc.as_dict()})
        return 2
    except KeyboardInterrupt:
        _print({"ok": False, "code": "E_INTERRUPTED", "message": "interrupted"})
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
