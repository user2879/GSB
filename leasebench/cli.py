"""Command-line entry point for validation, simulation, replay, and UI."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

from .errors import DeclError, LeasebenchError, SnapshotError
from .ledger import Ledger
from .logging_utils import StructuredLogger
from .service import Workbench, load_view_from_ledger, replay_ledger
from .ui import launch_ui, render_view_text, write_text
from .validation import load_declaration, load_events, load_session, parse_declaration


EXIT_CODES = {
    "E_DECL": 2,
    "E_QUOTA": 3,
    "E_SNAPSHOT": 4,
    "E_REPLAY": 5,
}


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--log", default=None, help="structured JSON-lines log path")
    parser.add_argument("--json-out", default=None, help="write the result JSON to this path")


def _add_session(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--watermark-scale", type=int, default=1000)
    parser.add_argument("--borrow-return-error-ms", type=int, default=50)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="leasebench",
        description="Serverless concurrency lease and quota-return diagnostics",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate declaration and NDJSON events")
    validate.add_argument("--declaration", required=True)
    validate.add_argument("--events", required=True)
    validate.add_argument("--inject-cycle", action="store_true")
    _add_session(validate)
    _add_common(validate)

    run = sub.add_parser("run", help="run/recover deterministic event batches")
    run.add_argument("--declaration", required=True)
    run.add_argument("--events", required=True)
    run.add_argument("--db", required=True)
    run.add_argument("--limit-events", type=int, default=None)
    _add_session(run)
    run.add_argument("--inject-cycle", action="store_true")
    run.add_argument("--inject-quota", action="store_true")
    run.add_argument("--inject-snapshot", action="store_true")
    run.add_argument("--inject-crash-after-pending", action="store_true")
    run.add_argument("--text-out", default=None)
    _add_common(run)

    replay = sub.add_parser("replay", help="verify a ledger by deterministic replay")
    replay.add_argument("--db", required=True)
    replay.add_argument("--inject-snapshot", action="store_true")
    _add_common(replay)

    view = sub.add_parser("view", help="render the current ledger view")
    view.add_argument("--db", required=True)
    view.add_argument("--text-out", default=None)
    _add_common(view)

    ui = sub.add_parser("ui", help="open the Tk diagnostic workbench")
    ui.add_argument("--db", required=True)
    ui.add_argument("--headless", action="store_true")
    ui.add_argument("--text-out", default=None)
    _add_common(ui)

    return parser


def _load_declaration_with_injection(path: str, inject_cycle: bool) -> Any:
    if not inject_cycle:
        return load_declaration(path)
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    mutated = copy.deepcopy(raw)
    function_ids = sorted(
        item["function_id"] for item in mutated.get("functions", []) if "function_id" in item
    )
    if len(function_ids) < 2:
        raise DeclError(
            "functions",
            "cycle injection requires at least two functions",
        )
    higher, lower = function_ids[0], function_ids[1]
    mutated.setdefault("preemptions", []).extend(
        [
            {"higher": higher, "lower": lower, "return_deadline_ms": 100},
            {"higher": lower, "lower": higher, "return_deadline_ms": 100},
        ]
    )
    return parse_declaration(mutated)


def _write_json(path: str | None, value: Any) -> None:
    if path is None:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate(args: argparse.Namespace, logger: StructuredLogger) -> dict[str, Any]:
    declaration = _load_declaration_with_injection(
        args.declaration, getattr(args, "inject_cycle", False)
    )
    events = load_events(args.events, declaration)
    session = load_session(args.watermark_scale, args.borrow_return_error_ms)
    result = {
        "ok": True,
        "declaration_version": declaration.declaration_version,
        "declaration_digest": declaration.digest,
        "function_count": len(declaration.functions),
        "preemption_count": len(declaration.preemptions),
        "event_count": len(events),
        "session": session.as_dict(),
    }
    logger.write("validation.ok", **result)
    return result


def _run(args: argparse.Namespace, logger: StructuredLogger) -> dict[str, Any]:
    declaration = _load_declaration_with_injection(args.declaration, args.inject_cycle)
    events = load_events(args.events, declaration)
    session = load_session(args.watermark_scale, args.borrow_return_error_ms)
    if args.inject_cycle:
        logger.write("injection.cycle", level="ERROR")
    if args.inject_quota:
        logger.write("injection.quota", level="ERROR")

    pending_hook = None
    if args.inject_crash_after_pending:
        def pending_hook(batch_id: str) -> None:
            logger.write(
                "injection.forced_termination",
                level="ERROR",
                batch_id=batch_id,
                note="process exits before batch commit; no exit report is claimed",
            )
            os._exit(17)

    with Workbench(
        db_path=args.db,
        declaration=declaration,
        events=events,
        session=session,
        logger=logger,
    ) as workbench:
        result = workbench.run(
            limit_event_seq=args.limit_events,
            pending_hook=pending_hook,
            inject_quota=args.inject_quota,
        )
        output = result.as_dict()
        output["ok"] = True
        text = render_view_text(result.view)
        if args.text_out:
            write_text(args.text_out, text)
        if args.inject_snapshot:
            logger.write("injection.snapshot", level="ERROR")
            tampered = workbench.ledger.tamper_content_digest()
            raise SnapshotError(
                "content_digest",
                "injected ledger content digest mismatch",
                batch_id=tampered,
            )
    logger.write(
        "run.ok",
        batches=output["batches_written"],
        cursor=output["final_cursor"],
        declaration_version=output["declaration_version"],
    )
    return output


def _replay(args: argparse.Namespace, logger: StructuredLogger) -> dict[str, Any]:
    if args.inject_snapshot:
        with Ledger(args.db) as ledger:
            tampered = ledger.tamper_content_digest()
        logger.write("injection.snapshot", level="ERROR", batch_id=tampered)
    result = replay_ledger(db_path=args.db, logger=logger)
    return {"ok": True, **result.as_dict()}


def _view(args: argparse.Namespace, logger: StructuredLogger) -> dict[str, Any]:
    view = load_view_from_ledger(args.db)
    text = render_view_text(view)
    if args.text_out:
        write_text(args.text_out, text)
    logger.write(
        "view.rendered",
        declaration_version=view["declaration_version"],
        ledger_batch=view["ledger_batch"],
        view_digest=view["view_digest"],
    )
    return {"ok": True, "view": view, "text": text}


def _ui(args: argparse.Namespace, logger: StructuredLogger) -> dict[str, Any]:
    if args.headless:
        view = load_view_from_ledger(args.db)
        text = render_view_text(view)
        if args.text_out:
            write_text(args.text_out, text)
        logger.write("ui.headless_render", view_digest=view["view_digest"])
        print(text)
        return {"ok": True, "mode": "headless", "view": view}

    def loader() -> dict[str, Any]:
        view = load_view_from_ledger(args.db)
        logger.write("ui.refresh", view_digest=view["view_digest"])
        return view

    launch_ui(loader)
    return {"ok": True, "mode": "tk"}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = StructuredLogger(args.log)
    try:
        if args.command == "validate":
            result = _validate(args, logger)
        elif args.command == "run":
            result = _run(args, logger)
        elif args.command == "replay":
            result = _replay(args, logger)
        elif args.command == "view":
            result = _view(args, logger)
        elif args.command == "ui":
            result = _ui(args, logger)
        else:
            raise AssertionError(args.command)
    except LeasebenchError as exc:
        logger.write("error", level="ERROR", **exc.as_dict())
        print(json.dumps({"ok": False, **exc.as_dict()}, ensure_ascii=False, sort_keys=True))
        return EXIT_CODES.get(exc.code, 1)
    except RuntimeError as exc:
        logger.write(
            "error",
            level="ERROR",
            code="E_UI",
            field="ui",
            message=str(exc),
        )
        print(json.dumps({"ok": False, "code": "E_UI", "message": str(exc)}, ensure_ascii=False))
        return 1
    _write_json(args.json_out, result)
    if args.command != "ui":
        summary = {
            key: value
            for key, value in result.items()
            if key not in {"view", "text"}
        }
        summary["ok"] = True
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
