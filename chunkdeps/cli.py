"""Command-line interface for the local chunk-dependency workbench."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any

from .client import WorkbenchClient
from .errors import (
    E_CHUNK,
    E_CONFLICT,
    E_INPUT,
    EXIT_CODES,
    BusinessError,
)
from .scenario import ScenarioRunner, load_scenario
from .server import StorageService
from .util import (
    atomic_write_json,
    canonical_text,
    chunk_count,
    digest_file,
    iter_file_chunks,
    validate_chunk_size,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chunkdeps",
        description="Local Windows object-chunk upload and metadata dependency workbench",
    )
    parser.add_argument("--host", default="127.0.0.1", help="loopback service host")
    parser.add_argument("--port", type=int, default=8765, help="loopback service port")
    parser.add_argument("--timeout", type=float, default=60.0, help="client timeout seconds")
    parser.add_argument("--json", action="store_true", help="emit stable JSON output")
    sub = parser.add_subparsers(dest="command", required=True)

    server = sub.add_parser("server", help="run the loopback storage simulator")
    server.add_argument("--root", default="workbench", help="workbench data directory")
    server.add_argument("--listen-host", default="127.0.0.1")
    server.add_argument("--listen-port", type=int, default=8765)
    server.add_argument("--global-concurrency", type=int, default=4)
    server.add_argument("--session-concurrency", type=int, default=2)
    server.add_argument("--log", default=None, help="structured service JSONL log")
    server.add_argument("--merge-pause-ms", type=int, default=None, help=argparse.SUPPRESS)
    server.add_argument(
        "--snapshot-on-start",
        action="store_true",
        help="restore the workbench snapshot wholesale before accepting requests",
    )
    server.add_argument("--snapshot-path", default=None)

    split = sub.add_parser("split", help="split a local sample deterministically")
    split.add_argument("--file", required=True)
    split.add_argument("--chunk-size", required=True, type=int)
    split.add_argument("--output-dir", required=True)
    split.add_argument("--algorithm", default="sha256")

    create = sub.add_parser("create", help="create an upload session")
    create.add_argument("--key", required=True)
    create.add_argument("--file", required=True)
    create.add_argument("--chunk-size", required=True, type=int)
    create.add_argument("--tags", default="{}", help="JSON object of string tags")
    create.add_argument("--algorithm", default="sha256")
    create.add_argument("--expected-base-version", type=int, default=None)

    upload = sub.add_parser("upload", help="upload missing chunks for a session")
    upload.add_argument("--session", required=True)
    upload.add_argument("--file", required=True)
    upload.add_argument("--concurrency", type=int, default=4)
    upload.add_argument("--indices", default=None, help="comma-separated explicit reupload indices")
    upload.add_argument("--inject", choices=["truncate-chunk"], default=None)

    put = sub.add_parser("put", help="create, upload and optionally commit a file")
    put.add_argument("--key", required=True)
    put.add_argument("--file", required=True)
    put.add_argument("--chunk-size", required=True, type=int)
    put.add_argument("--tags", default="{}")
    put.add_argument("--algorithm", default="sha256")
    put.add_argument("--concurrency", type=int, default=4)
    put.add_argument("--expected-base-version", type=int, default=None)
    put.add_argument("--no-commit", action="store_true")

    resume = sub.add_parser("resume", help="resume from the persistent bitmap")
    resume.add_argument("--session", required=True)

    abort = sub.add_parser("abort", help="abort a session and retain chunks")
    abort.add_argument("--session", required=True)
    abort.add_argument("--reason", default="operator aborted session")

    commit = sub.add_parser("commit", help="verify and atomically commit a session")
    commit.add_argument("--session", required=True)
    commit.add_argument("--expected-base-version", type=int, default=None)
    commit.add_argument(
        "--inject",
        choices=["corrupt-metadata", "version-conflict"],
        default=None,
    )

    delete = sub.add_parser("delete", help="delete the active metadata version")
    delete.add_argument("--key", required=True)

    sub.add_parser("audit", help="run a full chunk/metadata consistency audit")

    views = sub.add_parser("views", help="query prefix count views")
    views.add_argument("--prefix", default=None)

    graph = sub.add_parser("graph", help="show dependency graph")
    graph.add_argument("--events", action="store_true")

    invalidations = sub.add_parser("invalidations", help="show recent invalidation events")
    invalidations.add_argument("--limit", type=int, default=20)

    sessions = sub.add_parser("sessions", help="list upload sessions")
    sessions.add_argument("--state", default=None)
    sessions.add_argument("--key", default=None)

    objects = sub.add_parser("objects", help="list object versions")
    objects.add_argument("--key", default=None)

    snapshot = sub.add_parser("snapshot", help="save or restore a full workbench snapshot")
    snapshot_sub = snapshot.add_subparsers(dest="snapshot_command", required=True)
    snapshot_save = snapshot_sub.add_parser("save")
    snapshot_save.add_argument("--path", default=None)
    snapshot_restore = snapshot_sub.add_parser("restore")
    snapshot_restore.add_argument("--path", default=None)

    sub.add_parser("recover", help="show restart/merge recovery classification")
    sub.add_parser("status", help="show workbench diagnostics")
    ping = sub.add_parser("ping", help="check the local service")
    ping.add_argument("--show", action="store_true")

    script = sub.add_parser("script", help="run a local JSON scenario")
    script.add_argument("--file", required=True)

    check_port = sub.add_parser("check-port", help="check whether the loopback port is free")
    check_port.add_argument("--check-host", default="127.0.0.1")
    check_port.add_argument("--check-port", type=int, default=8765)

    sub.add_parser("check-env", help="report Python, sqlite3 and tkinter availability")
    return parser


def _client(args: argparse.Namespace) -> WorkbenchClient:
    return WorkbenchClient(args.host, args.port, timeout=args.timeout)


def _parse_tags(value: str) -> dict[str, str]:
    try:
        tags = json.loads(value)
    except ValueError as exc:
        raise BusinessError(E_INPUT, "tags must be valid JSON") from exc
    if not isinstance(tags, dict) or any(
        not isinstance(key, str) or not isinstance(item, str)
        for key, item in tags.items()
    ):
        raise BusinessError(E_INPUT, "tags must be a JSON object of string values")
    return tags


def _parse_indices(value: str | None) -> list[int] | None:
    if value is None:
        return None
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise BusinessError(E_INPUT, "indices must be comma-separated integers") from exc


def _check_port(host: str, port: int) -> bool:
    if not 0 < port < 65536:
        raise BusinessError(E_INPUT, "port is outside the valid range", {"port": port})
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError as exc:
            return False
    return True


def _split_local(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.file)
    chunk_size = validate_chunk_size(args.chunk_size)
    if not source.is_file():
        raise BusinessError(E_CHUNK, "source file does not exist", {"path": str(source)})
    if source.stat().st_size <= 0:
        raise BusinessError(E_CHUNK, "empty source file cannot be split", {"path": str(source)})
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for index, payload, digest in iter_file_chunks(source, chunk_size, args.algorithm):
        name = f"{index:08d}.part"
        (output / name).write_bytes(payload)
        entries.append(
            {
                "chunk_index": index,
                "file": name,
                "size": len(payload),
                "digest": digest,
            }
        )
    manifest = {
        "schema": "chunkdeps-local-split/v1",
        "source": str(source.resolve()),
        "source_size": source.stat().st_size,
        "source_digest": digest_file(source, args.algorithm),
        "chunk_size": chunk_size,
        "chunk_count": chunk_count(source.stat().st_size, chunk_size),
        "digest_algorithm": args.algorithm,
        "chunks": entries,
    }
    manifest_path = output / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    return {
        "output_dir": str(output.resolve()),
        "manifest": str(manifest_path.resolve()),
        **manifest,
    }


def _human_output(command: str, result: dict[str, Any]) -> str:
    if command in {"ping"}:
        return (
            f"service ok port={result.get('port')} "
            f"global={result.get('throttle', {}).get('global_limit')} "
            f"session={result.get('throttle', {}).get('session_limit')}"
        )
    if command == "create":
        return (
            f"session {result['session_id']} state={result['state']} "
            f"chunks={result['chunk_count']} bitmap={result['bitmap']}"
        )
    if command == "upload":
        session = result.get("session", {})
        return (
            f"uploaded={len(result.get('uploaded', []))} "
            f"skipped={len(result.get('skipped_indices', []))} "
            f"state={session.get('state')} bitmap={session.get('bitmap')}"
        )
    if command == "put":
        commit = result.get("commit") or {}
        version = commit.get("version") or {}
        upload = result.get("upload") or {}
        return (
            f"session={result.get('created_session', {}).get('session_id')} "
            f"uploaded={len(upload.get('uploaded', []))} "
            f"version={version.get('version')} state={(commit.get('session') or {}).get('state')}"
        )
    if command == "commit":
        version = result.get("version") or {}
        invalidation = result.get("invalidation") or {}
        return (
            f"committed version={version.get('version')} "
            f"metadata={version.get('metadata_digest')} "
            f"changed_fields={len(result.get('field_changes', []))} "
            f"invalidated={invalidation.get('affected_count')}"
        )
    if command == "resume":
        return (
            f"state={result.get('state')} skipped={len(result.get('skipped_indices', []))} "
            f"missing={len(result.get('missing_indices', []))} bitmap={result.get('bitmap')}"
        )
    if command == "abort":
        return (
            f"state={result.get('session', {}).get('state')} "
            f"chunks_retained={result.get('chunks_retained')} "
            f"invalidated={(result.get('invalidation') or {}).get('affected_count')}"
        )
    if command == "audit":
        return (
            f"audit passed objects={result.get('object_count')} "
            f"chunks={result.get('chunk_count')} run={result.get('run_id')}"
        )
    if command == "snapshot":
        if "mode" in result:
            return f"snapshot restore mode={result['mode']} path={result['path']}"
        return f"snapshot saved path={result['path']} digest={result['state_digest']}"
    if command == "delete":
        return (
            f"deleted key={result.get('object_key')} version={result.get('deleted_version')} "
            f"invalidated={(result.get('invalidation') or {}).get('affected_count')}"
        )
    if command == "split":
        return (
            f"split chunks={result.get('chunk_count')} size={result.get('chunk_size')} "
            f"manifest={result.get('manifest')}"
        )
    if command == "script":
        return f"scenario ok steps={result.get('step_count')}"
    return canonical_text(result)


def _run_server(args: argparse.Namespace) -> int:
    if not _check_port(args.listen_host, args.listen_port):
        raise BusinessError(
            E_CONFLICT,
            "listen port is already occupied",
            {"host": args.listen_host, "port": args.listen_port},
        )
    service = StorageService(
        args.root,
        host=args.listen_host,
        port=args.listen_port,
        global_concurrency=args.global_concurrency,
        session_concurrency=args.session_concurrency,
        log_path=args.log,
        merge_pause_ms=args.merge_pause_ms,
    )
    startup: dict[str, Any] = {
        "event": "server_ready",
        "host": service.host,
        "port": service.port,
        "root": str(service.engine.ledger.root),
        "global_concurrency": service.context.throttle.global_limit,
        "session_concurrency": service.context.throttle.session_limit,
    }
    if args.snapshot_on_start:
        params = {"path": args.snapshot_path} if args.snapshot_path else {}
        startup["snapshot_restore"] = service.engine.snapshot_restore(params)
    print(canonical_text(startup), flush=True)
    try:
        service.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        service.shutdown()
    return 0


def execute(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "server":
        raise BusinessError(E_INPUT, "server is a long-running command")
    if args.command == "split":
        return _split_local(args)
    if args.command == "check-port":
        free = _check_port(args.check_host, args.check_port)
        return {"host": args.check_host, "port": args.check_port, "free": free}
    if args.command == "check-env":
        import sqlite3

        try:
            import tkinter

            tkinter_available = True
            tkinter_version = str(tkinter.TkVersion)
        except Exception:  # noqa: BLE001 - environment probe
            tkinter_available = False
            tkinter_version = None
        return {
            "python": sys.version.split()[0],
            "sqlite3": sqlite3.sqlite_version,
            "tkinter": tkinter_available,
            "tkinter_version": tkinter_version,
            "platform": sys.platform,
        }
    if args.command == "script":
        scenario_path = Path(args.file)
        scenario = load_scenario(scenario_path)
        runner = ScenarioRunner(_client(args), base_dir=scenario_path.parent)
        return runner.run(scenario)

    client = _client(args)
    if args.command == "ping":
        return client.ping()
    if args.command == "status":
        return client.request("status")
    if args.command == "create":
        return client.create_session(
            object_key=args.key,
            source=args.file,
            chunk_size=args.chunk_size,
            tags=_parse_tags(args.tags),
            digest_algorithm=args.algorithm,
            expected_base_version=args.expected_base_version,
        )
    if args.command == "upload":
        return client.upload_session(
            session_id=args.session,
            source=args.file,
            concurrency=args.concurrency,
            reupload_indices=_parse_indices(args.indices),
            inject=args.inject,
        )
    if args.command == "put":
        return client.upload_file(
            object_key=args.key,
            source=args.file,
            chunk_size=args.chunk_size,
            tags=_parse_tags(args.tags),
            digest_algorithm=args.algorithm,
            concurrency=args.concurrency,
            expected_base_version=args.expected_base_version,
            commit=not args.no_commit,
        )
    if args.command == "resume":
        return client.request("resume_session", {"session_id": args.session})
    if args.command == "abort":
        return client.request(
            "abort_session", {"session_id": args.session, "reason": args.reason}
        )
    if args.command == "commit":
        params: dict[str, Any] = {"session_id": args.session}
        if args.expected_base_version is not None:
            params["expected_base_version"] = args.expected_base_version
        if args.inject is not None:
            params["inject"] = args.inject
        return client.request("commit_session", params)
    if args.command == "delete":
        return client.request("delete_object", {"object_key": args.key})
    if args.command == "audit":
        return client.request("audit", {})
    if args.command == "views":
        params = {"prefix": args.prefix} if args.prefix is not None else {}
        return client.request("list_views", params)
    if args.command == "graph":
        return client.request("get_graph", {"include_events": args.events})
    if args.command == "invalidations":
        return client.request("invalidations", {"limit": args.limit})
    if args.command == "sessions":
        params: dict[str, Any] = {}
        if args.state:
            params["state"] = args.state
        if args.key:
            params["object_key"] = args.key
        return client.request("list_sessions", params)
    if args.command == "objects":
        params = {"object_key": args.key} if args.key else {}
        return client.request("list_objects", params)
    if args.command == "snapshot":
        params = {"path": args.path} if args.path else {}
        return client.request(f"snapshot_{args.snapshot_command}", params)
    if args.command == "recover":
        return client.request("recovery_status", {})
    raise BusinessError(E_INPUT, "unsupported command", {"command": args.command})


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    globals_with_value = {"--host", "--port", "--timeout"}
    globals_without_value = {"--json"}
    moved: list[str] = []
    remaining: list[str] = []
    index = 0
    while index < len(raw_argv):
        token = raw_argv[index]
        if token in globals_with_value:
            moved.append(token)
            if index + 1 < len(raw_argv):
                moved.append(raw_argv[index + 1])
                index += 2
                continue
        elif any(token.startswith(option + "=") for option in globals_with_value):
            moved.append(token)
        elif token in globals_without_value:
            moved.append(token)
        else:
            remaining.append(token)
        index += 1
    argv = moved + remaining
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "server":
        try:
            return _run_server(args)
        except BusinessError as exc:
            if args.json:
                print(canonical_text({"ok": False, "error": exc.as_dict()}))
            else:
                print(f"[{exc.code}] {exc.message}: {canonical_text(exc.details)}", file=sys.stderr)
            return EXIT_CODES.get(exc.code, 1)
        except OSError as exc:
            error = BusinessError(E_CONFLICT, "server failed to bind or start", {"reason": str(exc)})
            if args.json:
                print(canonical_text({"ok": False, "error": error.as_dict()}))
            else:
                print(f"[{error.code}] {error.message}: {canonical_text(error.details)}", file=sys.stderr)
            return EXIT_CODES[E_CONFLICT]
    try:
        result = execute(args)
    except BusinessError as exc:
        error_envelope = {"ok": False, "error": exc.as_dict()}
        if args.json:
            print(canonical_text(error_envelope))
        else:
            print(f"[{exc.code}] {exc.message}", file=sys.stderr)
            if exc.details:
                print(canonical_text(exc.details), file=sys.stderr)
        return EXIT_CODES.get(exc.code, 1)
    except OSError as exc:
        error = BusinessError(E_CONFLICT, "local I/O or network failure", {"reason": str(exc)})
        if args.json:
            print(canonical_text({"ok": False, "error": error.as_dict()}))
        else:
            print(f"[{error.code}] {error.message}: {canonical_text(error.details)}", file=sys.stderr)
        return EXIT_CODES[E_CONFLICT]
    if args.json:
        print(canonical_text(result))
    else:
        print(_human_output(args.command, result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
