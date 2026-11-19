#!/usr/bin/env python3
"""Cross-platform end-to-end demonstration; expected faults are asserted."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SESSION = ROOT / "sessions" / "demo"


def remove_runtime_tree(path: Path) -> None:
    if not path.exists():
        return
    for child in sorted(path.rglob("*"), reverse=True):
        try:
            if child.is_dir():
                child.chmod(0o755)
            else:
                child.chmod(0o644)
        except OSError:
            pass
    try:
        path.chmod(0o755)
    except OSError:
        pass
    shutil.rmtree(path)


def run(*args: str, expected_exit: int = 0) -> dict:
    command = [sys.executable, "-m", "docdb_index", *args]
    print("$", " ".join(command[1:]))
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if process.returncode != expected_exit:
        print(process.stdout)
        print(process.stderr, file=sys.stderr)
        raise SystemExit(
            f"unexpected exit {process.returncode}; expected {expected_exit}"
        )
    payload = json.loads(process.stdout) if process.stdout.strip() else {}
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return payload


def main() -> int:
    remove_runtime_tree(SESSION)
    run(
        "init",
        "--session",
        str(SESSION),
        "--input",
        "samples/collections.json",
        "--reset",
    )
    run(
        "apply",
        "--session",
        str(SESSION),
        "--input",
        "samples/changes.json",
        "--batch-id",
        "incremental",
        "--batch-size",
        "2",
    )
    indexed = run(
        "query",
        "--session",
        str(SESSION),
        "--collection",
        "orders",
        "--index",
        "metrics.amount",
        "--lo",
        "-1000",
        "--hi",
        "100000",
        "--limit",
        "10000",
    )
    full = run(
        "scan-full",
        "--session",
        str(SESSION),
        "--collection",
        "orders",
        "--index",
        "metrics.amount",
        "--lo",
        "-1000",
        "--hi",
        "100000",
        "--limit",
        "10000",
    )
    if indexed["items"] != full["items"]:
        raise SystemExit("index query and full scan differ")
    print(f"verified index/full-scan equality on {len(indexed['items'])} documents")

    run(
        "apply",
        "--session",
        str(SESSION),
        "--input",
        "samples/changes.json",
        "--batch-id",
        "isolation",
        "--retain-stale",
    )
    retained = run(
        "query",
        "--session",
        str(SESSION),
        "--collection",
        "orders",
        "--index",
        "metrics.amount",
        "--lo",
        "-1000",
        "--hi",
        "100000",
        "--limit",
        "10000",
    )
    reference = run(
        "scan-full",
        "--session",
        str(SESSION),
        "--collection",
        "orders",
        "--index",
        "metrics.amount",
        "--lo",
        "-1000",
        "--hi",
        "100000",
        "--limit",
        "10000",
    )
    if retained["items"] != reference["items"]:
        raise SystemExit("stale isolation query differs from full scan")
    print(f"stale isolation filtered {retained['filtered_stale']} physical references")

    run(
        "diagnose",
        "--session",
        str(SESSION),
        "--collection",
        "orders",
        "--index",
        "metrics.amount",
    )
    run("reclaim", "--session", str(SESSION), "--collection", "orders")
    run("verify", "--session", str(SESSION))
    run(
        "snapshot",
        "create",
        "--session",
        str(SESSION),
        "--label",
        "snap-000001",
    )

    more_changes = SESSION / "demo-more.json"
    more_changes.write_text(
        json.dumps(
            {
                "collection": "orders",
                "changes": [
                    {
                        "op": "set_path",
                        "pk": "doc-000005",
                        "path": "metrics.amount",
                        "value": 123456,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    run(
        "apply",
        "--session",
        str(SESSION),
        "--input",
        str(more_changes),
        "--batch-id",
        "post-snapshot",
    )
    run(
        "snapshot",
        "restore",
        "--session",
        str(SESSION),
        "--snapshot",
        "snap-000001",
    )

    for suffix, kind, expected_code in (
        ("segment", "segment", "E_SEGMENT"),
        ("bound", "bound", "E_BOUND"),
        ("ledger", "ledger", "E_LEDGER"),
    ):
        fault_session = ROOT / "sessions" / f"demo-fault-{suffix}"
        remove_runtime_tree(fault_session)
        run(
            "init",
            "--session",
            str(fault_session),
            "--input",
            "samples/collections.json",
            "--reset",
        )
        run(
            "apply",
            "--session",
            str(fault_session),
            "--input",
            "samples/changes.json",
            "--batch-id",
            "incremental",
        )
        run(
            "inject",
            "--session",
            str(fault_session),
            "--collection",
            "orders",
            "--index",
            "metrics.amount",
            "--type",
            kind,
        )
        failure = run(
            "verify",
            "--session",
            str(fault_session),
            expected_exit=2,
        )
        if failure.get("code") != expected_code:
            raise SystemExit(
                f"fault {kind}: expected {expected_code}, got {failure.get('code')}"
            )
    print("demo completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
