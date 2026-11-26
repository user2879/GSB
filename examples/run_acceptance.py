#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def run(arguments: list[str], *, expected: int = 0, cwd: Path = REPO_ROOT) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT)
    completed = subprocess.run(
        [sys.executable, "-m", "affinity_scheduler", *arguments],
        cwd=str(cwd),
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != expected:
        raise SystemExit(
            f"command failed ({completed.returncode} != {expected}):\n"
            f"args={arguments}\nstdout={completed.stdout}\nstderr={completed.stderr}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"command did not return JSON: {arguments}\n{completed.stdout}\n{completed.stderr}"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the complete local acceptance workflow")
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--summary-out")
    args = parser.parse_args(argv)
    workdir = Path(args.workdir).resolve()
    existing_generated = [
        workdir / name
        for name in ("config", "logs", "data")
        if (workdir / name).exists() and any((workdir / name).iterdir())
    ]
    if existing_generated:
        raise SystemExit(
            "generated directories must be absent or empty: "
            + ", ".join(str(path) for path in existing_generated)
        )
    workdir.mkdir(parents=True, exist_ok=True)
    log_file = workdir / "logs" / "acceptance.jsonl"
    common = ["--log-file", str(log_file)]
    spec = workdir / "config" / "spec.json"
    spec_v2 = workdir / "config" / "spec-v2.json"
    records = workdir / "data" / "records.jsonl"
    state = workdir / "data" / "state.json"
    ledger = workdir / "data" / "ledger.sqlite3"

    generated = run([*common, "generate-sample", "--root", str(workdir)])
    validated = run([*common, "validate-spec", "--spec", str(spec)])
    mixed = run([*common, "normalize-records", "--records", str(records), "--spec", str(spec)])
    mixed_hashes = {
        row["pod_id"]: row["conclusion_bytes_sha256"] for row in mixed["records"]
    }
    if mixed["format_versions"] != [1, 2, 3]:
        raise SystemExit(f"unexpected mixed versions: {mixed['format_versions']}")

    migrated = run(
        [
            *common,
            "migrate",
            "--records",
            str(records),
            "--spec",
            str(spec),
            "--db",
            str(ledger),
            "--batch-size",
            "1",
        ]
    )
    if not migrated["migration_verification"]["conclusion_bytes_equal"]:
        raise SystemExit("migration changed a conclusion byte")
    normalized_after = run(
        [*common, "normalize-records", "--records", str(records), "--spec", str(spec)]
    )
    migrated_hashes = {
        row["pod_id"]: row["conclusion_bytes_sha256"] for row in normalized_after["records"]
    }
    if mixed_hashes != migrated_hashes:
        raise SystemExit("post-migration hashes differ from pre-migration hashes")

    scheduled_v1 = run(
        [
            *common,
            "schedule",
            "--spec",
            str(spec),
            "--records",
            str(records),
            "--state",
            str(state),
            "--ledger",
            str(ledger),
        ]
    )
    incremental_check = run(
        [
            *common,
            "verify-incremental",
            "--spec",
            str(spec_v2),
            "--state",
            str(state),
        ]
    )
    if not incremental_check["consistent"]:
        raise SystemExit("incremental and full recomputation differ")
    scheduled_v2 = run(
        [
            *common,
            "schedule",
            "--spec",
            str(spec_v2),
            "--records",
            str(records),
            "--state",
            str(state),
            "--ledger",
            str(ledger),
        ]
    )
    verified = run(
        [
            *common,
            "verify",
            "--ledger",
            str(ledger),
            "--spec",
            str(spec_v2),
            "--records",
            str(records),
        ]
    )

    width = run(
        [
            *common,
            "normalize-records",
            "--records",
            str(records) + ".readonly.bak",
            "--spec",
            str(spec_v2),
            "--inject",
            "v1-width",
        ],
        expected=2,
    )
    corruption = run(
        [
            *common,
            "normalize-records",
            "--records",
            str(records) + ".readonly.bak",
            "--spec",
            str(spec_v2),
            "--inject",
            "record-corruption",
        ],
        expected=2,
    )
    snapshot = run(
        [
            *common,
            "ledger-sync",
            "--ledger",
            str(ledger),
            "--spec",
            str(spec_v2),
            "--records",
            str(records),
            "--state",
            str(state),
            "--inject-digest",
        ],
        expected=2,
    )
    if width["code"] != "E_WIDTH":
        raise SystemExit("v1 width injection did not report E_WIDTH")
    if corruption["code"] != "E_FORMAT":
        raise SystemExit("record corruption did not report E_FORMAT")
    if snapshot["code"] != "E_SNAPSHOT":
        raise SystemExit("ledger digest injection did not report E_SNAPSHOT")
    recovery = run([*common, "ledger-recover", "--ledger", str(ledger)])

    result = {
        "ok": True,
        "workdir": str(workdir),
        "port_base": generated["port_base"],
        "spec": {
            "nodes": validated["node_count"],
            "pods": validated["pod_count"],
            "parameter_version": validated["parameter_version"],
            "scale_factor": validated["scale_factor"],
        },
        "records": {
            "mixed_format_versions": mixed["format_versions"],
            "post_migration_format_versions": normalized_after["format_versions"],
            "record_count": normalized_after["record_count"],
            "conclusion_hashes_equal": mixed_hashes == migrated_hashes,
            "readonly_backup": migrated["backup"],
        },
        "scheduling_v1": {
            "bindings": {
                pod_id: value["node_name"]
                for pod_id, value in scheduled_v1["bindings"].items()
            },
            "conclusion_digests": {
                pod_id: value["conclusion_digest"]
                for pod_id, value in scheduled_v1["bindings"].items()
            },
        },
        "incremental": {
            "affected_pods": incremental_check["affected_pods"],
            "affected_nodes": incremental_check["affected_nodes"],
            "consistent_with_full": incremental_check["consistent"],
            "unaffected_pods_byte_identical": incremental_check[
                "unaffected_pods_byte_identical"
            ],
            "scheduled_v2_rescored": scheduled_v2["affected_pods"],
            "ledger_snapshot": scheduled_v2["ledger_snapshot"],
        },
        "verification": verified,
        "fault_injection": {
            "v1_width": width["code"],
            "record_corruption": corruption["code"],
            "ledger_digest": snapshot["code"],
        },
        "recovery": recovery,
        "structured_log": str(log_file),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False)
    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
