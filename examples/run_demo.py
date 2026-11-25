"""End-to-end reproducible demonstration.

Run from the repository root:

    python examples/run_demo.py

It generates deterministic samples into ``workspace/`` (ignored by git), runs
the full incremental command stream, rebuilds from a blank isolated snapshot
and writes machine-readable artifacts plus a human diagnostic report.
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantcalib.data.samples import generate_workspace  # noqa: E402
from quantcalib.report import write_report  # noqa: E402
from quantcalib.runner import (  # noqa: E402
    comparable_view,
    run_full_rebuild,
    run_incremental,
)
from quantcalib.schedule.commands import write_command_stream  # noqa: E402


def reserve_loopback_port() -> int:
    """Prove loopback bind capability used by local concurrent tooling."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> int:
    workdir = ROOT / "workspace"
    if workdir.exists():
        shutil.rmtree(workdir)
    generate_workspace(workdir)

    commands = json.loads((ROOT / "examples" / "command_stream.json").read_text(
        encoding="utf-8"))
    write_command_stream(workdir / "commands.json", commands)

    port = reserve_loopback_port()
    print(f"loopback bind OK on 127.0.0.1:{port}")

    incremental = run_incremental(workdir, workdir / "commands.json")
    isolated = Path(tempfile.mkdtemp(prefix="quantcalib_rebuild_")) / "snapshot"
    rebuild = run_full_rebuild(workdir, isolated_snapshot=isolated)
    equal = comparable_view(incremental[-1]) == comparable_view(rebuild)

    out = workdir / "out"
    out.mkdir(exist_ok=True)
    (out / "incremental_result.json").write_text(
        json.dumps(incremental[-1], indent=2, sort_keys=True),
        encoding="utf-8")
    (out / "full_rebuild_result.json").write_text(
        json.dumps(rebuild, indent=2, sort_keys=True), encoding="utf-8")
    write_report(out / "diagnostic_report.json", incremental[-1])

    summary = {
        "field_level_equal": equal,
        "command_steps": len(incremental) - 1,
        "final_cache_hits": incremental[-1]["reuse"]["hit_count"],
        "final_recomputes": incremental[-1]["reuse"]["recompute_count"],
        "failures": incremental[-1]["failures"],
        "loopback_port": port,
        "artifacts": [
            str(out / "incremental_result.json"),
            str(out / "full_rebuild_result.json"),
            str(out / "diagnostic_report.json"),
        ],
    }
    (out / "demo_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
