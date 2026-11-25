from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .lease import LeaseClient
from .storage import ArtifactStore, atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--ready-file", required=True)
    parser.add_argument("--exit-report", required=True)
    parser.add_argument("--ttl-ms", type=int, default=100)
    args = parser.parse_args()

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    client = LeaseClient(args.host, args.port)
    lease = client.acquire(
        lease_key="sigkill-recovery-demo",
        holder="do-not-kill-gracefully",
        scopes=["demo:persisted-stage", "demo:unfinished-stage"],
        ttl_ms=args.ttl_ms,
    )

    # A real persisted stage product. Recovery does not depend on process memory.
    store = ArtifactStore(workspace / "cache")
    store.write(
        "sigkill-demo-persisted-stage",
        {"marker": np.asarray([20261126], dtype=np.int64)},
        {"stage": "persisted_before_sigkill", "lease_generation": lease["generation"]},
    )
    atomic_write_json(
        args.ready_file,
        {
            "ready": True,
            "pid": __import__("os").getpid(),
            "lease_id": lease["lease_id"],
            "generation": lease["generation"],
            "expires_ms": lease["expires_ms"],
            "exit_report": args.exit_report,
        },
    )

    # No signal handler and no exit-report writer. SIGKILL cannot be caught.
    while True:
        time.sleep(0.05)


if __name__ == "__main__":
    raise SystemExit(main())
