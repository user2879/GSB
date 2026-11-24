from __future__ import annotations

import json
import socket
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable

from receipt_diag.engine import DiagnosticsEngine
from receipt_diag.ledger import Ledger
from receipt_diag.samplegen import basic_sample, changed_sample
from receipt_diag.specs import load_specs


def free_ports(count: int = 2) -> list[int]:
    sockets: list[socket.socket] = []
    ports: list[int] = []
    try:
        for _ in range(count):
            holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            holder.bind(("127.0.0.1", 0))
            holder.listen(1)
            sockets.append(holder)
            ports.append(holder.getsockname()[1])
    finally:
        for holder in sockets:
            holder.close()
    return ports


def write_sample(
    directory: str | Path,
    *,
    changed: bool = False,
    mutate: Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], None] | None = None,
    sort_keys: bool = True,
) -> Path:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    base = basic_sample(port_base=43171)
    documents = changed_sample(base) if changed else base
    request_doc, worker_doc, rules_doc = documents
    ports = free_ports(2)
    for worker, port in zip(worker_doc["workers"], ports):
        worker["port"] = port
    if mutate is not None:
        mutate(request_doc, worker_doc, rules_doc)
    for name, document in zip(
        ("request_stream.json", "worker_spec.json", "merge_rules.json"),
        (request_doc, worker_doc, rules_doc),
    ):
        (target / name).write_text(
            json.dumps(document, ensure_ascii=False, indent=2, sort_keys=sort_keys) + "\n",
            encoding="utf-8",
        )
    return target


def run_fresh(
    sample_dir: str | Path,
    root: str | Path,
    *,
    receipt_format: str | None = None,
    injection: str = "none",
) -> tuple[dict[str, Any], Ledger, DiagnosticsEngine]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    spec = load_specs(sample_dir)
    engine = DiagnosticsEngine(spec, root / "artifacts", receipt_format=receipt_format)
    ledger = Ledger(root / "ledger.db")
    report = engine.run(ledger, injection=injection)
    return report, ledger, engine
