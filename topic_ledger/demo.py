from __future__ import annotations

import copy
import shutil
import socket
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from .app import load_bundle
from .canonical import canonical_json, fingerprint
from .cluster import SubprocessCluster
from .engine import TopicLedgerEngine
from .errors import BusinessError
from .leases import LeaseManager, SlotPool, monotonic_ms
from .replay import compare_reports, final_full_rebuild_equivalence, run_command_stream
from .samplegen import build_sample_bundle, write_sample_bundle
from .validation import validate_topology


def _versions() -> dict[str, str]:
    try:
        pytest_version = version("pytest")
    except PackageNotFoundError:
        pytest_version = "not-installed"
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pytest": pytest_version,
    }


def _port_check(topology: dict[str, Any]) -> list[dict[str, Any]]:
    checks = []
    for node in topology["nodes"]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            in_use = sock.connect_ex((node["host"], int(node["port"]))) == 0
        checks.append(
            {
                "node_id": node["node_id"],
                "host": node["host"],
                "port": node["port"],
                "available": not in_use,
            }
        )
    return checks


def _validation_probe() -> dict[str, Any]:
    bad = build_sample_bundle()
    bad["topology"]["nodes"][0]["port"] = "not-an-integer"
    try:
        validate_topology(bad["topology"])
    except BusinessError as exc:
        return {"code": exc.code, "field": exc.field, "message": exc.message}
    raise AssertionError("invalid topology was unexpectedly accepted")


def _worker_duplicate_probe(cluster: SubprocessCluster, bundle: dict[str, Any]) -> dict[str, Any]:
    request = {
        "batch_id": "demo-duplicate-initial",
        "shard_id": "demo-duplicate-shard",
        "term_counts": [[0, 2], [4, 3]],
        "document_refs": [{"doc_id": "demo-doc", "revision": 1}],
        "config": bundle["inference"],
        "vocab_size": len(bundle["vocab"]),
    }
    first = cluster.request("node-a", {"op": "sample", **request})
    rejected = cluster.request(
        "node-a",
        {"op": "sample", **request, "batch_id": "demo-duplicate-rejected", "prior_receipt_digest": None},
    )
    accepted = cluster.request(
        "node-a",
        {
            "op": "sample",
            **request,
            "batch_id": "demo-duplicate-overwrite",
            "prior_receipt_digest": first["receipt"]["receipt_digest"],
        },
    )
    return {
        "first_digest": first["receipt"]["receipt_digest"],
        "rejected_code": rejected.get("code"),
        "rejected_batch_id": rejected.get("batch_id"),
        "recorded_rejection": rejected.get("recorded_rejection"),
        "overwrite_ok": accepted.get("ok") is True,
        "elapsed_monotonic_ms": first["receipt"].get("elapsed_monotonic_ms"),
    }


def _lease_probe() -> dict[str, Any]:
    pool = SlotPool(global_concurrency=1, node_quotas={"node-demo": 1})
    ticket = pool.acquire("node-demo", timeout_seconds=1.0)
    manager = LeaseManager(ttl_ms=20, clock=monotonic_ms)
    lease = manager.issue(
        shard_id="demo-expiring-shard",
        node_id="node-demo",
        lease_limit=1,
        batch_id="demo-lease-batch",
        ticket=ticket,
    )
    generation_before = manager.generation
    time.sleep(0.045)
    expired = manager.recover_expired()
    return {
        "shard_id": lease.shard_id,
        "generation": lease.generation,
        "generation_before_recovery": generation_before,
        "expired_count": len(expired),
        "code": manager.ended[-1]["code"],
        "slot_released": ticket.released,
        "global_held_after_recovery": pool.metrics()["global_held"],
    }


def _budget_probe(root: Path, bundle: dict[str, Any]) -> dict[str, Any]:
    low_budget_bundle = copy.deepcopy(bundle)
    low_budget_bundle["inference"]["sample_point_budget"] = 1
    result = run_command_stream(
        bundle=low_budget_bundle,
        state_dir=root / "budget",
        commands=[{"command_id": "budget-run", "op": "run_sampling", "batch_id": "budget-demo"}],
        concurrency=1,
    )
    report = result["report"]
    return {
        "pending_shards": len(report["pending"]),
        "code_count": report["code_counts"].get("E_BUDGET_PENDING", 0),
        "first_pending": report["pending"][0] if report["pending"] else None,
    }


def run_demo(root: str | Path = ".") -> dict[str, Any]:
    root = Path(root).resolve()
    runtime_dir = root / "runtime" / "demo"
    snapshot_dir = root / "snapshots" / "demo"
    artifacts_dir = root / "artifacts"
    for path in (runtime_dir, snapshot_dir):
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    bundle = build_sample_bundle()
    written = write_sample_bundle(root, bundle)
    loaded = load_bundle(root)
    port_checks = _port_check(loaded["topology"])
    if not all(item["available"] for item in port_checks):
        raise RuntimeError(f"declared loopback ports are occupied: {port_checks}")

    cluster = SubprocessCluster.start(
        loaded["topology"], runtime_dir / "cluster", override_free_ports=False
    )
    engine = TopicLedgerEngine(
        topology=loaded["topology"],
        inference=loaded["inference"],
        vocab=loaded["vocab"],
        documents=loaded["documents"],
        state_dir=runtime_dir / "engine",
        snapshot_dir=snapshot_dir,
        dispatcher=cluster.sample,
        global_concurrency_override=3,
    )
    responses = []
    try:
        for command in loaded["commands"]:
            responses.append(engine.handle_command(copy.deepcopy(command)))
        worker_protocol = _worker_duplicate_probe(cluster, loaded)
        worker_rejections = cluster.worker_rejections()
    finally:
        cluster.close()

    main_report = engine.canonical_report()
    main_report["worker_rejections"] = {
        node_id: [item for item in rejections if item["batch_id"] != "demo-duplicate-rejected"]
        for node_id, rejections in worker_rejections.items()
    }
    # The explicit protocol probe is recorded separately and is not part of the command stream.
    for node_id in main_report["worker_rejections"]:
        main_report["worker_rejections"][node_id] = []

    replay = run_command_stream(
        bundle=loaded,
        state_dir=runtime_dir / "blank-replay",
        commands=loaded["commands"],
        use_cluster=True,
        concurrency=2,
        override_free_ports=True,
    )
    replay_comparison = compare_reports(main_report, replay["report"])
    final_equivalence = final_full_rebuild_equivalence(
        source_run={"engine": engine, "report": main_report},
        work_dir=runtime_dir / "final-full",
        use_cluster=True,
    )
    lease_probe = _lease_probe()
    budget_probe = _budget_probe(runtime_dir, loaded)

    result = {
        "schema": "topic-ledger-verification/v1",
        "versions": _versions(),
        "generated_files": [str(path.relative_to(root)) for path in written],
        "port_checks": port_checks,
        "validation_probe": _validation_probe(),
        "worker_duplicate_probe": worker_protocol,
        "lease_recovery_probe": lease_probe,
        "budget_probe": budget_probe,
        "ledger": {
            "all_closed": main_report["closure"]["all_closed"],
            "by_topic": main_report["closure"]["by_topic"],
            "global_distribution_entries": len(main_report["global_distribution"]),
            "authority_decisions": len(main_report["authority"]),
            "overwrite_entries": len(main_report["overwrites"]),
        },
        "incremental_summary": {
            "code_counts": main_report["code_counts"],
            "pending": len(main_report["pending"]),
            "reuse_hit_count": main_report["cache_totals"]["reuse_hit_count"],
            "config_hash": main_report["config_hash"],
            "report_sha256": fingerprint(main_report),
        },
        "command_replay_equality": {
            "equal": replay_comparison["equal"],
            "difference_count": replay_comparison["difference_count"],
            "differences": replay_comparison["differences"][:20],
            "left_sha256": replay_comparison["left_sha256"],
            "right_sha256": replay_comparison["right_sha256"],
        },
        "final_full_rebuild_equality": {
            "equal": final_equivalence["equal"],
            "difference_count": final_equivalence["comparison"]["difference_count"],
            "differences": final_equivalence["comparison"]["differences"][:20],
        },
        "snapshot_pointer": str((snapshot_dir / "current.json").relative_to(root)),
    }
    result["overall_equal"] = bool(
        result["ledger"]["all_closed"]
        and result["command_replay_equality"]["equal"]
        and result["final_full_rebuild_equality"]["equal"]
        and lease_probe["slot_released"]
        and worker_protocol["rejected_code"] == "E_DUPLICATE_DELIVERY"
    )

    output_path = artifacts_dir / "verification-run.json"
    output_path.write_text(canonical_json(result) + "\n", encoding="utf-8")
    markdown_path = artifacts_dir / "verification-summary.md"
    markdown_path.write_text(_render_markdown(result), encoding="utf-8")
    return result


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Verification Run",
        "",
        f"- Overall equality/closure: `{result['overall_equal']}`",
        f"- Python: `{result['versions']['python']}`",
        f"- NumPy: `{result['versions']['numpy']}`",
        f"- pytest: `{result['versions']['pytest']}`",
        f"- Ledger closed: `{result['ledger']['all_closed']}`",
        f"- Command replay equal: `{result['command_replay_equality']['equal']}`",
        f"- Final full rebuild equal: `{result['final_full_rebuild_equality']['equal']}`",
        f"- Lease slot released: `{result['lease_recovery_probe']['slot_released']}`",
        f"- Duplicate rejection batch: `{result['worker_duplicate_probe']['rejected_batch_id']}`",
        f"- Budget pending count: `{result['budget_probe']['code_count']}`",
        "",
        "```json",
        canonical_json(
            {
                "ledger": result["ledger"],
                "lease": result["lease_recovery_probe"],
                "replay": result["command_replay_equality"],
                "final": result["final_full_rebuild_equality"],
            }
        ),
        "```",
        "",
    ]
    return "\n".join(lines)
