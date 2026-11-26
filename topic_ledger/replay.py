from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from typing import Any

from .canonical import fingerprint
from .cluster import SubprocessCluster
from .engine import TopicLedgerEngine


def deep_diff(left: Any, right: Any, path: str = "$", limit: int = 50) -> list[str]:
    differences: list[str] = []

    def visit(a: Any, b: Any, current_path: str) -> None:
        if len(differences) >= limit:
            return
        if type(a) is not type(b):
            differences.append(f"{current_path}: type {type(a).__name__} != {type(b).__name__}")
            return
        if isinstance(a, dict):
            keys = sorted(set(a) | set(b), key=str)
            for key in keys:
                child = f"{current_path}.{key}"
                if key not in a:
                    differences.append(f"{child}: missing in left")
                elif key not in b:
                    differences.append(f"{child}: missing in right")
                else:
                    visit(a[key], b[key], child)
        elif isinstance(a, list):
            if len(a) != len(b):
                differences.append(f"{current_path}: length {len(a)} != {len(b)}")
            for index, (item_a, item_b) in enumerate(zip(a, b)):
                visit(item_a, item_b, f"{current_path}[{index}]")
        elif a != b:
            differences.append(f"{current_path}: {a!r} != {b!r}")

    visit(left, right, path)
    return differences


def compare_reports(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    differences = deep_diff(left, right)
    return {
        "equal": not differences,
        "difference_count": len(differences),
        "differences": differences,
        "left_sha256": fingerprint(left),
        "right_sha256": fingerprint(right),
    }


def run_command_stream(
    *,
    bundle: dict[str, Any],
    state_dir: str | Path,
    commands: list[dict[str, Any]] | None = None,
    use_cluster: bool = False,
    concurrency: int = 1,
    override_free_ports: bool = True,
    clock: Any = "manual",
) -> dict[str, Any]:
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    cluster: SubprocessCluster | None = None
    dispatcher = None
    topology = copy.deepcopy(bundle["topology"])
    try:
        if use_cluster:
            cluster = SubprocessCluster.start(
                topology,
                state_dir / "cluster",
                override_free_ports=override_free_ports,
            )
            dispatcher = cluster.sample
        engine = TopicLedgerEngine(
            topology=topology,
            inference=copy.deepcopy(bundle["inference"]),
            vocab=copy.deepcopy(bundle["vocab"]),
            documents=copy.deepcopy(bundle["documents"]),
            state_dir=state_dir / "engine",
            dispatcher=dispatcher,
            clock=clock,
            global_concurrency_override=concurrency,
        )
        responses = []
        for command in commands if commands is not None else bundle.get("commands", []):
            responses.append(engine.handle_command(copy.deepcopy(command)))
        report = engine.canonical_report()
        worker_rejections = cluster.worker_rejections() if cluster is not None else {}
        report["worker_rejections"] = worker_rejections
        return {
            "engine": engine,
            "cluster": cluster,
            "responses": responses,
            "report": report,
            "state_dir": state_dir,
        }
    finally:
        if cluster is not None:
            cluster.close()


def replay_equivalence(
    bundle: dict[str, Any],
    commands: list[dict[str, Any]],
    *,
    work_root: str | Path | None = None,
    use_cluster: bool = False,
    concurrency_left: int = 1,
    concurrency_right: int = 3,
) -> dict[str, Any]:
    temporary_root: tempfile.TemporaryDirectory[str] | None = None
    if work_root is None:
        temporary_root = tempfile.TemporaryDirectory(prefix="topic-ledger-replay-")
        root = Path(temporary_root.name)
    else:
        root = Path(work_root)
        root.mkdir(parents=True, exist_ok=True)

    left = run_command_stream(
        bundle=bundle,
        state_dir=root / "incremental",
        commands=commands,
        use_cluster=use_cluster,
        concurrency=concurrency_left,
    )
    right = run_command_stream(
        bundle=bundle,
        state_dir=root / "blank-replay",
        commands=commands,
        use_cluster=use_cluster,
        concurrency=concurrency_right,
    )
    comparison = compare_reports(left["report"], right["report"])
    result = {
        "equal": comparison["equal"],
        "comparison": comparison,
        "left_report": left["report"],
        "right_report": right["report"],
        "left_state_dir": str(root / "incremental"),
        "right_state_dir": str(root / "blank-replay"),
        "use_cluster": use_cluster,
        "concurrencies": [concurrency_left, concurrency_right],
    }
    if temporary_root is not None:
        temporary_root.cleanup()
    return result


SEMANTIC_FIELDS = (
    "documents",
    "shard_receipts",
    "authority",
    "overwrites",
    "global_distribution",
    "closure",
)


def semantic_report(report: dict[str, Any]) -> dict[str, Any]:
    return {field: report[field] for field in SEMANTIC_FIELDS}


def final_full_rebuild_equivalence(
    *,
    source_run: dict[str, Any],
    work_dir: str | Path,
    use_cluster: bool = False,
) -> dict[str, Any]:
    engine: TopicLedgerEngine = source_run["engine"]
    final_bundle = {
        "topology": engine.topology,
        "inference": engine.inference,
        "vocab": engine.vocab,
        "documents": list(engine.documents.values()),
    }
    full_run = run_command_stream(
        bundle=final_bundle,
        state_dir=Path(work_dir),
        commands=[{"command_id": "full-01", "op": "run_sampling", "batch_id": "full"}],
        use_cluster=use_cluster,
        concurrency=engine.effective_concurrency,
    )
    left = semantic_report(source_run["report"])
    right = semantic_report(full_run["report"])
    comparison = compare_reports(left, right)
    return {
        "equal": comparison["equal"],
        "comparison": comparison,
        "full_report": full_run["report"],
    }
