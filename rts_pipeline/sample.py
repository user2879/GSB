"""One-command sample corpus, thread/budget matrix, and reproducibility proof."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from .canonical import atomic_write_json, hash_without, remove_tree, sha256_bytes
from .chunker import prepare_chunks
from .constants import ERROR_EXIT_CODES, PIPELINE_VERSION, SCHEMA_VERSION
from .errors import PipelineError
from .generator import generate_workspace
from .runner import load_workspace, run_workspace

SAMPLE_MARKER = ".rts-pipeline-sample"

SCENARIOS = (
    {
        "name": "normal",
        "seed": 20261119,
        "ticks": 36,
        "factions": 4,
        "chunk_ticks": 7,
        "threads": 4,
        "budget_workers": 4,
        "overflow_at": None,
    },
    {
        "name": "overflow",
        "seed": 20261120,
        "ticks": 30,
        "factions": 4,
        "chunk_ticks": 6,
        "threads": 4,
        "budget_workers": 4,
        "overflow_at": (15, "B"),
    },
    {
        "name": "contention",
        "seed": 20261121,
        "ticks": 45,
        "factions": 8,
        "chunk_ticks": 5,
        "threads": 8,
        "budget_workers": 8,
        "overflow_at": None,
    },
    {
        "name": "budget_tight",
        "seed": 20261122,
        "ticks": 42,
        "factions": 6,
        "chunk_ticks": 4,
        "threads": 12,
        "budget_workers": 1,
        "overflow_at": None,
    },
)


def _prepare_sample_root(root: Path) -> None:
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        marker = root / SAMPLE_MARKER
        if not marker.is_file():
            raise PipelineError(
                "E_IO",
                "refusing to replace a non-sample directory",
                path=str(root),
            )
        remove_tree(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / SAMPLE_MARKER).write_text("rts-pipeline-sample-v1\n", encoding="ascii")


def _tree_manifest(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative == "sample_report.json" or relative == SAMPLE_MARKER:
            continue
        hashes[relative] = sha256_bytes(path.read_bytes())
    return hashes


def _tree_hash(manifest: dict[str, str]) -> str:
    digest_input = "".join(
        f"{relative}\0{digest}\n" for relative, digest in sorted(manifest.items())
    ).encode("utf-8")
    return sha256_bytes(digest_input)


def _max_block_bytes(scenario_root: Path) -> int:
    game, rules, events = load_workspace(scenario_root)
    manifest = prepare_chunks(
        scenario_root, events, rules, game, game["chunk_ticks"]
    )
    return max(entry["estimated_memory_bytes"] for entry in manifest["chunks"])


def _scenario_result(root: Path, spec: dict[str, Any], max_block_bytes: int) -> dict[str, Any]:
    threads = int(spec["threads"])
    budget = max_block_bytes * int(spec["budget_workers"])
    if spec["budget_workers"] == 1:
        budget = max_block_bytes + max(1, max_block_bytes // 8)
    run = run_workspace(
        root,
        threads=threads,
        memory_budget_bytes=budget,
        chunk_ticks=spec["chunk_ticks"],
    )
    report = run.get("exit_report")
    if run["status"] != "ok" or report is None:
        return {
            "status": run["status"],
            "result_code": run["result_code"],
            "exit_code": run["exit_code"],
            "exit_report_written": run["exit_report_written"],
        }
    return {
        "status": "ok",
        "result_code": run["result_code"],
        "exit_code": run["exit_code"],
        "exit_report_written": True,
        "business_hash": report["business_hash"],
        "parallel_ledger_hash": report["parallel_ledger_hash"],
        "serial_ledger_hash": report["serial_ledger_hash"],
        "verification_hash": report["verification_hash"],
        "full_consistency": report["summary"]["full_consistency"],
        "truth_match": report["summary"]["truth_match"],
        "event_count": report["summary"]["event_count"],
        "tick_count": report["summary"]["tick_count"],
        "overflow_count": report["summary"]["overflow_count"],
        "threads_requested": threads,
        "memory_budget_bytes": budget,
        "effective_parallelism": report["summary"]["effective_parallelism"],
        "degradation_count": report["summary"]["degradation_count"],
    }


def _matrix(root: Path, max_block_bytes: int, chunk_ticks: int) -> list[dict[str, Any]]:
    configs = (
        (1, max_block_bytes),
        (2, max_block_bytes * 2),
        (4, max_block_bytes * 4),
        (6, max_block_bytes * 6),
        (8, max_block_bytes * 2),
        (8, max_block_bytes),
    )
    rows: list[dict[str, Any]] = []
    for threads, budget in configs:
        run = run_workspace(
            root,
            threads=threads,
            memory_budget_bytes=budget,
            chunk_ticks=chunk_ticks,
        )
        if run["status"] != "ok":
            rows.append(
                {
                    "threads_requested": threads,
                    "memory_budget_bytes": budget,
                    "status": run["status"],
                    "result_code": run["result_code"],
                }
            )
            continue
        report = run["exit_report"]
        rows.append(
            {
                "threads_requested": threads,
                "memory_budget_bytes": budget,
                "effective_parallelism": report["summary"]["effective_parallelism"],
                "business_hash": report["business_hash"],
                "verification_hash": report["verification_hash"],
                "full_consistency": report["summary"]["full_consistency"],
                "truth_match": report["summary"]["truth_match"],
                "degradation_count": report["summary"]["degradation_count"],
                "result_code": run["result_code"],
            }
        )
    return rows


def execute_sample_set(root: Path, base_seed: int | None = None) -> dict[str, Any]:
    root = Path(root)
    scenario_results: dict[str, Any] = {}
    for index, original in enumerate(SCENARIOS):
        spec = dict(original)
        if base_seed is not None:
            spec["seed"] = base_seed + index
        scenario_root = root / spec["name"]
        overflow_at = spec["overflow_at"]
        if isinstance(overflow_at, list):
            overflow_at = tuple(overflow_at)
        generate_workspace(
            scenario_root,
            seed=spec["seed"],
            ticks=spec["ticks"],
            scenario=spec["name"],
            faction_count=spec["factions"],
            chunk_ticks=spec["chunk_ticks"],
            overflow_at=overflow_at,
        )
        max_block = _max_block_bytes(scenario_root)
        scenario_results[spec["name"]] = _scenario_result(
            scenario_root, spec, max_block
        )

    normal_root = root / "normal"
    normal_max = _max_block_bytes(normal_root)
    matrix = _matrix(normal_root, normal_max, SCENARIOS[0]["chunk_ticks"])
    scenario_results["normal"] = _scenario_result(
        normal_root, SCENARIOS[0], normal_max
    )
    matrix_business_hashes = {
        row["business_hash"] for row in matrix if "business_hash" in row
    }
    matrix_complete = all(
        row.get("full_consistency") and row.get("truth_match") for row in matrix
    )
    checks = {
        "all_exit_reports_committed": all(
            result.get("exit_report_written") for result in scenario_results.values()
        ),
        "all_parallel_serial_consistent": all(
            result.get("full_consistency") for result in scenario_results.values()
        ),
        "all_truth_ledgers_match": all(
            result.get("truth_match") for result in scenario_results.values()
        ),
        "normal_result_ok": scenario_results["normal"].get("result_code") == "OK",
        "overflow_detected": scenario_results["overflow"].get("result_code")
        == "E_RESOURCE_OVERFLOW",
        "contention_parallel": scenario_results["contention"].get(
            "effective_parallelism", 0
        )
        > 1,
        "budget_tight_degraded_without_denial": (
            scenario_results["budget_tight"].get("status") == "ok"
            and 0
            < scenario_results["budget_tight"].get("effective_parallelism", 0)
            < scenario_results["budget_tight"].get("threads_requested", 0)
            and scenario_results["budget_tight"].get("degradation_count", 0) > 0
        ),
        "thread_budget_matrix_single_business_hash": (
            len(matrix) == 6
            and matrix_complete
            and len(matrix_business_hashes) == 1
        ),
    }
    return {
        "scenarios": scenario_results,
        "thread_budget_matrix": matrix,
        "checks": checks,
    }


def run_sample(root: Path, base_seed: int | None = None) -> dict[str, Any]:
    root = Path(root)
    _prepare_sample_root(root)
    primary = execute_sample_set(root, base_seed=base_seed)
    primary_tree = _tree_manifest(root)
    with tempfile.TemporaryDirectory(prefix="rts-sample-repro-") as temporary:
        reproduction_root = Path(temporary) / "sample"
        _prepare_sample_root(reproduction_root)
        execute_sample_set(reproduction_root, base_seed=base_seed)
        reproduction_tree = _tree_manifest(reproduction_root)
    byte_identical = primary_tree == reproduction_tree
    differing_files = sorted(
        relative
        for relative in set(primary_tree) | set(reproduction_tree)
        if primary_tree.get(relative) != reproduction_tree.get(relative)
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "base_seed": base_seed,
        "scenarios": primary["scenarios"],
        "thread_budget_matrix": primary["thread_budget_matrix"],
        "checks": primary["checks"],
        "reproducibility": {
            "byte_identical_scenario_trees": byte_identical,
            "primary_tree_hash": _tree_hash(primary_tree),
            "reproduction_tree_hash": _tree_hash(reproduction_tree),
            "compared_file_count": len(primary_tree),
            "differing_files": differing_files,
        },
    }
    report["all_checks_passed"] = all(report["checks"].values()) and byte_identical
    report["sample_report_hash"] = hash_without(report, "sample_report_hash")
    atomic_write_json(root / "sample_report.json", report)
    return report


def sample_exit_code(report: dict[str, Any]) -> int:
    if report.get("all_checks_passed"):
        return 0
    return ERROR_EXIT_CODES["E_INTERNAL"]
