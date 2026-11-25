from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from search_proto.engine import SearchEngine
from search_proto.generator import generate_samples
from search_proto.loader import load_inputs
from search_proto.report import comparison_projection
from search_proto.util import read_json, write_json


ROOT = Path.cwd()


def _load_sample() -> object:
    return load_inputs(
        ROOT / "samples/config.json",
        ROOT / "samples/controlled_vocab.json",
        ROOT / "samples/stopwords.json",
        ROOT / "samples/documents.jsonl",
        ROOT / "samples/queries.jsonl",
        ROOT / "samples/edits.jsonl",
    )


def test_local_sample_incremental_equals_full_and_has_flips() -> None:
    inputs = _load_sample()
    assert not inputs.failures

    incremental = SearchEngine(
        inputs.config,
        inputs.stopwords,
        inputs.vocabulary,
        inputs.documents,
        inputs.queries,
        failures=inputs.failures,
    )
    incremental.materialize()
    incremental.run_batches()
    incremental.apply_edits(inputs.edits)
    incremental.materialize()
    incremental.run_batches()

    pre = SearchEngine(
        inputs.config,
        inputs.stopwords,
        inputs.vocabulary,
        inputs.documents,
        inputs.queries,
        failures=inputs.failures,
    )
    pre.apply_edits(inputs.edits)
    full = SearchEngine(
        pre.config,
        pre.stopwords,
        pre.vocabulary,
        pre.ordered_documents,
        pre.ordered_queries,
        failures=pre.sorted_failures(),
    )
    full.materialize()
    full.run_batches()

    assert comparison_projection(incremental) == comparison_projection(full)
    report = incremental.report()
    assert report["boundary_document_count"] > 0
    assert report["order_flip_count"] > 0
    assert report["cache_entry_counts"]["reused"] > 0
    causes = {
        evidence["cause"]
        for ranking in incremental.query_results.values()
        for evidence in ranking["order_flip"]["evidence"]
    }
    assert "tie_break_rule" in causes
    assert "term_frequency_saturation_rounding" in causes


def test_real_interrupt_then_resume_matches_full(workspace_tmp: Path) -> None:
    samples = workspace_tmp / "samples"
    generate_samples(samples, seed=77, document_count=5, query_count=2)
    config = read_json(samples / "config.json")
    config["batch_point_budget"] = 5
    config["per_query_point_quota"] = 1000
    write_json(samples / "config.json", config)

    common = [
        sys.executable,
        "-m",
        "search_proto",
        "run",
        "--config",
        str(samples / "config.json"),
        "--vocabulary",
        str(samples / "controlled_vocab.json"),
        "--stopwords",
        str(samples / "stopwords.json"),
        "--documents",
        str(samples / "documents.jsonl"),
        "--queries",
        str(samples / "queries.jsonl"),
        "--edits",
        str(samples / "edits.jsonl"),
    ]
    state_dir = workspace_tmp / "state"
    crash = subprocess.run(
        common
        + [
            "--out-dir",
            str(workspace_tmp / "crash-out"),
            "--snapshot-dir",
            str(state_dir),
            "--mode",
            "incremental",
            "--crash-after-batch",
            "1",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert crash.returncode == 137

    resumed = subprocess.run(
        common
        + [
            "--out-dir",
            str(workspace_tmp / "resumed-out"),
            "--snapshot-dir",
            str(state_dir),
            "--mode",
            "incremental",
            "--resume",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert resumed.returncode == 0, resumed.stderr + resumed.stdout
    meta = read_json(workspace_tmp / "resumed-out" / "run_meta.json")
    assert meta["resume"]["decision"] == "resume"

    full = subprocess.run(
        common
        + [
            "--out-dir",
            str(workspace_tmp / "full-out"),
            "--mode",
            "full",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert full.returncode == 0, full.stderr + full.stdout

    for name in ("scores.json", "ranking.json", "failures.json", "index.json", "term_stats.json"):
        assert read_json(workspace_tmp / "resumed-out" / name) == read_json(
            workspace_tmp / "full-out" / name
        ), name
