from __future__ import annotations

import json
from pathlib import Path

from cloudshard.demo_spec import build_demo_declaration
from cloudshard.pipeline import check_batch_boundaries, execute_pipeline
from cloudshard.verifier import verify_spec_file


def test_full_local_pipeline_and_verify(tmp_path: Path):
    spec_path = tmp_path / "declaration.json"
    spec_path.write_text(json.dumps(build_demo_declaration()), encoding="utf-8")
    workdir = tmp_path / "run"
    report = execute_pipeline(spec_path, workdir)
    verification = verify_spec_file(spec_path, Path(report["ledger"]))
    assert verification["ok"] is True
    assert verification["outcomes_verified"] == 13
    assert check_batch_boundaries(__import__("cloudshard.spec", fromlist=["load_spec"]).load_spec(spec_path))["all_equal"]


def test_demo_sample_file_is_valid():
    from cloudshard.spec import load_spec

    spec = load_spec(Path(__file__).resolve().parents[1] / "samples" / "declaration.json")
    assert len(spec.nodes) == 3
    assert len(spec.shards) == 8
    assert len(spec.events) == 8
