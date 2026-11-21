from __future__ import annotations

import json
from pathlib import Path

from workbench.generator import generate_dataset

from .conftest import Workspace


def test_invalid_samples_are_isolated_and_sorted(tmp_path: Path) -> None:
    samples = tmp_path / "samples"
    config = tmp_path / "config"
    generate_dataset(samples, config, seed=7, sample_count=4)
    ws = Workspace(
        root=tmp_path,
        samples=samples,
        config_dir=config,
        config_path=config / "alignment.json",
        injections_path=config / "injections.json",
    )

    png = (samples / "sample-02" / "image.png").read_bytes()
    (samples / "sample-02" / "image.png").write_bytes(png[:-32])

    regions_path = samples / "sample-03" / "regions.json"
    regions = json.loads(regions_path.read_text(encoding="utf-8"))
    regions["regions"][0]["rect"] = [250, 10, 40, 20]
    regions_path.write_text(json.dumps(regions), encoding="utf-8")

    (samples / "sample-04" / "text.json").unlink()

    report = ws.run(mode="full")
    failures = report["failures"]
    assert [item["sample"] for item in failures] == ["sample-02", "sample-03", "sample-04"]
    assert {item["code"] for item in failures} >= {
        "E_PNG_TRUNCATED",
        "E_REGION_INVALID",
        "E_TEXT_MISSING",
    }
    for item in failures:
        assert "E_SAMPLE_INVALID" in item["codes"]
    assert report["inputs"]["eligible_sample_names"] == ["sample-01"]
    assert "E_SAMPLE_INVALID" in report["business_codes"]
    assert len(report["retrieval_index"]["queries"]) == 1


def test_missing_required_assertion_is_padded(workspace: Workspace) -> None:
    text_path = workspace.samples / "sample-02" / "text.json"
    document = json.loads(text_path.read_text(encoding="utf-8"))
    document["assertions"] = [
        item for item in document["assertions"] if item["attribute"] != "shape"
    ]
    text_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")

    report = workspace.run(mode="full")
    sample = next(item for item in report["samples"] if item["sample"] == "sample-02")
    assert sample["status"] == "eligible"
    assert sample["padding_positions"]
    text_artifact = json.loads(
        (workspace.root / "output" / "artifacts" / "text_features.json").read_text(
            encoding="utf-8"
        )
    )["samples"]["sample-02"]
    padding_entries = [entry for entry in text_artifact["entries"] if entry["padded"]]
    assert padding_entries
    assert all(entry["rounding_residual"] for entry in padding_entries)
