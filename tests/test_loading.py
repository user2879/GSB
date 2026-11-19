from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from layout_workbench.config import RecognitionConfig
from layout_workbench.loading import (
    LoadError,
    load_image_source,
    scan_image_sources,
)
from layout_workbench.cache import CacheStore
from layout_workbench.pipeline import Pipeline
from layout_workbench.samplegen import generate_samples


def test_scan_and_load_valid_sample(sample_data: Path) -> None:
    sources = scan_image_sources(sample_data)
    assert [source.name for source in sources] == ["sample_a", "sample_b", "sample_c"]
    image = load_image_source(sources[0], RecognitionConfig())
    assert image.width == 360
    assert image.height == 220
    assert image.mode == "L"
    assert image.gray.dtype.name == "uint8"
    assert image.gray.shape == (220, 360)
    assert len(image.content_fingerprint) == 64
    assert [line["bbox"] for line in image.truth["lines"]] == [
        [35, 42, 134, 63],
        [35, 124, 122, 145],
    ]


def _copied_source(sample_data: Path, tmp_path: Path, name: str = "sample_a"):
    source = next(s for s in scan_image_sources(sample_data) if s.name == name)
    png = tmp_path / source.png_path.name
    truth = tmp_path / source.truth_path.name
    shutil.copyfile(source.png_path, png)
    shutil.copyfile(source.truth_path, truth)
    return png, truth


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda t: t["lines"][0]["chars"][0].__setitem__("bbox", [40, 42, 60, 63]), "overlap"),
        (lambda t: t["lines"][0]["chars"][1].__setitem__("bbox", [56.0, 42, 71, 63]), "integer"),
        (lambda t: t["lines"][0].__setitem__("bbox", [0, 0, 999, 20]), "outside"),
        (
            lambda t: t["lines"][0]["chars"].reverse(),
            "strictly increasing",
        ),
    ],
)
def test_invalid_truth_returns_business_code(
    sample_data: Path, tmp_path: Path, mutate, message: str
) -> None:
    png, truth = _copied_source(sample_data, tmp_path)
    value = json.loads(truth.read_text(encoding="utf-8"))
    mutate(value)
    truth.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(LoadError) as caught:
        load_image_source(
            scan_image_sources(tmp_path)[0],
            RecognitionConfig(),
        )
    assert caught.value.code == "E_TRUTH_INVALID"
    assert message in caught.value.message


def test_invalid_png_signature_returns_image_invalid(sample_data: Path, tmp_path: Path) -> None:
    png, truth = _copied_source(sample_data, tmp_path)
    png.write_bytes(b"not-a-png")
    with pytest.raises(LoadError) as caught:
        load_image_source(scan_image_sources(tmp_path)[0], RecognitionConfig())
    assert caught.value.code == "E_IMAGE_INVALID"


def test_wrong_resolution_and_mode_are_rejected(sample_data: Path, tmp_path: Path) -> None:
    png, _ = _copied_source(sample_data, tmp_path)
    with pytest.raises(LoadError) as caught:
        load_image_source(scan_image_sources(tmp_path)[0], RecognitionConfig(image_width=100))
    assert caught.value.code == "E_IMAGE_INVALID"

    from PIL import Image

    rgb = tmp_path / "rgb.png"
    Image.open(png).convert("RGB").save(rgb)
    truth = tmp_path / "rgb.truth.json"
    shutil.copyfile(tmp_path / "sample_a.truth.json", truth)
    rgb_source = next(source for source in scan_image_sources(tmp_path) if source.name == "rgb")
    with pytest.raises(LoadError) as caught:
        load_image_source(rgb_source, RecognitionConfig())
    assert caught.value.code == "E_IMAGE_INVALID"
    assert "mode" in caught.value.message


def test_pipeline_failure_list_is_sorted_and_uses_business_code(
    sample_data: Path, tmp_path: Path
) -> None:
    for name in ("sample_a", "sample_b"):
        png, truth = _copied_source(sample_data, tmp_path, name)
        value = json.loads(truth.read_text(encoding="utf-8"))
        value["lines"][0]["bbox"] = [0, 0, 999, 20]
        truth.write_text(json.dumps(value), encoding="utf-8")
    report = Pipeline(RecognitionConfig(), CacheStore(tmp_path / "cache")).run(
        scan_image_sources(tmp_path)
    )
    assert report["images"] == []
    assert [item["image"] for item in report["failures"]] == ["sample_a", "sample_b"]
    assert all(item["code"] == "E_TRUTH_INVALID" for item in report["failures"])


def test_sample_generation_is_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_samples(first)
    generate_samples(second)
    first_files = sorted(path.relative_to(first) for path in first.rglob("*") if path.is_file())
    second_files = sorted(path.relative_to(second) for path in second.rglob("*") if path.is_file())
    assert first_files == second_files
    for relative in first_files:
        assert (first / relative).read_bytes() == (second / relative).read_bytes()
