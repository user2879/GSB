from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from ancient_vlayout.config import load_config
from ancient_vlayout.loader import load_dataset
from ancient_vlayout.util import read_json


def test_valid_loader_decodes_uint8_and_truth(generated_samples: Path) -> None:
    config = load_config(generated_samples / "config.json")
    dataset = load_dataset(generated_samples, config)
    assert dataset.failures == []
    assert sorted(dataset.images) == ["page_01.png", "page_02.png", "page_03.png", "page_04.png"]
    for image in dataset.images.values():
        assert image.array.dtype == np.uint8
        assert image.array.shape == (720, 600)
        assert image.source_mode == "L"
        assert len(image.content_fingerprint) == 64
        assert image.truth["image"] == image.name


def test_invalid_png_and_truth_failures_are_sorted(samples_copy: Path) -> None:
    config = load_config(samples_copy / "config.json")
    raw = (samples_copy / "page_04.png").read_bytes()
    raw_overrides = {"page_04.png": raw[: len(raw) // 2]}

    def break_truth(truth: dict) -> dict:
        mutated = copy.deepcopy(truth)
        mutated["columns"][1]["chars"][2]["bbox"] = [950, 100, 1010, 160]
        return mutated

    dataset = load_dataset(
        samples_copy,
        config,
        raw_overrides=raw_overrides,
        truth_mutators={"page_03.png": break_truth},
    )
    assert [item["file"] for item in dataset.failures] == sorted(
        item["file"] for item in dataset.failures
    )
    assert [(item["file"], item["code"]) for item in dataset.failures] == [
        ("page_03.png", "E_TRUTH_INVALID"),
        ("page_04.png", "E_IMAGE_INVALID"),
    ]
    assert sorted(dataset.images) == ["page_01.png", "page_02.png"]


def test_duplicate_reading_index_is_rejected(samples_copy: Path) -> None:
    config = load_config(samples_copy / "config.json")

    def duplicate_index(truth: dict) -> dict:
        mutated = copy.deepcopy(truth)
        mutated["columns"][0]["chars"][1]["reading_index"] = 0
        return mutated

    dataset = load_dataset(
        samples_copy,
        config,
        truth_mutators={"page_01.png": duplicate_index},
    )
    failure = next(item for item in dataset.failures if item["file"] == "page_01.png")
    assert failure["code"] == "E_TRUTH_INVALID"
    assert "duplicate reading index" in failure["message"]


def test_generator_is_byte_deterministic(tmp_path: Path) -> None:
    from ancient_vlayout.samples import generate_samples
    from ancient_vlayout.util import sha256_bytes

    first = tmp_path / "first"
    second = tmp_path / "second"
    one = generate_samples(first)
    two = generate_samples(second)
    assert one["config_hash"] == two["config_hash"]
    assert one["pages"] == two["pages"]
    for path in sorted(first.iterdir()):
        if path.is_file() and path.suffix in {".png", ".json"}:
            assert sha256_bytes(path.read_bytes()) == sha256_bytes((second / path.name).read_bytes())
