from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mdet_render.conditions import encode_image, encode_motion, encode_text
from mdet_render.errors import BusinessError
from mdet_render.fixed import FixedPoint
from mdet_render.renderer import Renderer

from conftest import load_workspace


def _condition(type_name: str, values: list[int], digest: str) -> dict:
    return {
        "cache_key": digest,
        "content_digest": digest,
        "stage": "encode",
        "payload": {"condition_type": type_name, "vector": values},
    }


def test_text_bow_uses_vocabulary_order_and_records_hits(sample_root: Path) -> None:
    workspace = load_workspace(sample_root)
    frame = workspace.frames[0]
    condition = encode_text(frame.prompt, workspace.vocabulary, workspace.config.fixed)
    assert condition.values.shape == (40,)
    expected_hits = sorted(workspace.vocabulary.index[term]
                           for term in (*frame.prompt.entities, *frame.prompt.attributes))
    assert condition.details["hit_positions"] == expected_hits
    assert condition.details["hit_terms"] == [workspace.vocabulary.terms[index]
                                              for index in expected_hits]
    assert int(condition.values.sum()) > 0


def test_image_encoding_has_block_means_and_gradients(sample_root: Path) -> None:
    workspace = load_workspace(sample_root)
    image = workspace.image_for(0)
    assert image is not None
    condition = encode_image(image, 4, 2, workspace.config.fixed)
    assert condition.values.shape == (40,)
    cells = condition.details["cells"]
    assert len(cells) == 8
    assert any(any(component != 0 for component in cell["gradient_xy"]) for cell in cells)
    assert cells[0]["fixed"] == condition.values[:5].tolist()


def test_fusion_order_is_type_then_dimension(sample_root: Path, tmp_path: Path) -> None:
    workspace = load_workspace(sample_root)
    renderer = Renderer(workspace, cache_dir=tmp_path / "cache", ledger_path=tmp_path / "ledger.jsonl")
    values = [2] * 40
    conditions = {
        "text": _condition("text", values, "t"),
        "motion": _condition("motion", values, "m"),
        "image": _condition("image", values, "i"),
    }
    _, payload = renderer._fuse(0, conditions)
    trace = payload["trace"]
    assert trace[0]["condition_type"] == "image"
    assert trace[0]["dimension"] == 0
    assert trace[39]["condition_type"] == "image"
    assert trace[39]["dimension"] == 39
    assert trace[40]["condition_type"] == "motion"
    assert trace[80]["condition_type"] == "text"
    assert all("quantization_step" in item and "integer_result" in item and "residual" in item
               for item in trace)
    renderer.close()


def test_fraction_bits_and_quant_level_change_fusion_integers(sample_root: Path, tmp_path: Path) -> None:
    workspace = load_workspace(sample_root)
    renderer = Renderer(workspace, cache_dir=tmp_path / "cache", ledger_path=tmp_path / "ledger.jsonl")

    def fused_for(fixed: FixedPoint):
        workspace.config.fixed = fixed
        frame = workspace.frames[0]
        encoded = [
            encode_text(frame.prompt, workspace.vocabulary, fixed),
            encode_image(workspace.image_for(0), 4, 2, fixed),
            encode_motion(workspace.motion, fixed, frame=0),
        ]
        conditions = {
            item.condition_type: {
                "payload": {"vector": item.values.tolist()},
                "cache_key": item.condition_type,
                "content_digest": item.digest,
            }
            for item in encoded
        }
        return renderer._fuse(0, conditions)

    first_digest, first = fused_for(FixedPoint(6, 32, 0))
    second_digest, second = fused_for(FixedPoint(10, 32, 0))
    third_digest, third = fused_for(FixedPoint(10, 32, 2))
    assert first["vector"] != second["vector"]
    assert second["vector"] != third["vector"]
    assert len({first_digest, second_digest, third_digest}) == 3
    renderer.close()


def test_latent_initialization_ignores_global_numpy_state(sample_root: Path, tmp_path: Path) -> None:
    workspace = load_workspace(sample_root)
    renderer = Renderer(workspace, cache_dir=tmp_path / "cache", ledger_path=tmp_path / "ledger.jsonl")
    fusion = {"content_digest": "abc123"}
    np.random.seed(1)
    d1, p1 = renderer._latent_init(fusion)
    np.random.seed(999)
    _ = np.random.random(1000)
    d2, p2 = renderer._latent_init(fusion)
    assert d1 == d2
    assert p1 == p2
    renderer.close()


def test_denoise_step_records_digest_rounding_and_residual(sample_root: Path, tmp_path: Path) -> None:
    workspace = load_workspace(sample_root)
    renderer = Renderer(workspace, cache_dir=tmp_path / "cache", ledger_path=tmp_path / "ledger.jsonl")
    config = workspace.config
    previous = np.arange(config.height * config.width * 3, dtype=np.int64).reshape(
        config.height, config.width, 3) % 17
    noise = np.full_like(previous, 3)
    fusion = np.arange(config.condition_dim, dtype=np.int64) - 10
    d1, p1 = renderer._denoise_step(previous, fusion, noise, 1, 2)
    d2, p2 = renderer._denoise_step(previous, fusion, noise, 1, 2)
    assert d1 == d2
    assert p1 == p2
    summary = p1["summary"]
    assert summary["output_digest"] == d1
    assert summary["rounding_direction"] in {"down", "up", "mixed", "exact"}
    assert isinstance(summary["residual"], int)
    assert summary["rounding_counts"]["down"] >= 0
    renderer.close()


def test_fixed_point_rejects_unknown_enums() -> None:
    with pytest.raises(BusinessError) as caught:
        FixedPoint(7, 32, 1).validate()
    assert caught.value.code == "E_ENUM_FRACTION_BITS"
    with pytest.raises(BusinessError) as caught:
        FixedPoint(8, 20, 1).validate()
    assert caught.value.code == "E_ENUM_STORAGE_BITS"
    with pytest.raises(BusinessError) as caught:
        FixedPoint(8, 32, 8).validate()
    assert caught.value.code == "E_ENUM_QUANT_LEVEL"
