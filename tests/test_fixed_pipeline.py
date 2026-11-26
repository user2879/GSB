from __future__ import annotations

import random

import numpy as np
import pytest

from mmfm.conditions import encode_image, encode_motion, encode_text
from mmfm.errors import BusinessError
from mmfm.fixed import weighted_accumulate
from mmfm.pipeline import chunk_partition, decode_rows
from mmfm.renderer import Renderer


def condition_artifacts(workspace, profile):
    frame = workspace.conditions.frames[0]
    config = workspace.config
    return {
        "text": encode_text(
            frame.prompt,
            workspace.conditions.vocab,
            profile,
            config.condition_dim,
        ),
        "image": encode_image(
            frame.image_png,
            config=config,
            profile=profile,
            asset_id=frame.image_id,
        ),
        "motion": encode_motion(
            frame.motion_vectors,
            config=config,
            profile=profile,
            asset_id=frame.motion_id,
        ),
    }


def test_quant_profiles_change_intermediate_integers_and_residuals(workspace):
    q4 = workspace.config.quant_profiles["q4"]
    q12 = workspace.config.quant_profiles["q12"]
    low = condition_artifacts(workspace, q4)
    high = condition_artifacts(workspace, q12)
    fused_low = weighted_accumulate(
        {kind: artifact["vector"] for kind, artifact in low.items()},
        workspace.config.fusion_weights_permille,
        ["image", "motion", "text"],
        q4,
    )
    fused_high = weighted_accumulate(
        {kind: artifact["vector"] for kind, artifact in high.items()},
        workspace.config.fusion_weights_permille,
        ["image", "motion", "text"],
        q12,
    )
    assert fused_low["vector"] != fused_high["vector"]
    assert fused_low["residual_total"] != fused_high["residual_total"]
    order = [(item["condition_type"], item["dimension"]) for item in fused_high["stages"]]
    assert order == sorted(order)


def test_text_encoder_records_ordered_hits(workspace):
    frame = workspace.conditions.frames[0]
    artifact = encode_text(
        frame.prompt,
        workspace.conditions.vocab,
        workspace.config.quant_profiles["q8"],
        workspace.config.condition_dim,
    )
    assert len(artifact["vector"]) == workspace.config.condition_dim
    positions = [hit["index"] for hit in artifact["hits"]]
    vocab_order = [
        term.index
        for term in workspace.conditions.vocab.terms
        if term.term in set(frame.prompt["entities"]) | set(frame.prompt["attributes"])
    ]
    assert positions == vocab_order


def test_render_is_independent_of_global_random_state(workspace, tmp_path):
    random.seed(1)
    np.random.seed(1)
    first = Renderer(workspace, tmp_path / "a", budget_override=100).render()
    random.seed(999)
    np.random.seed(999)
    second = Renderer(workspace, tmp_path / "b", budget_override=100).render()
    assert [frame["pixel_digest"] for frame in first["frames"]] == [
        frame["pixel_digest"] for frame in second["frames"]
    ]
    assert [frame["png_sha256"] for frame in first["frames"]] == [
        frame["png_sha256"] for frame in second["frames"]
    ]


def test_chunk_partition_is_disjoint_and_decode_oob_rejected(workspace):
    partitions = chunk_partition(workspace.config.image.height, 3)
    assert [(item["start_row"], item["end_row"]) for item in partitions] == [(0, 10), (10, 21), (21, 32)]
    assert all(item["start_row"] < item["end_row"] for item in partitions)
    with pytest.raises(BusinessError) as caught:
        decode_rows([0] * workspace.config.latent_dim, config=workspace.config, start_row=-1, end_row=2)
    assert caught.value.code == "E_DECODE_OOB"
    with pytest.raises(BusinessError) as caught:
        decode_rows(
            [0] * workspace.config.latent_dim,
            config=workspace.config,
            start_row=0,
            end_row=workspace.config.image.height + 1,
        )
    assert caught.value.code == "E_DECODE_OOB"
