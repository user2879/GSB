"""业务要求 7：异常注入的序列化、副本隔离与分类业务码。"""

from __future__ import annotations

import json

import numpy as np
import pytest

from photometric_altopt.errors import E_VIEW_SKIPPED, BusinessError
from photometric_altopt.faults import (
    FaultConfig,
    build_valid_mask,
    fault_summary,
    inject_nonfinite_albedo,
    inject_zero_light,
    validate_skipped_views,
)


def test_fault_config_roundtrip_json():
    fault = FaultConfig(
        nonfinite_albedo=True,
        nonfinite_pixels=[[1, 2]],
        zero_light=True,
        skipped_views=[3],
    )
    text = json.dumps(fault.to_dict(), sort_keys=True)
    restored = FaultConfig.from_dict(json.loads(text))
    assert restored == fault
    assert fault_summary(restored)["active"] is True


def test_injection_uses_copies():
    albedo = np.ones((4, 4))
    fault = FaultConfig(nonfinite_albedo=True, nonfinite_pixels=[[0, 0]])
    polluted = inject_nonfinite_albedo(albedo, fault)
    assert np.isnan(polluted[0, 0])
    assert np.all(np.isfinite(albedo))  # 原数组未被修改

    lights = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    changed = inject_zero_light(lights, FaultConfig(zero_light=True, zero_light_index=1))
    assert changed[1] == [0.0, 0.0, 0.0]
    assert lights[1] == [0.0, 1.0, 0.0]  # 原清单未被修改


def test_valid_mask_skips_views():
    mask = build_valid_mask(6, 3, 3, [1, 4])
    assert not mask[1].any()
    assert not mask[4].any()
    assert mask[0].all() and mask[5].all()


def test_validate_skipped_view_index():
    with pytest.raises(BusinessError) as exc:
        validate_skipped_views([99], num_views=6)
    assert exc.value.code == E_VIEW_SKIPPED
    assert exc.value.layer == "pipeline"


def test_ill_conditioned_threshold_serialization():
    fault = FaultConfig(ill_conditioned=True, ill_conditioned_threshold=1.0)
    assert FaultConfig.from_dict(fault.to_dict()).ill_conditioned_threshold == 1.0
