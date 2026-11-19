"""标准异常注入用例（全部经输入/参数生效，可序列化、可复现）。"""

from __future__ import annotations

from .injection import InjectionConfig


def build_injection_suite(center_pixel: tuple[int, int]) -> dict[str, InjectionConfig]:
    row, col = center_pixel
    return {
        "inj_ior_zero": InjectionConfig(
            name="inj_ior_zero",
            body_overrides={"sphere_glass": {"ior": 0.0}},
            pixel_overrides=[],
            render_overrides={},
        ),
        "inj_ior_above_range": InjectionConfig(
            name="inj_ior_above_range",
            body_overrides={"sphere_water": {"ior": 3.5}},
            pixel_overrides=[],
            render_overrides={},
        ),
        "inj_negative_radius": InjectionConfig(
            name="inj_negative_radius",
            body_overrides={"sphere_dense": {"radius": -0.6}},
            pixel_overrides=[],
            render_overrides={},
        ),
        "inj_zero_direction": InjectionConfig(
            name="inj_zero_direction",
            body_overrides={},
            pixel_overrides=[
                {"row": row, "col": col, "patch": {"direction": [0.0, 0.0, 0.0]}}
            ],
            render_overrides={},
        ),
        "inj_depth_zero": InjectionConfig(
            name="inj_depth_zero",
            body_overrides={},
            pixel_overrides=[
                {"row": row, "col": col, "patch": {"max_depth": 0}}
            ],
            render_overrides={},
        ),
        "inj_absorb_negative": InjectionConfig(
            name="inj_absorb_negative",
            body_overrides={"slab_front": {"absorption": [0.02, -0.05, 0.03]}},
            pixel_overrides=[],
            render_overrides={},
        ),
        "inj_depth_zero_render": InjectionConfig(
            name="inj_depth_zero_render",
            body_overrides={},
            pixel_overrides=[],
            render_overrides={"max_depth": 0},
        ),
        "inj_bad_config": InjectionConfig(
            name="inj_bad_config",
            body_overrides={"unknown_body": {"ior": 1.2}},
            pixel_overrides=[],
            render_overrides={},
        ),
    }
