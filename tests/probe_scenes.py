"""确定性探针场景：用于全反射（TIR）等针对性验证。"""

from __future__ import annotations

from refrac.scene import BackgroundPlane, Camera, DirectionalLight, Scene, TransparentBody


def tir_scene() -> Scene:
    """倾斜厚玻璃块：内部光线在倾斜背面上超过临界角发生全反射，
    随后从板面侧面逸出并射向背景，整个过程无像素隔离。"""
    light = DirectionalLight(direction=(0.3, -0.8, -0.5), color=(1.0, 1.0, 1.0), intensity=1.0)
    background = BackgroundPlane(
        body_id="bg", point=(0.0, 0.0, -5.0), normal=(0.0, 0.0, 1.0),
        base_color=(0.7, 0.7, 0.7), checker_color=(0.2, 0.2, 0.2), checker_size=1.0,
    )
    glass = TransparentBody(
        body_id="tilted_glass", kind="slab", ior=1.8, absorption=(0.02, 0.02, 0.02),
        center=(0.0, 0.0, 0.0), thickness=1.0, half_extent=0.6,
        slab_normal=(0.45, 0.0, -0.893),
    )
    camera = Camera(eye=(0.0, 0.0, 3.0), look_at=(0.0, 0.0, 0.0),
                    up=(0.0, 1.0, 0.0), fov_y_deg=45.0, width=11, height=11)
    return Scene(
        format_version=1, seed=99, camera=camera, light=light, background=background,
        bodies=(glass,), sample_count=4, max_depth=24, generator_tag="tir-block-v1",
    )
