"""Integer projection helpers shared by generation and reconstruction."""

from __future__ import annotations

from .fixed import div_round_nearest
from .models import Camera


def project_world(
    camera: Camera,
    world_x: int,
    world_y: int,
    world_z: int,
) -> tuple[int, int, int]:
    """Project a physical integer point and return (u, v, depth)."""
    projection = camera.projection_numerator()
    homogeneous = (world_x, world_y, world_z, 1)
    first = sum(projection[0][index] * homogeneous[index] for index in range(4))
    second = sum(projection[1][index] * homogeneous[index] for index in range(4))
    third = sum(projection[2][index] * homogeneous[index] for index in range(4))
    if third <= 0:
        return -1, -1, third
    u, _, _ = div_round_nearest(first, third)
    v, _, _ = div_round_nearest(second, third)
    depth, _, _ = div_round_nearest(third, camera.rotation_scale)
    return u, v, depth


def project_fixed(
    camera: Camera,
    x_fixed: int,
    y_fixed: int,
    z_fixed: int,
    q_scale_value: int,
) -> tuple[int, int, int] | None:
    """Project a Q-format point into integer image coordinates and Q depth."""
    projection = camera.projection_numerator()
    homogeneous = (x_fixed, y_fixed, z_fixed, q_scale_value)
    first = sum(projection[0][index] * homogeneous[index] for index in range(4))
    second = sum(projection[1][index] * homogeneous[index] for index in range(4))
    third = sum(projection[2][index] * homogeneous[index] for index in range(4))
    if third <= 0:
        return None
    u, _, _ = div_round_nearest(first, third)
    v, _, _ = div_round_nearest(second, third)
    depth, _, _ = div_round_nearest(third, camera.rotation_scale)
    return u, v, depth
