"""Small bridge so tests can regenerate a texture PNG with a new fixed seed."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from scripts.generate_samples import make_texture


def regenerate_texture(path: Path, size: int, seed: int) -> None:
    image = make_texture(size, size, seed=seed,
                         control_points=[(size // 4, size // 4, max(2, size // 8)),
                                         (3 * size // 4, size // 2, max(2, size // 10))])
    Image.fromarray(image, mode="RGB").save(path)
