"""为增量编辑演示生成备选视图资产（确定性：复用样本生成器的固定相机布局）。"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mvrwb.sampler import (  # noqa: E402
    BACKGROUND_VALUE, IMAGE_HEIGHT, IMAGE_WIDTH, MARKER_VALUE, _camera_poses,
)
from mvrwb.util import read_json, write_json_atomic  # noqa: E402


def main(sample_dir: str) -> None:
    rng = np.random.default_rng(777)
    # 备选 view_0002：整体亮度提升且标记轻微位移，真实改变图像字节指纹
    image = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), BACKGROUND_VALUE + 10, dtype=np.uint8)
    for _ in range(18):
        px = int(rng.integers(20, IMAGE_WIDTH - 20))
        py = int(rng.integers(20, IMAGE_HEIGHT - 20))
        image[py - 3:py + 4, px - 3:px + 4] = MARKER_VALUE
    Image.fromarray(image, mode="L").save(
        os.path.join(sample_dir, "alt_view_0002.png"), format="PNG"
    )

    # 备选 view_0003 相机：平移量微调，保持旋转正交与内参可逆
    camera = read_json(os.path.join(sample_dir, "views", "view_0003.cam.json"))
    camera["translation"] = [value + 0.05 for value in camera["translation"]]
    write_json_atomic(os.path.join(sample_dir, "alt_view_0003.cam.json"), camera)

    # 额外第七视图
    extras = os.path.join(sample_dir, "extras")
    os.makedirs(extras, exist_ok=True)
    eye, rotation, translation = _camera_poses(7, radius=5.5, height=2.2)[-1]
    focal = 360.0
    image7 = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), BACKGROUND_VALUE, dtype=np.uint8)
    for _ in range(16):
        px = int(rng.integers(18, IMAGE_WIDTH - 18))
        py = int(rng.integers(18, IMAGE_HEIGHT - 18))
        image7[py - 3:py + 4, px - 3:px + 4] = MARKER_VALUE
    Image.fromarray(image7, mode="L").save(
        os.path.join(extras, "view_0007.png"), format="PNG"
    )
    write_json_atomic(
        os.path.join(extras, "view_0007.cam.json"),
        {
            "view_id": 7,
            "intrinsics": [[focal, 0.0, IMAGE_WIDTH / 2.0],
                           [0.0, focal, IMAGE_HEIGHT / 2.0], [0.0, 0.0, 1.0]],
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
            "width": IMAGE_WIDTH,
            "height": IMAGE_HEIGHT,
        },
    )
    print("alt assets written")


if __name__ == "__main__":
    main(sys.argv[1])
