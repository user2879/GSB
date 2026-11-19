"""按固定种子生成本地确定性样本。

产出（均位于 samples/ 下，可重复运行且字节级一致）：
* vocab.json              受限词表（伪词 + 类别词）
* prompts.json            提示词样本集
* noise/noise_<steps>.npy 预置噪声表（多个步数控件档位）
* images/init_<tier>.npy  图生视频初始图（HxWx3, uint8, 0-255）
* manifest.json           样本清单与生成参数
"""
from __future__ import annotations

import json
import os

import numpy as np

from vbench.config import dump_json
from vbench.hashing import fingerprint
from vbench.stages import RESOLUTION_TIERS

MASTER_SEED = 20261120
SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")

PROMPT_TEMPLATES = [
    "a calm ocean at dawn with soft light",
    "neon city rain street reflections night",
    "mountain forest mist sunrise drone view",
    "desert dunes wind golden hour shadows",
    "space nebula stars slow cosmic drift",
    "a calm ocean under stars with gentle waves",
    "neon city market crowd warm lantern light",
]


def build_vocab(rng: np.random.Generator) -> list[str]:
    pseudo = [
        f"tok{idx:03d}{chr(ord('a') + (idx % 26))}"
        for idx in range(48)
    ]
    category = [
        "ocean", "dawn", "light", "city", "rain", "night", "mountain",
        "forest", "mist", "sunrise", "desert", "dunes", "wind", "space",
        "nebula", "stars", "drift", "calm", "neon", "street", "view",
        "golden", "hour", "shadows", "cosmic", "slow", "soft", "reflections",
    ]
    vocab = pseudo + category
    # 固定顺序并去重，词表本身受主种子指纹约束。
    seen: dict[str, None] = {}
    for word in vocab:
        seen.setdefault(word, None)
    return list(seen.keys())


def main() -> None:
    rng = np.random.default_rng(MASTER_SEED)
    os.makedirs(os.path.join(SAMPLE_DIR, "noise"), exist_ok=True)
    os.makedirs(os.path.join(SAMPLE_DIR, "images"), exist_ok=True)

    vocab = build_vocab(rng)
    dump_json(vocab, os.path.join(SAMPLE_DIR, "vocab.json"))

    prompts = [
        {
            "id": f"prompt_{index:02d}",
            "text": template,
            "tokens": template.split(),
            "prompt_fingerprint": fingerprint(template),
        }
        for index, template in enumerate(PROMPT_TEMPLATES)
    ]
    dump_json(prompts, os.path.join(SAMPLE_DIR, "prompts.json"))

    noise_files = []
    for steps in (4, 6, 8, 12):
        table_rng = np.random.default_rng(
            np.random.SeedSequence([MASTER_SEED, steps])
        )
        # 噪声强度随步序缓慢衰减，全部为有限正数。
        table = table_rng.uniform(0.02, 0.10, size=steps) * np.linspace(
            1.0, 0.4, steps
        )
        path = os.path.join(SAMPLE_DIR, "noise", f"noise_{steps}.npy")
        np.save(path, table.astype(np.float64))
        noise_files.append(
            {
                "steps": steps,
                "path": os.path.relpath(path, SAMPLE_DIR),
                "fingerprint": fingerprint(table),
            }
        )

    image_files = []
    for tier, (height, width) in RESOLUTION_TIERS.items():
        image_rng = np.random.default_rng(
            np.random.SeedSequence([MASTER_SEED, height, width])
        )
        gradient = np.linspace(0.0, 1.0, width, dtype=np.float64)
        base = np.zeros((height, width, 3), dtype=np.float64)
        base[:, :, 0] = gradient[None, :]
        base[:, :, 1] = np.linspace(0.0, 1.0, height)[:, None]
        base[:, :, 2] = 0.5
        noise = image_rng.uniform(-0.08, 0.08, size=(height, width, 3))
        pixels = np.clip(np.rint((base + noise) * 255.0), 0, 255).astype(np.uint8)
        path = os.path.join(SAMPLE_DIR, "images", f"init_{tier}.npy")
        np.save(path, pixels)
        image_files.append(
            {
                "tier": tier,
                "height": height,
                "width": width,
                "path": os.path.relpath(path, SAMPLE_DIR),
                "fingerprint": fingerprint(pixels),
            }
        )

    manifest = {
        "master_seed": MASTER_SEED,
        "numpy_version": np.__version__,
        "vocab_size": len(vocab),
        "prompt_count": len(prompts),
        "noise_tables": noise_files,
        "init_images": image_files,
    }
    dump_json(manifest, os.path.join(SAMPLE_DIR, "manifest.json"))
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
