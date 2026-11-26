"""本地确定性合成竖排古籍版面（固定分辨率 PNG + 真值 + 栏序 + 关键点版本）。"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from .fixedpoint import FixedPoint, int_cos_sin
from .fonts import GLYPH_ORDER, glyph_bitmap, scale_bitmap
from .hashing import digest
from .rng import DeterministicRng

PAGE_W = 320
PAGE_H = 480
BLOCK = 24
INK = 0
PAPER = 255

COL_X = (240, 168, 96, 24)
TOP_Y = 12
BLOCK_STEP = 44
BLOCKS_PER_COL = 10
ROTATIONS = (-5, -3, 0, 3, 5)


def _rotate_jitter_block(bitmap24: list[list[int]], rng: DeterministicRng,
                         degrees: int, frac_bits: int) -> tuple[list[list[int]], int]:
    """整数域旋转 + 笔画抖动；返回 24x24 墨块位图与抖动墨点数。"""
    fp = FixedPoint(frac_bits)
    cos_q4, sin_q4 = int_cos_sin(degrees, frac_bits)
    cos_q = fp.div(cos_q4, 10000)
    sin_q = fp.div(sin_q4, 10000)
    half = BLOCK // 2
    canvas = [[0] * BLOCK for _ in range(BLOCK)]
    jitter_count = 0
    for ty in range(BLOCK):
        for tx in range(BLOCK):
            dx = fp.from_int(tx - half)
            dy = fp.from_int(ty - half)
            src_xq = fp.mul(cos_q, dx) - fp.mul(sin_q, dy)
            src_yq = fp.mul(sin_q, dx) + fp.mul(cos_q, dy)
            jitter = (rng.below(3) - 1)
            jitter_count += abs(jitter)
            sx, _ = fp.round_to_int(src_xq + fp.from_int(jitter))
            sy, _ = fp.round_to_int(src_yq)
            sx += half
            sy += half
            if 0 <= sx < BLOCK and 0 <= sy < BLOCK:
                canvas[ty][tx] = bitmap24[sy][sx]
    return canvas, jitter_count


def _apply_occlusion(canvas: list[list[int]], rng: DeterministicRng) -> dict:
    """在字块中部覆盖一条同纸色窄带，制造可合并的同色碎片分裂。"""
    band_top = 9 + rng.below(4)
    band_h = 2 + rng.below(2)
    left = 4 + rng.below(4)
    width = 12 + rng.below(5)
    for y in range(band_top, min(band_top + band_h, BLOCK)):
        for x in range(left, min(left + width, BLOCK)):
            canvas[y][x] = 0
    return {"band_top": band_top, "band_h": band_h, "left": left, "width": width}


def _render_page(seed: int, frac_bits: int):
    rng = DeterministicRng(seed)
    page = [[PAPER] * PAGE_W for _ in range(PAGE_H)]
    blocks: list[dict] = []
    order = 0
    for col_idx, base_x in enumerate(COL_X):
        for slot in range(BLOCKS_PER_COL):
            block_rng = rng.child(col_idx * 100 + slot)
            name = GLYPH_ORDER[block_rng.below(len(GLYPH_ORDER))]
            bitmap = scale_bitmap(glyph_bitmap(name), 2)
            degrees = block_rng.choose(ROTATIONS)
            canvas, jitter_count = _rotate_jitter_block(bitmap, block_rng, degrees, frac_bits)
            occlusion = _apply_occlusion(canvas, block_rng)
            jitter_x = block_rng.below(3) - 1
            jitter_y = block_rng.below(3) - 1
            x0 = base_x + jitter_x
            y0 = TOP_Y + slot * BLOCK_STEP + jitter_y
            ink_pixels = 0
            for y in range(BLOCK):
                for x in range(BLOCK):
                    if canvas[y][x]:
                        page[y0 + y][x0 + x] = INK
                        ink_pixels += 1
            blocks.append({
                "block_index": order,
                "column_index": col_idx,
                "glyph": name,
                "rotation_deg": degrees,
                "bbox": {"x": x0, "y": y0, "w": BLOCK, "h": BLOCK},
                "keypoint": {"x": x0 + BLOCK // 2, "y": y0 + BLOCK // 2},
                "reading_order": order,
                "ink_pixels": ink_pixels,
                "jitter_pixels": jitter_count,
                "occlusion": occlusion,
            })
            order += 1
    return page, blocks


def _columns_declaration() -> dict:
    columns = []
    for idx, base_x in enumerate(COL_X):
        columns.append({
            "column_index": idx,
            "x_interval": {"x_min": base_x - 4, "x_max": base_x + BLOCK + 3},
        })
    candidates = [
        {"path": [0, 1, 2, 3], "priority": 1, "cost": 1},
        {"path": [0, 1, 3, 2], "priority": 2, "cost": 7},
    ]
    adjacency = [
        {"from": 0, "to": 1},
        {"from": 1, "to": 2},
        {"from": 2, "to": 3},
    ]
    return {"columns": columns, "candidate_paths": candidates, "adjacency": adjacency}


def _annotator_versions(blocks: list[dict], seed: int) -> dict:
    rng = DeterministicRng(seed ^ 0x55AA55AA)
    v1 = []
    v2 = []
    for block in blocks:
        kp = block["keypoint"]
        v1.append({
            "block_index": block["block_index"],
            "column_index": block["column_index"],
            "x": kp["x"],
            "y": kp["y"],
            "confidence_q": 200 + rng.below(40),
        })
        v2.append({
            "block_index": block["block_index"],
            "column_index": block["column_index"],
            "x": kp["x"] + (1 if block["block_index"] % 3 == 0 else 0),
            "y": kp["y"] + (1 if block["block_index"] % 4 == 0 else 0),
            "confidence_q": 180 + rng.below(60),
        })
    return {
        "active_version": "annot-v1",
        "versions": {
            "annot-v1": {"fingerprint": digest(v1), "points": v1},
            "annot-v2": {"fingerprint": digest(v2), "points": v2},
        },
    }


def _write_png(page: list[list[int]], path: Path) -> str:
    img = Image.new("L", (PAGE_W, PAGE_H), PAPER)
    img.putdata([cell for row in page for cell in row])
    img.save(path, format="PNG", optimize=False)
    return digest(path.read_bytes())


def generate_sample(root: Path, sample_id: str, seed: int, frac_bits: int = 4) -> dict:
    sample_dir = root / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    page, blocks = _render_page(seed, frac_bits)
    png_path = sample_dir / "page.png"
    image_fingerprint = _write_png(page, png_path)
    truth = {
        "sample_id": sample_id,
        "seed": seed,
        "page": {"width": PAGE_W, "height": PAGE_H, "mode": "L"},
        "image_fingerprint": image_fingerprint,
        "blocks": blocks,
    }
    (sample_dir / "truth.json").write_text(
        json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")
    declaration = _columns_declaration()
    declaration["sample_id"] = sample_id
    (sample_dir / "columns.json").write_text(
        json.dumps(declaration, ensure_ascii=False, indent=2), encoding="utf-8")
    versions = _annotator_versions(blocks, seed)
    versions["sample_id"] = sample_id
    (sample_dir / "keypoint_versions.json").write_text(
        json.dumps(versions, ensure_ascii=False, indent=2), encoding="utf-8")
    return truth


def inject_bad_samples(root: Path) -> list[str]:
    """独立注入三类坏样本，用于证明 E_SAMPLE_INVALID 隔离。"""
    created: list[str] = []
    good = generate_sample(root, "bad_oob", 9001)
    blocks = good["blocks"]
    blocks[0]["keypoint"]["x"] = PAGE_W + 20
    (root / "bad_oob" / "truth.json").write_text(
        json.dumps(good, ensure_ascii=False, indent=2), encoding="utf-8")
    created.append("bad_oob")

    generate_sample(root, "bad_cycle", 9002)
    cycle_path = root / "bad_cycle" / "columns.json"
    decl = json.loads(cycle_path.read_text(encoding="utf-8"))
    decl["adjacency"].append({"from": 3, "to": 0})
    cycle_path.write_text(json.dumps(decl, ensure_ascii=False, indent=2), encoding="utf-8")
    created.append("bad_cycle")

    generate_sample(root, "bad_header", 9003)
    png = root / "bad_header" / "page.png"
    img = Image.open(png)
    img.resize((PAGE_W - 16, PAGE_H)).save(png, format="PNG")
    created.append("bad_header")
    return created


def build_default_corpus(root: Path, frac_bits: int = 4) -> list[str]:
    ids = ["page_a", "page_b", "page_c"]
    for idx, sample_id in enumerate(ids):
        generate_sample(root, sample_id, 1000 + idx * 37, frac_bits)
    return ids
