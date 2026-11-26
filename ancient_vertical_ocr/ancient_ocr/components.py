"""连通域细定位、碎片合并、定点关键点回归与字模比对（共享比对点数预算）。"""

from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image

from .config import Config
from .fixedpoint import FixedPoint
from .fonts import GLYPH_ORDER, glyph_bitmap, scale_bitmap
from .hashing import cache_key, digest

STAGE = "fine"


@dataclass
class CompareLedger:
    compare_points_consumed: int = 0
    deductions: list[dict] = field(default_factory=list)
    pending: list[dict] = field(default_factory=list)


@dataclass
class Component:
    component_id: str
    pixels: set[tuple[int, int]]
    bbox: dict
    merge_reasons: list[str] = field(default_factory=list)


def _label_components(region_pixels: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    remaining = set(region_pixels)
    labels: list[set[tuple[int, int]]] = []
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        group = {seed}
        while stack:
            x, y = stack.pop()
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if (nx, ny) in remaining:
                    remaining.discard((nx, ny))
                    group.add((nx, ny))
                    stack.append((nx, ny))
        labels.append(group)
    return labels


def _bbox_of(pixels: set[tuple[int, int]]) -> dict:
    xs = [x for x, _ in pixels]
    ys = [y for _, y in pixels]
    return {"x": min(xs), "y": min(ys), "w": max(xs) - min(xs) + 1,
            "h": max(ys) - min(ys) + 1}


def _merge_fragments(parts: list[set[tuple[int, int]]], block_size: int,
                     anchors: list[tuple[int, int]] | None = None
                     ) -> list[Component]:
    """被遮挡分裂的同色碎片合并。

    优先按确定性字块锚点网格把落入同一 24x24 字块邻域的碎片聚合；
    无锚点时退化为纵向 bbox 拼合规则。每条合并都记录依据。
    """
    components = [Component(f"frag-{idx:03d}", set(part), _bbox_of(part))
                  for idx, part in enumerate(parts)]
    if anchors:
        buckets: dict[tuple[int, int], list[Component]] = {}
        for comp in components:
            cx = comp.bbox["x"] + comp.bbox["w"] // 2
            cy = comp.bbox["y"] + comp.bbox["h"] // 2
            best = min(
                anchors,
                key=lambda anchor: (abs(cx - anchor[0]) + abs(cy - anchor[1])))
            if abs(cx - best[0]) <= block_size and abs(cy - best[1]) <= block_size:
                buckets.setdefault(best, []).append(comp)
            else:
                buckets.setdefault((cx, cy), [comp])
        merged: list[Component] = []
        for anchor in sorted(buckets):
            group = sorted(buckets[anchor],
                           key=lambda item: (item.bbox["y"], item.bbox["x"]))
            base = group[0]
            base.component_id = f"blk-{anchor[0]:03d}-{anchor[1]:03d}"
            for other in group[1:]:
                reason = (
                    f"merge {base.component_id} += {other.component_id}: "
                    f"anchor=({anchor[0]},{anchor[1]}) "
                    f"dx={abs(other.bbox['x']-base.bbox['x'])} "
                    f"dy={abs(other.bbox['y']-base.bbox['y'])} same_color_block")
                base.pixels |= other.pixels
                base.merge_reasons.append(reason)
            base.bbox = _bbox_of(base.pixels)
            merged.append(base)
        return merged
    merged = []
    used: set[int] = set()
    for idx, comp in enumerate(components):
        if idx in used:
            continue
        current = comp
        for jdx in range(idx + 1, len(components)):
            if jdx in used:
                continue
            other = components[jdx]
            horizontal_overlap = not (
                current.bbox["x"] + current.bbox["w"] <= other.bbox["x"]
                or other.bbox["x"] + other.bbox["w"] <= current.bbox["x"])
            vertical_gap = max(
                other.bbox["y"] - (current.bbox["y"] + current.bbox["h"]),
                current.bbox["y"] - (other.bbox["y"] + other.bbox["h"]))
            joined_h = max(
                current.bbox["y"] + current.bbox["h"],
                other.bbox["y"] + other.bbox["h"]) - min(current.bbox["y"], other.bbox["y"])
            if horizontal_overlap and 0 <= vertical_gap <= 6 and joined_h <= block_size + 4:
                current.merge_reasons.append(
                    f"merge {current.component_id}+{other.component_id}: gap={vertical_gap}")
                current.pixels |= other.pixels
                current.bbox = _bbox_of(current.pixels)
                used.add(jdx)
        merged.append(current)
    return merged


def _ink_set(image: Image.Image, region: dict | None = None) -> set[tuple[int, int]]:
    grayscale = list(image.convert("L").getdata())
    width = image.size[0]
    ink: set[tuple[int, int]] = set()
    height = image.size[1]
    if region is None:
        for y in range(height):
            for x in range(width):
                if grayscale[y * width + x] < 128:
                    ink.add((x, y))
        return ink
    box = region["bbox"]
    for y in range(box["y"], box["y"] + box["h"]):
        for x in range(box["x"], box["x"] + box["w"]):
            if grayscale[y * width + x] < 128:
                ink.add((x, y))
    return ink


def _block_anchors(image: Image.Image) -> list[tuple[int, int]]:
    """由竖直投影确定栏带，再按固定步长推导字块中心锚点。"""
    grayscale = list(image.convert("L").getdata())
    width, height = image.size
    column_ink = [
        sum(1 for y in range(height) if grayscale[y * width + x] < 128)
        for x in range(width)]
    anchors: list[tuple[int, int]] = []
    in_band = False
    band_start = 0
    for x, value in enumerate(column_ink + [0]):
        if value > 0 and not in_band:
            band_start, in_band = x, True
        elif value == 0 and in_band:
            center_x = (band_start + x - 1) // 2
            for slot in range(20):
                center_y = 24 + slot * 44
                if center_y < height:
                    anchors.append((center_x, center_y))
            in_band = False
    return sorted(set(anchors))


def _filter_noise(components: list[Component], cfg: Config) -> tuple[list[Component], list[dict]]:
    kept: list[Component] = []
    dropped: list[dict] = []
    for comp in components:
        area = len(comp.pixels)
        aspect = max(comp.bbox["w"], comp.bbox["h"]) / max(1, min(comp.bbox["w"], comp.bbox["h"]))
        if area < cfg.min_component_area:
            dropped.append({"component_id": comp.component_id, "reason": "area_below_min",
                            "area": area, "aspect_ratio": round(aspect, 3)})
        elif aspect > cfg.max_aspect_ratio:
            dropped.append({"component_id": comp.component_id, "reason": "aspect_ratio_exceed",
                            "area": area, "aspect_ratio": round(aspect, 3)})
        else:
            kept.append(comp)
    return kept, dropped


def regress_keypoint(component: Component, cfg: Config) -> dict:
    """块内墨迹重心 + 边界扩展，在定点整数域回归，逐点记录舍入方向与可信度。"""
    fp = FixedPoint(cfg.frac_bits)
    count = len(component.pixels)
    sum_x = sum(x for x, _ in component.pixels)
    sum_y = sum(y for _, y in component.pixels)
    mean_x_q = fp.mean(sum_x, count)
    mean_y_q = fp.mean(sum_y, count)
    box = component.bbox
    center_x_q = fp.from_int(box["x"] + box["w"] // 2)
    center_y_q = fp.from_int(box["y"] + box["h"] // 2)
    blend = 3
    kp_x_q = (fp.mul(mean_x_q, fp.from_int(blend)) + center_x_q) >> 2
    kp_y_q = (fp.mul(mean_y_q, fp.from_int(blend)) + center_y_q) >> 2
    kp_x, round_x = fp.round_to_int(kp_x_q)
    kp_y, round_y = fp.round_to_int(kp_y_q)
    fill_ratio_q = fp.div(count * 255, box["w"] * box["h"])
    confidence_q = min(255, fill_ratio_q + fp.from_int(min(3, len(component.merge_reasons))))
    return {
        "component_id": component.component_id,
        "x": kp_x,
        "y": kp_y,
        "round_x": round_x,
        "round_y": round_y,
        "confidence_q": confidence_q,
        "bbox": box,
        "merge_reasons": component.merge_reasons,
        "area": count,
    }


def _template_variants() -> dict[str, list[list[list[int]]]]:
    variants = {}
    for name in GLYPH_ORDER:
        base = scale_bitmap(glyph_bitmap(name), 2)
        variants[name] = [base]
        for shift in (1, 2):
            shifted = [[0] * 24 for _ in range(24)]
            for y in range(24):
                for x in range(24 - shift):
                    shifted[y][x + shift] = base[y][x]
            variants[name].append(shifted)
    return variants


TEMPLATES = _template_variants()


def compare_cost(ink_pixels: int, variant_count: int) -> int:
    return ink_pixels * variant_count


def match_glyph(component: Component, image: Image.Image, cfg: Config,
                ledger: CompareLedger) -> dict | None:
    """字模比对，按 墨点数 * 模板变体数 扣减共享比对点数预算；不足则入待办。"""
    variant_count = min(cfg.template_variants, 3)
    cost = compare_cost(len(component.pixels), variant_count)
    if ledger.compare_points_consumed + cost > cfg.compare_point_budget:
        ledger.pending.append({
            "stage": STAGE,
            "component_id": component.component_id,
            "cost": cost,
            "consumed": 0,
        })
        return None
    grayscale = list(image.convert("L").getdata())
    width = image.size[0]
    box = component.bbox
    observed = {
        (x, y)
        for y in range(box["y"], box["y"] + box["h"])
        for x in range(box["x"], box["x"] + box["w"])
        if grayscale[y * width + x] < 128
    }
    best_name, best_overlap = None, -1
    for name, templates in TEMPLATES.items():
        for variant_index, template in enumerate(templates[:variant_count]):
            overlap = 0
            for ty in range(min(24, box["h"])):
                for tx in range(min(24, box["w"])):
                    if template[ty][tx] and (box["x"] + tx, box["y"] + ty) in observed:
                        overlap += 1
            if overlap > best_overlap:
                best_name, best_overlap = name, overlap
    ledger.compare_points_consumed += cost
    ledger.deductions.append({
        "stage": STAGE,
        "unit": f"{component.component_id}:{best_name}",
        "budget": "compare_point",
        "cost": cost,
        "remaining": cfg.compare_point_budget - ledger.compare_points_consumed,
        "variants": variant_count,
    })
    return {"component_id": component.component_id, "glyph": best_name,
            "overlap": best_overlap, "cost": cost}


def fine_cache_key(sample, cfg: Config) -> str:
    return cache_key(
        STAGE,
        ("coarse", sample.artifacts["coarse"]["cache_key"]),
        ("image", sample.image_fingerprint),
        ("quant", cfg.frac_bits, cfg.quant_level),
        ("keypoint_version", sample.active_keypoint_fingerprint),
    )


def run_fine_stage(image: Image.Image, regions: list[dict], cfg: Config,
                   ledger: CompareLedger) -> dict:
    components_out: list[dict] = []
    regressions: list[dict] = []
    matches: list[dict] = []
    dropped: list[dict] = []
    ink = _ink_set(image)
    raw_parts = _label_components(ink)
    anchors = _block_anchors(image)
    merged = _merge_fragments(raw_parts, 24, anchors)
    kept, dropped = _filter_noise(merged, cfg)
    region_boxes = [region["bbox"] for region in regions]

    def covering_region(box: dict) -> int:
        for idx, region_box in enumerate(region_boxes):
            if not (box["x"] + box["w"] <= region_box["x"]
                    or region_box["x"] + region_box["w"] <= box["x"]
                    or box["y"] + box["h"] <= region_box["y"]
                    or region_box["y"] + region_box["h"] <= box["y"]):
                return idx
        return -1

    for comp in sorted(kept, key=lambda item: (item.bbox["x"], item.bbox["y"])):
        region_id = covering_region(comp.bbox)
        components_out.append({
            "component_id": comp.component_id,
            "region_id": region_id,
            "bbox": comp.bbox,
            "area": len(comp.pixels),
            "merge_reasons": comp.merge_reasons,
        })
        regressions.append(regress_keypoint(comp, cfg))
        match = match_glyph(comp, image, cfg, ledger)
        if match is not None:
            matches.append(match)
    return {
        "components": components_out,
        "regressions": regressions,
        "matches": matches,
        "dropped": dropped,
        "fingerprint": digest([components_out, regressions, matches, dropped]),
    }
