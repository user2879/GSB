"""滑窗粗定位确定性内核：定点特征、指纹投影、扫描行预算与待扫续扫。"""

from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image

from .config import Config
from .errors import E_SCAN_BUDGET_EXHAUSTED, BudgetExhaustedError
from .fixedpoint import FixedPoint
from .hashing import cache_key, digest

STAGE = "coarse"


@dataclass
class WindowUnit:
    sample_id: str
    row: int
    col: int
    x: int
    y: int
    fingerprint: str
    consumed_scan_rows: int = 0
    done: bool = False

    @property
    def sort_key(self) -> str:
        return f"{self.sample_id}|{STAGE}|r{self.row:03d}c{self.col:03d}"


@dataclass
class WindowScore:
    row: int
    col: int
    x: int
    y: int
    fingerprint: str
    ink_density_q: int
    valley_q: int
    gradient_q: int
    score_q: int

    def to_dict(self) -> dict:
        return {
            "row": self.row, "col": self.col, "x": self.x, "y": self.y,
            "fingerprint": self.fingerprint,
            "ink_density_q": self.ink_density_q,
            "valley_q": self.valley_q,
            "gradient_q": self.gradient_q,
            "score_q": self.score_q,
        }


@dataclass
class ScanLedger:
    scan_rows_consumed: int = 0
    deductions: list[dict] = field(default_factory=list)
    pending: list[WindowUnit] = field(default_factory=list)
    completed: list[WindowUnit] = field(default_factory=list)


def enumerate_windows(sample_id: str, cfg: Config) -> list[WindowUnit]:
    units: list[WindowUnit] = []
    rows = (480 - cfg.window_size) // cfg.window_stride + 1
    cols = (320 - cfg.window_size) // cfg.window_stride + 1
    for row in range(rows):
        y = row * cfg.window_stride
        for col in range(cols):
            x = col * cfg.window_stride
            units.append(WindowUnit(sample_id, row, col, x, y, fingerprint=""))
    return units


def _window_pixels(image: Image.Image, unit: WindowUnit, size: int) -> list[list[int]]:
    grayscale = list(image.convert("L").getdata())
    width = image.size[0]
    pixels: list[list[int]] = []
    for y in range(unit.y, unit.y + size):
        line = grayscale[y * width + unit.x: y * width + unit.x + size]
        pixels.append(list(line))
    return pixels


def _projection_weights(fingerprint: str) -> tuple[int, int, int]:
    """窗口指纹派生的固定投影权重（定点 q4，和为 16）。"""
    head = int(fingerprint[:8], 16)
    first = 4 + head % 7
    second = 3 + (head >> 8) % 6
    third = 16 - first - second
    return first, second, third


def score_window(pixels: list[list[int]], fp: FixedPoint) -> tuple[int, int, int, int, str]:
    size = len(pixels)
    flat = [cell for row in pixels for cell in row]
    ink = sum(1 for cell in flat if cell < 128)
    area = size * size
    ink_density_q = fp.div(ink * 255, area)

    column_ink = [sum(1 for y in range(size) if pixels[y][x] < 128) for x in range(size)]
    mean_ink = sum(column_ink) // size
    valley_depth = sum(max(0, mean_ink - value) for value in column_ink)
    valley_q = fp.div(valley_depth * 255, area)

    horizontal = 0
    vertical = 0
    for y in range(size):
        for x in range(size):
            if pixels[y][x] >= 128:
                continue
            if x + 1 < size and pixels[y][x + 1] < 128:
                horizontal += 1
            if y + 1 < size and pixels[y + 1][x] < 128:
                vertical += 1
    gradient_q = fp.div(abs(vertical - horizontal) * 255, area)

    fingerprint = digest(flat)
    w1, w2, w3 = _projection_weights(fingerprint)
    score_q = (fp.mul(fp.from_int(w1), ink_density_q)
               + fp.mul(fp.from_int(w2), valley_q)
               + fp.mul(fp.from_int(w3), gradient_q))
    return ink_density_q, valley_q, gradient_q, score_q, fingerprint


def coarse_cache_key(sample: "object", cfg: Config) -> str:
    return cache_key(
        STAGE,
        ("image", sample.image_fingerprint),
        ("params", cfg.param_fingerprint()),
        ("quant", cfg.frac_bits, cfg.quant_level),
    )


def run_coarse_scan(image: Image.Image, units: list[WindowUnit], cfg: Config,
                    ledger: ScanLedger) -> tuple[list[WindowScore], list[WindowUnit]]:
    """按行/列升序消费扫描行预算；预算耗尽即入待扫队列，保留指纹断点。"""
    fp = FixedPoint(cfg.frac_bits)
    scores: list[WindowScore] = []
    ordered = sorted(units, key=lambda item: (item.row, item.col, item.sample_id))
    still_pending: list[WindowUnit] = []
    for unit in ordered:
        if unit.done:
            continue
        need = cfg.window_size - unit.consumed_scan_rows
        if ledger.scan_rows_consumed + need > cfg.scan_row_budget:
            still_pending.append(unit)
            ledger.pending = still_pending
            continue
        pixels = _window_pixels(image, unit, cfg.window_size)
        density, valley, gradient, score, fingerprint = score_window(pixels, fp)
        unit.fingerprint = fingerprint
        unit.consumed_scan_rows += need
        unit.done = True
        ledger.scan_rows_consumed += need
        ledger.deductions.append({
            "stage": STAGE,
            "unit": unit.sort_key,
            "budget": "scan_row",
            "cost": need,
            "remaining": cfg.scan_row_budget - ledger.scan_rows_consumed,
        })
        ledger.completed.append(unit)
        if score >= cfg.candidate_score_threshold:
            scores.append(WindowScore(
                row=unit.row, col=unit.col, x=unit.x, y=unit.y,
                fingerprint=fingerprint, ink_density_q=density, valley_q=valley,
                gradient_q=gradient, score_q=score))
    ledger.pending = still_pending
    return scores, still_pending


def merge_candidates(scores: list[WindowScore], cfg: Config) -> list[dict]:
    """对得分窗口做固定顺序的相邻合并，得到候选区域。"""
    chosen = sorted(scores, key=lambda item: (item.row, item.col))
    merged: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    for score in chosen:
        if (score.row, score.col) in occupied:
            continue
        x0, y0 = score.x, score.y
        x1, y1 = x0 + cfg.window_size, y0 + cfg.window_size
        members = [(score.row, score.col)]
        for other in chosen:
            if other is score or (other.row, other.col) in occupied:
                continue
            ox0, oy0 = other.x, other.y
            ox1, oy1 = ox0 + cfg.window_size, oy0 + cfg.window_size
            if not (ox1 < x0 or ox0 > x1 or oy1 < y0 or oy0 > y1):
                x0, y0 = min(x0, ox0), min(y0, oy0)
                x1, y1 = max(x1, ox1), max(y1, oy1)
                members.append((other.row, other.col))
        for member in members:
            occupied.add(member)
        merged.append({
            "region_id": len(merged),
            "bbox": {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0},
            "members": [f"r{r}c{c}" for r, c in sorted(members)],
            "max_score_q": max(
                item.score_q for item in chosen
                if (item.row, item.col) in set(members)),
        })
    return merged


def assert_budget_kind(name: str) -> None:
    if name != "scan_row":
        raise BudgetExhaustedError(E_SCAN_BUDGET_EXHAUSTED, 0, 0)
