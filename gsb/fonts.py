"""Built-in bitmap glyph library and deterministic template variants."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from gsb.constants import E_TEMPLATE_VARIANT_DIMENSION, TEMPLATE_VERSION
from gsb.errors import ValidationIssue
from gsb.util import sha256_obj

TEMPLATE_WIDTH = 12
TEMPLATE_HEIGHT = 12
VARIANT_NAMES = ("base", "bold", "thin")
VARIANT_WEIGHTS = (64, 2, 1)

BASE_GLYPHS: dict[str, tuple[str, ...]] = {
    "古": (
        "............",
        "...######...",
        "..#......#..",
        "..#......#..",
        "..########..",
        ".....##.....",
        ".....##.....",
        "...#.....#..",
        "..#.......#.",
        "..#.......#.",
        "..#.......#.",
        "............",
    ),
    "籍": (
        "............",
        "..##.###.##.",
        "..##.###.##.",
        ".#..#..#..#.",
        ".#########..",
        ".#..#..#..#.",
        ".#########..",
        ".....##.....",
        "..#######...",
        "..#..#..#...",
        "..#######...",
        "............",
    ),
    "数": (
        "............",
        "...#..#.....",
        "..######....",
        "...#..#.....",
        ".########...",
        "...#.#......",
        "..##.##.....",
        ".#...#.#....",
        "....#..#....",
        "...#....#...",
        "..#......#..",
        "............",
    ),
    "字": (
        "............",
        ".....##.....",
        "..########..",
        "..#......#..",
        "............",
        "...######...",
        "..........#.",
        ".....##.....",
        ".....##.....",
        ".....##.....",
        "....###.....",
        "............",
    ),
    "化": (
        "............",
        "...#....#...",
        "..#....#....",
        "..#...#.....",
        ".#....#.....",
        ".#....#.....",
        ".#.....#....",
        ".#......#...",
        ".#.......#..",
        ".#........#.",
        ".#..........",
        "............",
    ),
    "首": (
        "............",
        ".....##.....",
        "..########..",
        "..#......#..",
        "............",
        "..########..",
        "..#......#..",
        "..########..",
        "..#......#..",
        "..########..",
        "............",
        "............",
    ),
    "版": (
        "............",
        "..#.....#...",
        "..#....#....",
        "..#####.....",
        "..#...#.....",
        "..#....#....",
        "..#.....#...",
        "..###..#.#..",
        "..#...#..#..",
        "..#..#...#..",
        "..#.#.....#.",
        "............",
    ),
    "竖": (
        "............",
        "...#...#....",
        "..#.....#...",
        ".#.......#..",
        "..########..",
        ".....##.....",
        "..########..",
        "..#......#..",
        "..########..",
        ".....##.....",
        "##########..",
        "............",
    ),
    "排": (
        "............",
        "...#.#.#....",
        "...#.#.#....",
        "...#.#.#....",
        ".#######....",
        "...#.#.#....",
        "...#.#.#....",
        ".#######....",
        "...#.#.#....",
        "...#.#.#....",
        "...#.#.#....",
        "............",
    ),
    "校": (
        "............",
        "...#....#...",
        "...#..##....",
        ".#####.###..",
        "...#...#....",
        "..###..#....",
        ".#.#...#....",
        "...#....#...",
        "...#.....#..",
        "...#......#.",
        "...#.....#..",
        "............",
    ),
    "己": (
        "............",
        "..########..",
        ".........#..",
        ".........#..",
        "..########..",
        "..#.........",
        "..#.........",
        "..#.........",
        "..########..",
        "............",
        "............",
        "............",
    ),
    "已": (
        "............",
        "...#######..",
        ".........#..",
        ".........#..",
        "..########..",
        "..#.........",
        "..#.........",
        "..#.........",
        "..########..",
        "............",
        "............",
        "............",
    ),
    "未": (
        "............",
        ".....##.....",
        "..########..",
        ".....##.....",
        ".....##.....",
        "##########..",
        ".....##.....",
        "....####....",
        "...#..#.#...",
        "..#...#..#..",
        ".#....#...#.",
        "............",
    ),
    "末": (
        "............",
        ".....##.....",
        ".....##.....",
        "##########..",
        ".....##.....",
        ".....##.....",
        "....####....",
        "...#..#.#...",
        "..#...#..#..",
        ".#....#...#.",
        "............",
        "............",
    ),
    "日": (
        "............",
        "..########..",
        "..#......#..",
        "..#......#..",
        "..########..",
        "..#......#..",
        "..#......#..",
        "..#......#..",
        "..########..",
        "............",
        "............",
        "............",
    ),
    "曰": (
        "............",
        "..########..",
        "..#......#..",
        "..#......#..",
        "..#......#..",
        "..########..",
        "..#......#..",
        "..#......#..",
        "..########..",
        "............",
        "............",
        "............",
    ),
}


def _normalize_rows(rows: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(row[:TEMPLATE_WIDTH].ljust(TEMPLATE_WIDTH, ".") for row in rows)
    if len(normalized) < TEMPLATE_HEIGHT:
        normalized += ("." * TEMPLATE_WIDTH,) * (TEMPLATE_HEIGHT - len(normalized))
    return normalized[:TEMPLATE_HEIGHT]


def _shift(rows: tuple[str, ...], dx: int, dy: int) -> tuple[str, ...]:
    canvas = [["." for _ in range(TEMPLATE_WIDTH)] for _ in range(TEMPLATE_HEIGHT)]
    for y, row in enumerate(rows):
        for x, value in enumerate(row):
            nx = x + dx
            ny = y + dy
            if value == "#" and 0 <= nx < TEMPLATE_WIDTH and 0 <= ny < TEMPLATE_HEIGHT:
                canvas[ny][nx] = "#"
    return tuple("".join(row) for row in canvas)


def _dilate(rows: tuple[str, ...]) -> tuple[str, ...]:
    canvas = [list(row) for row in rows]
    for y, row in enumerate(rows):
        for x, value in enumerate(row):
            if value != "#":
                continue
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < TEMPLATE_WIDTH and 0 <= ny < TEMPLATE_HEIGHT:
                        canvas[ny][nx] = "#"
    return tuple("".join(row) for row in canvas)


def _thin(rows: tuple[str, ...]) -> tuple[str, ...]:
    canvas = [list(row) for row in rows]
    for y, row in enumerate(rows):
        for x, value in enumerate(row):
            if value != "#":
                continue
            neighbors = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < TEMPLATE_WIDTH and 0 <= ny < TEMPLATE_HEIGHT and rows[ny][nx] == "#":
                        neighbors += 1
            if neighbors < 2:
                canvas[y][x] = "."
    return tuple("".join(row) for row in canvas)


def make_variant_rows(rows: tuple[str, ...], name: str) -> tuple[str, ...]:
    if name == "base":
        return rows
    if name == "bold":
        return _dilate(rows)
    if name == "thin":
        return _thin(rows)
    raise ValueError(f"unknown variant: {name}")


def template_manifest_payload() -> dict[str, Any]:
    glyphs: list[dict[str, Any]] = []
    for char in sorted(BASE_GLYPHS):
        rows = _normalize_rows(BASE_GLYPHS[char])
        variants = []
        for name in VARIANT_NAMES:
            variant_rows = make_variant_rows(rows, name)
            variants.append(
                {
                    "name": name,
                    "width": TEMPLATE_WIDTH,
                    "height": TEMPLATE_HEIGHT,
                    "rows": list(variant_rows),
                }
            )
        glyphs.append(
            {
                "char": char,
                "code_point": ord(char),
                "template_id": f"U+{ord(char):04X}",
                "variants": variants,
            }
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "template_version": TEMPLATE_VERSION,
        "width": TEMPLATE_WIDTH,
        "height": TEMPLATE_HEIGHT,
        "variant_names": list(VARIANT_NAMES),
        "variant_weights": list(VARIANT_WEIGHTS),
        "glyphs": glyphs,
    }
    payload["content_sha256"] = sha256_obj(payload)
    return payload


@dataclass(frozen=True)
class TemplateVariant:
    name: str
    width: int
    height: int
    bits: tuple[int, ...]

    def ink_count(self) -> int:
        return sum(self.bits)


@dataclass(frozen=True)
class GlyphTemplate:
    char: str
    code_point: int
    template_id: str
    variants: tuple[TemplateVariant, ...]


@dataclass(frozen=True)
class TemplateLibrary:
    template_version: str
    width: int
    height: int
    weights: tuple[int, ...]
    glyphs: tuple[GlyphTemplate, ...]
    content_fingerprint: str

    def by_code_point(self) -> dict[int, GlyphTemplate]:
        return {glyph.code_point: glyph for glyph in self.glyphs}


def load_templates(
    path: str | Path,
    *,
    expected_version: str,
    expected_width: int,
    expected_height: int,
    expected_weights: tuple[int, ...],
) -> tuple[TemplateLibrary | None, list[ValidationIssue]]:
    template_path = Path(path)
    issues: list[ValidationIssue] = []
    try:
        raw = json.loads(template_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, [
            ValidationIssue(
                code=E_TEMPLATE_VARIANT_DIMENSION,
                message="template manifest not found",
                source=str(template_path),
            )
        ]
    except (OSError, json.JSONDecodeError) as exc:
        return None, [
            ValidationIssue(
                code=E_TEMPLATE_VARIANT_DIMENSION,
                message=f"template manifest cannot be loaded: {exc}",
                source=str(template_path),
            )
        ]
    if not isinstance(raw, dict):
        return None, [
            ValidationIssue(
                code=E_TEMPLATE_VARIANT_DIMENSION,
                message="template manifest must be an object",
                source=str(template_path),
            )
        ]
    version = raw.get("template_version")
    if version != expected_version:
        issues.append(
            ValidationIssue(
                code=E_TEMPLATE_VARIANT_DIMENSION,
                message="template_version does not match configuration",
                source=str(template_path),
            )
        )
    width = raw.get("width")
    height = raw.get("height")
    if width != expected_width or height != expected_height:
        issues.append(
            ValidationIssue(
                code=E_TEMPLATE_VARIANT_DIMENSION,
                message="template dimensions do not match configuration",
                source=str(template_path),
            )
        )
    weights_raw = raw.get("variant_weights")
    if (
        not isinstance(weights_raw, list)
        or tuple(weights_raw) != expected_weights
        or len(weights_raw) != len(raw.get("variant_names", []))
    ):
        issues.append(
            ValidationIssue(
                code=E_TEMPLATE_VARIANT_DIMENSION,
                message="variant weights do not match configuration",
                source=str(template_path),
            )
        )
    glyphs_raw = raw.get("glyphs")
    glyphs: list[GlyphTemplate] = []
    seen_code_points: set[int] = set()
    if not isinstance(glyphs_raw, list) or not glyphs_raw:
        issues.append(
            ValidationIssue(
                code=E_TEMPLATE_VARIANT_DIMENSION,
                message="template glyphs must be a non-empty list",
                source=str(template_path),
            )
        )
    else:
        for glyph_index, glyph_raw in enumerate(glyphs_raw):
            if not isinstance(glyph_raw, dict):
                issues.append(
                    ValidationIssue(
                        code=E_TEMPLATE_VARIANT_DIMENSION,
                        message=f"glyphs[{glyph_index}] must be an object",
                        source=str(template_path),
                    )
                )
                continue
            char = glyph_raw.get("char")
            code_point = glyph_raw.get("code_point")
            template_id = glyph_raw.get("template_id")
            if (
                not isinstance(char, str)
                or len(char) != 1
                or isinstance(code_point, bool)
                or not isinstance(code_point, int)
                or code_point != ord(char)
                or not isinstance(template_id, str)
            ):
                issues.append(
                    ValidationIssue(
                        code=E_TEMPLATE_VARIANT_DIMENSION,
                        message=f"glyphs[{glyph_index}] identity is invalid",
                        source=str(template_path),
                    )
                )
                continue
            if code_point in seen_code_points:
                issues.append(
                    ValidationIssue(
                        code=E_TEMPLATE_VARIANT_DIMENSION,
                        message=f"duplicate code point {code_point}",
                        source=str(template_path),
                    )
                )
                continue
            seen_code_points.add(code_point)
            variants_raw = glyph_raw.get("variants")
            variants: list[TemplateVariant] = []
            if not isinstance(variants_raw, list) or len(variants_raw) != len(expected_weights):
                issues.append(
                    ValidationIssue(
                        code=E_TEMPLATE_VARIANT_DIMENSION,
                        message=f"{char} variant count does not match weights",
                        source=str(template_path),
                    )
                )
                continue
            for variant_index, variant_raw in enumerate(variants_raw):
                if not isinstance(variant_raw, dict):
                    issues.append(
                        ValidationIssue(
                            code=E_TEMPLATE_VARIANT_DIMENSION,
                            message=f"{char} variant {variant_index} must be an object",
                            source=str(template_path),
                        )
                    )
                    continue
                variant_name = variant_raw.get("name")
                variant_width = variant_raw.get("width")
                variant_height = variant_raw.get("height")
                rows = variant_raw.get("rows")
                if (
                    variant_width != expected_width
                    or variant_height != expected_height
                    or not isinstance(rows, list)
                    or len(rows) != expected_height
                    or any(not isinstance(row, str) or len(row) != expected_width for row in rows)
                    or any(character not in ".#" for row in rows for character in row)
                ):
                    issues.append(
                        ValidationIssue(
                            code=E_TEMPLATE_VARIANT_DIMENSION,
                            message=f"{char} variant {variant_index} dimensions or rows are invalid",
                            source=str(template_path),
                            details={
                                "char": char,
                                "variant": variant_name,
                                "declared": [variant_width, variant_height],
                                "rows": len(rows) if isinstance(rows, list) else None,
                            },
                        )
                    )
                    continue
                bits = tuple(1 if character == "#" else 0 for row in rows for character in row)
                variants.append(
                    TemplateVariant(
                        name=str(variant_name),
                        width=variant_width,
                        height=variant_height,
                        bits=bits,
                    )
                )
            if len(variants) == len(expected_weights):
                glyphs.append(
                    GlyphTemplate(
                        char=char,
                        code_point=code_point,
                        template_id=template_id,
                        variants=tuple(variants),
                    )
                )
    if issues:
        return None, issues
    fingerprint = sha256_obj(
        {
            "version": version,
            "width": width,
            "height": height,
            "weights": weights_raw,
            "glyphs": glyphs_raw,
        }
    )
    return (
        TemplateLibrary(
            template_version=str(version),
            width=int(width),
            height=int(height),
            weights=tuple(int(item) for item in weights_raw),
            glyphs=tuple(sorted(glyphs, key=lambda item: item.code_point)),
            content_fingerprint=fingerprint,
        ),
        [],
    )
