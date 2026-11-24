"""样本加载与字段校验层。

样本目录中每个样本由三个同名文件组成：
``<id>.png`` 图像、``<id>.assertion.json`` 文本断言、``<id>.regions.json`` 区域真值。
任何非法项都产出 ``E_SAMPLE_INVALID`` 失败记录并跳过该样本，其余样本继续处理。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from .errors import E_SAMPLE_INVALID, Failure, make_failure
from .pngio import PngError, read_png

CONTROLLED_ENTITIES = ("circle", "square", "triangle", "block", "line")
CONTROLLED_ATTRIBUTES = ("color", "size", "texture", "position", "orientation")


@dataclass
class SampleBundle:
    sample_id: str
    image: "object"          # PngImage
    assertion: dict[str, Any]
    regions: list[dict[str, Any]]
    png_path: str
    assertion_path: str
    regions_path: str


def _load_json(path: str, sid: str, kind: str, failures: list[Failure]) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        failures.append(make_failure(E_SAMPLE_INVALID, "sample", sid, kind, f"缺少文件 {os.path.basename(path)}"))
    except json.JSONDecodeError as exc:
        failures.append(make_failure(E_SAMPLE_INVALID, "sample", sid, kind, f"JSON 解析失败: {exc}"))
    except OSError as exc:
        failures.append(make_failure(E_SAMPLE_INVALID, "sample", sid, kind, f"文件读取失败: {exc}"))
    return None


def _validate_regions(regions: Any, width: int, height: int, sid: str) -> list[Failure]:
    failures: list[Failure] = []
    if not isinstance(regions, list) or not regions:
        failures.append(make_failure(E_SAMPLE_INVALID, "sample", sid, "regions", "区域真值必须是非空数组"))
        return failures
    for idx, region in enumerate(regions):
        field_name = f"regions[{idx}]"
        if not isinstance(region, dict):
            failures.append(make_failure(E_SAMPLE_INVALID, "sample", sid, field_name, "区域必须是对象"))
            continue
        for key in ("x", "y", "w", "h"):
            value = region.get(key)
            if not (isinstance(value, int) and not isinstance(value, bool)):
                failures.append(
                    make_failure(E_SAMPLE_INVALID, "sample", sid, f"{field_name}.{key}",
                                 f"区域坐标 {key} 必须是整数，实际 {value!r}")
                )
        if all(isinstance(region.get(k), int) and not isinstance(region.get(k), bool)
               for k in ("x", "y", "w", "h")):
            x, y, w, h = region["x"], region["y"], region["w"], region["h"]
            if w <= 0 or h <= 0:
                failures.append(
                    make_failure(E_SAMPLE_INVALID, "sample", sid, field_name, "区域宽高必须为正整数")
                )
            if not (0 <= x and 0 <= y and x + w <= width and y + h <= height):
                failures.append(
                    make_failure(E_SAMPLE_INVALID, "sample", sid, field_name,
                                 f"区域矩形 ({x},{y},{w},{h}) 落在 {width}x{height} 图像之外")
                )
        label = region.get("label")
        if not isinstance(label, str) or not label:
            failures.append(
                make_failure(E_SAMPLE_INVALID, "sample", sid, f"{field_name}.label", "区域 label 必须是非空字符串")
            )
    return failures


def _validate_assertion(assertion: Any, sid: str) -> list[Failure]:
    failures: list[Failure] = []
    if not isinstance(assertion, dict):
        return [make_failure(E_SAMPLE_INVALID, "sample", sid, "assertion", "文本断言必须是 JSON 对象")]
    if assertion.get("sample_id") != sid:
        failures.append(
            make_failure(E_SAMPLE_INVALID, "sample", sid, "assertion.sample_id",
                         f"断言 sample_id={assertion.get('sample_id')!r} 与文件标识 {sid!r} 不一致")
        )
    entities = assertion.get("entities")
    if not isinstance(entities, list) or not entities:
        failures.append(make_failure(E_SAMPLE_INVALID, "sample", sid, "assertion.entities", "实体列表必须是非空数组"))
    else:
        for idx, entity in enumerate(entities):
            fname = f"assertion.entities[{idx}]"
            if not isinstance(entity, dict):
                failures.append(make_failure(E_SAMPLE_INVALID, "sample", sid, fname, "实体必须是对象"))
                continue
            etype = entity.get("type")
            if etype not in CONTROLLED_ENTITIES:
                failures.append(
                    make_failure(E_SAMPLE_INVALID, "sample", sid, f"{fname}.type",
                                 f"实体类型 {etype!r} 不在受控词表 {list(CONTROLLED_ENTITIES)} 内")
                )
            if not isinstance(entity.get("id"), str) or not entity["id"]:
                failures.append(
                    make_failure(E_SAMPLE_INVALID, "sample", sid, f"{fname}.id", "实体 id 必须是非空字符串")
                )
            attrs = entity.get("attributes")
            if not isinstance(attrs, dict) or not attrs:
                failures.append(
                    make_failure(E_SAMPLE_INVALID, "sample", sid, f"{fname}.attributes", "属性字段必须是非空对象")
                )
            else:
                for key, value in attrs.items():
                    if key not in CONTROLLED_ATTRIBUTES:
                        failures.append(
                            make_failure(E_SAMPLE_INVALID, "sample", sid, f"{fname}.attributes.{key}",
                                         f"属性 {key!r} 不在受控词表 {list(CONTROLLED_ATTRIBUTES)} 内")
                        )
                    if not isinstance(value, str) or not value:
                        failures.append(
                            make_failure(E_SAMPLE_INVALID, "sample", sid, f"{fname}.attributes.{key}",
                                         "属性取值必须是非空字符串")
                        )
    return failures


def load_sample(sample_dir: str, sid: str, expected_size: tuple[int, int] | None) -> tuple[SampleBundle | None, list[Failure]]:
    """加载并校验单个样本（三件套）。"""
    failures: list[Failure] = []
    png_path = os.path.join(sample_dir, f"{sid}.png")
    assertion_path = os.path.join(sample_dir, f"{sid}.assertion.json")
    regions_path = os.path.join(sample_dir, f"{sid}.regions.json")

    try:
        image = read_png(png_path)
    except FileNotFoundError:
        failures.append(make_failure(E_SAMPLE_INVALID, "sample", sid, "png", f"缺少图像文件 {sid}.png"))
        image = None
    except PngError as exc:
        failures.append(make_failure(E_SAMPLE_INVALID, "sample", sid, "png", str(exc)))
        image = None

    assertion = _load_json(assertion_path, sid, "assertion", failures)
    regions = _load_json(regions_path, sid, "regions", failures)

    if image is not None:
        if expected_size is not None and (image.width, image.height) != expected_size:
            failures.append(
                make_failure(E_SAMPLE_INVALID, "sample", sid, "png.resolution",
                             f"分辨率 {image.width}x{image.height} 与固定分辨率 "
                             f"{expected_size[0]}x{expected_size[1]} 不一致")
            )
        if image.bit_depth != 8 or image.color_type != 2:
            failures.append(
                make_failure(E_SAMPLE_INVALID, "sample", sid, "png.header",
                             f"容器头要求 8 位真彩(type=2)，实际 depth={image.bit_depth} type={image.color_type}")
            )
        if regions is not None:
            failures.extend(_validate_regions(regions, image.width, image.height, sid))
    if assertion is not None:
        failures.extend(_validate_assertion(assertion, sid))

    if failures or image is None or assertion is None or regions is None:
        return None, failures
    return SampleBundle(sid, image, assertion, regions, png_path, assertion_path, regions_path), failures


def scan_samples(sample_dir: str, fixed_size: tuple[int, int]) -> tuple[dict[str, SampleBundle], list[Failure]]:
    """扫描样本目录，按文件前缀发现样本，校验 ID 唯一性并返回合法样本映射。"""
    failures: list[Failure] = []
    discovered: set[str] = set()
    if not os.path.isdir(sample_dir):
        return {}, failures
    for name in sorted(os.listdir(sample_dir)):
        if name.endswith(".png"):
            discovered.add(name[:-4])
        elif name.endswith(".assertion.json"):
            discovered.add(name[: -len(".assertion.json")])
        elif name.endswith(".regions.json"):
            discovered.add(name[: -len(".regions.json")])

    bundles: dict[str, SampleBundle] = {}
    seen: set[str] = set()
    for sid in sorted(discovered):
        if sid in seen:
            failures.append(
                make_failure(E_SAMPLE_INVALID, "sample", sid, "sample_id", "样本标识重复")
            )
            continue
        seen.add(sid)
        bundle, sample_failures = load_sample(sample_dir, sid, fixed_size)
        failures.extend(sample_failures)
        if bundle is not None:
            bundles[sid] = bundle
    return bundles, failures
