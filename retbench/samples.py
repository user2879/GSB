"""样本目录扫描、PNG/区域真值/文本断言校验。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from .errors import E_SAMPLE_INVALID, FailureItem

TEXT_SUFFIX = ".text.json"
REGION_SUFFIX = ".regions.json"


@dataclass
class Sample:
    name: str
    image_path: str
    text_path: str
    region_path: str
    width: int
    height: int
    image_array: np.ndarray            # H x W x 3, uint8
    regions: list[dict[str, Any]]
    entity: str
    attributes: list[str]
    assertions: list[dict[str, Any]]
    source_fingerprint: str
    image_fingerprint: str = ""
    text_fingerprint: str = ""

    def primitive_count(self) -> int:
        return len(self.regions)

    def assertion_len(self) -> int:
        return len(self.assertions)


@dataclass
class SampleSet:
    samples: dict[str, Sample] = field(default_factory=dict)
    failures: list[FailureItem] = field(default_factory=list)
    skipped: set[str] = field(default_factory=set)

    def ordered_names(self) -> list[str]:
        return sorted(self.samples)


def _failure(sample: str, stage: str, message: str, detail: dict[str, Any] | None = None) -> FailureItem:
    return FailureItem(E_SAMPLE_INVALID, sample, stage, message, detail or {})


def load_text_description(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_text(data: dict[str, Any], vocab: dict[str, list[str]]) -> str | None:
    if not isinstance(data, dict):
        return "文本描述必须为 JSON 对象"
    entity = data.get("entity")
    if not isinstance(entity, str) or not entity:
        return "文本描述缺少 entity 字段"
    if entity not in vocab["entities"]:
        return f"entity {entity!r} 不在受控词表内"
    attrs = data.get("attributes")
    if not isinstance(attrs, list) or not attrs:
        return "attributes 必须为非空列表"
    for attr in attrs:
        if not isinstance(attr, str) or attr not in vocab["attributes"]:
            return f"属性 {attr!r} 缺失或不在受控词表内"
    assertions = data.get("assertions")
    if not isinstance(assertions, list) or not assertions:
        return "assertions 必须为非空列表"
    for index, assertion in enumerate(assertions):
        if not isinstance(assertion, dict):
            return f"断言 #{index} 必须为对象"
        for key in ("predicate", "object"):
            if key not in assertion:
                return f"断言 #{index} 缺少字段 {key}"
        if assertion["predicate"] not in vocab["relations"]:
            return f"断言 #{index} 谓词不在受控词表内"
        if not isinstance(assertion["object"], str) or not assertion["object"]:
            return f"断言 #{index} object 必须为非空字符串"
    return None


def _validate_regions(regions: Any, width: int, height: int) -> str | None:
    if not isinstance(regions, list) or not regions:
        return "区域真值必须为非空列表"
    for index, region in enumerate(regions):
        if not isinstance(region, dict) or "bbox" not in region or "shape" not in region:
            return f"区域 #{index} 缺少 bbox/shape"
        bbox = region["bbox"]
        if not (isinstance(bbox, list) and len(bbox) == 4
                and all(isinstance(v, int) for v in bbox)):
            return f"区域 #{index} bbox 必须为 4 个整数"
        x, y, box_w, box_h = bbox
        if x < 0 or y < 0 or box_w <= 0 or box_h <= 0 or x + box_w > width or y + box_h > height:
            return f"区域 #{index} bbox 越界或尺寸非法"
        if not isinstance(region["shape"], str) or not region["shape"]:
            return f"区域 #{index} shape 非法"
    return None


def load_sample(
    name: str,
    image_path: str,
    text_path: str,
    region_path: str,
    resolution: tuple[int, int],
    vocab: dict[str, list[str]],
) -> tuple[Sample | None, FailureItem | None]:
    stage = "sample.validate"
    for path in (image_path, text_path, region_path):
        if not os.path.isfile(path):
            return None, _failure(name, stage, f"样本文件缺失: {os.path.basename(path)}")

    try:
        with Image.open(image_path) as image:
            if image.format != "PNG":
                return None, _failure(name, stage, "图像容器头不是 PNG")
            image.load()
            if image.mode != "RGB":
                image = image.convert("RGB")
            width, height = image.size
            if (width, height) != resolution:
                return None, _failure(name, stage,
                                      f"尺寸 {width}x{height} 与配置 {resolution[0]}x{resolution[1]} 不一致",
                                      {"actual": [width, height]})
            image_array = np.asarray(image, dtype=np.uint8).copy()
    except Exception as exc:  # noqa: BLE001 - 容器头损坏等统一进失败清单
        return None, _failure(name, stage, f"PNG 读取失败: {exc}")

    try:
        with open(region_path, "r", encoding="utf-8") as handle:
            region_data = json.load(handle)
        regions = region_data.get("regions") if isinstance(region_data, dict) else None
    except Exception as exc:  # noqa: BLE001
        return None, _failure(name, stage, f"区域真值 JSON 解析失败: {exc}")
    region_error = _validate_regions(regions, width, height)
    if region_error:
        return None, _failure(name, stage, region_error)

    try:
        text_data = load_text_description(text_path)
    except Exception as exc:  # noqa: BLE001
        return None, _failure(name, stage, f"文本描述 JSON 解析失败: {exc}")
    text_error = _validate_text(text_data, vocab)
    if text_error:
        return None, _failure(name, stage, text_error)

    from .fingerprint import digest_obj

    image_fingerprint = digest_obj({
        "image_bytes_len": os.path.getsize(image_path),
        "image_sha_dims": [int(v) for v in image_array.shape],
        "regions": regions,
    })
    text_fingerprint = digest_obj(text_data)
    fingerprint_src = {
        "image": image_fingerprint,
        "text": text_fingerprint,
    }

    sample = Sample(
        name=name,
        image_path=image_path,
        text_path=text_path,
        region_path=region_path,
        width=width,
        height=height,
        image_array=image_array,
        regions=regions,
        entity=text_data["entity"],
        attributes=list(text_data["attributes"]),
        assertions=list(text_data["assertions"]),
        source_fingerprint=digest_obj(fingerprint_src),
        image_fingerprint=image_fingerprint,
        text_fingerprint=text_fingerprint,
    )
    return sample, None


def scan_samples(root: str, resolution: tuple[int, int],
                 vocab: dict[str, list[str]]) -> SampleSet:
    result = SampleSet()
    if not os.path.isdir(root):
        result.failures.append(_failure("<root>", "sample.scan", f"样本目录不存在: {root}"))
        return result

    png_files: dict[str, str] = {}
    for entry in sorted(os.listdir(root)):
        if entry.endswith(".png"):
            name = entry[:-4]
            if name in png_files:
                result.failures.append(_failure(name, "sample.scan", "样本名不唯一"))
                continue
            png_files[name] = os.path.join(root, entry)

    for name in sorted(png_files):
        image_path = png_files[name]
        text_path = os.path.join(root, f"{name}{TEXT_SUFFIX}")
        region_path = os.path.join(root, f"{name}{REGION_SUFFIX}")
        sample, failure = load_sample(
            name, image_path, text_path, region_path, resolution, vocab
        )
        if failure is not None:
            result.failures.append(failure)
            result.skipped.add(name)
            continue
        result.samples[name] = sample
    return result


def load_injection_manifest(path: str) -> list[dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("注入清单必须为列表")
    return data
