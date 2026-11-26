"""样本/关键点版本/命令流的加载与严格校验，坏样本隔离为 E_SAMPLE_INVALID。"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from .config import Config
from .errors import (E_KEYPOINT_VERSION_INVALID, E_SAMPLE_INVALID,
                     PipelineValidationError, ValidationIssue)
from .hashing import digest
from .sample_gen import PAGE_H, PAGE_W

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _has_cycle(adjacency: list[dict]) -> bool:
    graph: dict[int, list[int]] = {}
    for edge in adjacency:
        graph.setdefault(edge["from"], []).append(edge["to"])
    visiting: set[int] = set()
    visited: set[int] = set()

    def walk(node: int) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for nxt in graph.get(node, []):
            if walk(nxt):
                return True
        visiting.discard(node)
        visited.add(node)
        return False

    return any(walk(node) for node in list(graph.keys()))


@dataclass
class Sample:
    sample_id: str
    directory: Path
    image: Image.Image
    image_fingerprint: str
    truth: dict
    declaration: dict
    keypoint_versions: dict
    active_keypoint_fingerprint: str


@dataclass
class LoadResult:
    samples: dict[str, Sample] = field(default_factory=dict)
    issues: list[ValidationIssue] = field(default_factory=list)


def _validate_png(path: Path, sample_id: str) -> Image.Image:
    raw = path.read_bytes()
    if not raw.startswith(PNG_SIGNATURE):
        raise PipelineValidationError(
            E_SAMPLE_INVALID, sample_id, "page.png:0", "PNG 容器头签名不合法")
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception as exc:  # pragma: no cover - PIL 自身异常类型不稳定
        raise PipelineValidationError(
            E_SAMPLE_INVALID, sample_id, "page.png", f"PNG 解码失败: {exc}") from exc
    if image.mode != "L":
        raise PipelineValidationError(
            E_SAMPLE_INVALID, sample_id, "page.png:mode",
            f"像素模式必须为 L，实际为 {image.mode}")
    if image.size != (PAGE_W, PAGE_H):
        raise PipelineValidationError(
            E_SAMPLE_INVALID, sample_id, "page.png:size",
            f"分辨率必须为 {PAGE_W}x{PAGE_H}，实际为 {image.size[0]}x{image.size[1]}")
    return image


def _validate_truth(truth: dict, sample_id: str) -> None:
    if truth.get("page") != {"width": PAGE_W, "height": PAGE_H, "mode": "L"}:
        raise PipelineValidationError(
            E_SAMPLE_INVALID, sample_id, "truth.json:page", "版面声明与固定分辨率不一致")
    blocks = truth.get("blocks", [])
    seen_orders: list[int] = []
    for block in blocks:
        loc = f"truth.json:blocks[{block.get('block_index')}]"
        kp = block.get("keypoint", {})
        bbox = block.get("bbox", {})
        for key in ("x", "y"):
            if not _is_int(kp.get(key)):
                raise PipelineValidationError(
                    E_SAMPLE_INVALID, sample_id, f"{loc}.keypoint.{key}",
                    "关键点坐标必须为整数")
        if not (0 <= kp["x"] < PAGE_W and 0 <= kp["y"] < PAGE_H):
            raise PipelineValidationError(
                E_SAMPLE_INVALID, sample_id, f"{loc}.keypoint",
                f"关键点 ({kp['x']},{kp['y']}) 越出版面 {PAGE_W}x{PAGE_H}")
        for key in ("x", "y", "w", "h"):
            if not _is_int(bbox.get(key)) or bbox[key] <= 0:
                raise PipelineValidationError(
                    E_SAMPLE_INVALID, sample_id, f"{loc}.bbox.{key}",
                    "边界框必须为正整数")
        if not (0 <= bbox["x"] and bbox["x"] + bbox["w"] <= PAGE_W and
                0 <= bbox["y"] and bbox["y"] + bbox["h"] <= PAGE_H):
            raise PipelineValidationError(
                E_SAMPLE_INVALID, sample_id, f"{loc}.bbox", "边界框越出版面")
        if not _is_int(block.get("reading_order")):
            raise PipelineValidationError(
                E_SAMPLE_INVALID, sample_id, f"{loc}.reading_order", "阅读序号必须为整数")
        seen_orders.append(block["reading_order"])
    if seen_orders != list(range(len(seen_orders))):
        raise PipelineValidationError(
            E_SAMPLE_INVALID, sample_id, "truth.json:blocks.reading_order",
            f"阅读序号必须从 0 连续无跳号，实际为 {seen_orders}")


def _validate_declaration(declaration: dict, sample_id: str) -> None:
    columns = declaration.get("columns", [])
    indices = [col["column_index"] for col in columns]
    intervals = sorted(
        ((col["x_interval"]["x_min"], col["x_interval"]["x_max"]) for col in columns),
        key=lambda item: item[0])
    for prev, nxt in zip(intervals, intervals[1:]):
        if nxt[0] <= prev[1]:
            raise PipelineValidationError(
                E_SAMPLE_INVALID, sample_id, "columns.json:x_interval",
                f"栏横坐标区间重叠: {prev} 与 {nxt}")
    valid = set(indices)
    for path in declaration.get("candidate_paths", []):
        for col in path["path"]:
            if col not in valid:
                raise PipelineValidationError(
                    E_SAMPLE_INVALID, sample_id,
                    f"columns.json:candidate_paths:{path.get('priority')}",
                    f"候选路径引用了不存在的栏 {col}")
    adjacency = declaration.get("adjacency", [])
    for edge in adjacency:
        if edge["from"] == edge["to"]:
            raise PipelineValidationError(
                E_SAMPLE_INVALID, sample_id,
                f"columns.json:adjacency:{edge['from']}->{edge['to']}",
                "栏间邻接关系不允许自引用")
        if edge["from"] not in valid or edge["to"] not in valid:
            raise PipelineValidationError(
                E_SAMPLE_INVALID, sample_id, "columns.json:adjacency",
                f"邻接关系引用了不存在的栏 {edge}")
    if _has_cycle(adjacency):
        raise PipelineValidationError(
            E_SAMPLE_INVALID, sample_id, "columns.json:adjacency",
            "栏间邻接关系构成环")


def _validate_keypoint_versions(payload: dict, sample_id: str) -> str:
    active = payload.get("active_version")
    versions = payload.get("versions", {})
    if active not in versions:
        raise PipelineValidationError(
            E_KEYPOINT_VERSION_INVALID, sample_id, "keypoint_versions.json:active_version",
            f"激活版本 {active!r} 不存在")
    for name, version in versions.items():
        points = version.get("points", [])
        for point in points:
            loc = f"keypoint_versions.json:{name}[{point.get('block_index')}]"
            for key in ("x", "y", "block_index", "column_index"):
                if not _is_int(point.get(key)):
                    raise PipelineValidationError(
                        E_KEYPOINT_VERSION_INVALID, sample_id, f"{loc}.{key}",
                        "关键点字段必须为整数")
            if not (0 <= point["x"] < PAGE_W and 0 <= point["y"] < PAGE_H):
                raise PipelineValidationError(
                    E_KEYPOINT_VERSION_INVALID, sample_id, loc,
                    f"关键点 ({point['x']},{point['y']}) 越界")
        recomputed = digest(points)
        if recomputed != version.get("fingerprint"):
            raise PipelineValidationError(
                E_KEYPOINT_VERSION_INVALID, sample_id,
                f"keypoint_versions.json:{name}.fingerprint",
                "关键点版本指纹与点集内容不一致")
    return versions[active]["fingerprint"]


def load_sample(sample_dir: Path) -> Sample:
    sample_id = sample_dir.name
    image = _validate_png(sample_dir / "page.png", sample_id)
    truth = json.loads((sample_dir / "truth.json").read_text(encoding="utf-8"))
    _validate_truth(truth, sample_id)
    actual_image_fp = digest((sample_dir / "page.png").read_bytes())
    if truth.get("image_fingerprint") != actual_image_fp:
        raise PipelineValidationError(
            E_SAMPLE_INVALID, sample_id, "truth.json:image_fingerprint",
            "图像内容指纹与真值记录不一致，疑似图像被替换")
    declaration = json.loads((sample_dir / "columns.json").read_text(encoding="utf-8"))
    _validate_declaration(declaration, sample_id)
    keypoints = json.loads(
        (sample_dir / "keypoint_versions.json").read_text(encoding="utf-8"))
    active_fp = _validate_keypoint_versions(keypoints, sample_id)
    return Sample(
        sample_id=sample_id,
        directory=sample_dir,
        image=image,
        image_fingerprint=actual_image_fp,
        truth=truth,
        declaration=declaration,
        keypoint_versions=keypoints,
        active_keypoint_fingerprint=active_fp,
    )


def load_corpus(samples_root: str | Path) -> LoadResult:
    result = LoadResult()
    for sample_dir in sorted(Path(samples_root).iterdir(), key=lambda p: p.name):
        if not sample_dir.is_dir():
            continue
        try:
            sample = load_sample(sample_dir)
        except PipelineValidationError as exc:
            result.issues.append(exc.issue)
            continue
        result.samples[sample.sample_id] = sample
    return result


def load_command_stream(path: str | Path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise PipelineValidationError(
            E_SAMPLE_INVALID, "<commands>", path, "命令流必须为 JSON 数组")
    allowed = {
        "run", "replace_image", "add_sample", "remove_sample",
        "set_window", "set_quant", "replace_keypoint_version", "set_budget",
        "crash", "recover",
    }
    for idx, command in enumerate(payload):
        if command.get("op") not in allowed:
            raise PipelineValidationError(
                E_SAMPLE_INVALID, "<commands>", f"[{idx}].op",
                f"未知命令 {command.get('op')!r}")
    return payload


def validate_config_against_page(cfg: Config) -> None:
    if PAGE_W % cfg.window_size or PAGE_H % cfg.window_size:
        raise PipelineValidationError(
            E_SAMPLE_INVALID, "<config>", "window_size",
            "注入配置的滑窗尺寸不能整除版面边长")
