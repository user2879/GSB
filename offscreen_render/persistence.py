"""分块级持久化、版本识别、归档恢复与原子写出（业务要求 7、8）。

状态目录布局
------------
* ``tile_<idx>.json``        分块正式状态（原子替换）
* ``tile_<idx>.json.tmp.<pid>`` 同目录临时文件
* ``archive/``                版本标识变化时旧分块的归档目录
* ``version.json``            当前运行的规范化版本标识
* ``INTERRUPT``               中断标记
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from typing import Any

import numpy as np

from .errors import E_IO, E_STATE, RenderError
from .renderer import TileResult
from .scene import Scene

TILE_PREFIX = "tile_"
TILE_SUFFIX = ".json"
VERSION_FILE = "version.json"
ARCHIVE_DIRNAME = "archive"


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_version_id(scene: Scene) -> str:
    """由介质、光源、相机、步长与阈值的规范化摘要构成版本标识。"""
    cfg = scene.config
    payload = {
        "volumes": [
            {"index": v.index, "min": v.min_corner.tolist(), "max": v.max_corner.tolist(),
             "sigma_t": v.sigma_t, "albedo": v.albedo, "ior": v.ior}
            for v in scene.volumes
        ],
        "lights": [{"index": l.index, "position": l.position.tolist(),
                    "intensity": l.intensity} for l in scene.lights],
        "camera": {"position": scene.camera.position.tolist(),
                   "forward": scene.camera.forward.tolist(),
                   "fov_v_deg": scene.camera.fov_v_deg, "aspect": scene.camera.aspect},
        "render": {
            "width": cfg.width, "height": cfg.height, "supersample": cfg.supersample,
            "step_length": cfg.step_length, "max_steps": cfg.max_steps,
            "transmittance_cutoff": cfg.transmittance_cutoff,
            "fp32_mantissa_bits": cfg.fp32_mantissa_bits,
            "transmittance_tolerance": cfg.transmittance_tolerance,
            "tile_width": cfg.tile_width, "tile_height": cfg.tile_height,
        },
    }
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return f"v1-{digest[:32]}"


def _tile_path(state_dir: str, tile_index: int) -> str:
    return os.path.join(state_dir, f"{TILE_PREFIX}{tile_index:05d}{TILE_SUFFIX}")


def _content_digest(tile: TileResult) -> str:
    h = hashlib.sha256()
    for arr in (tile.grayscale, tile.diff_map, tile.steps_map):
        h.update(np.ascontiguousarray(arr).tobytes())
    h.update(tile.reason_map.astype("<U16").tobytes())
    h.update(_canonical_json({"x0": tile.x0, "y0": tile.y0,
                              "w": tile.width, "h": tile.height}).encode())
    return h.hexdigest()


def _atomic_write(path: str, text: str) -> None:
    directory = os.path.dirname(path) or "."
    tmp = f"{path}.tmp.{os.getpid()}.{thread_safe_nonce()}"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except OSError as exc:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise RenderError(E_IO, f"原子写出失败: {path}", {"message": str(exc)}) from exc


_NONCE_COUNTER = 0


def thread_safe_nonce() -> int:
    global _NONCE_COUNTER
    import threading
    with threading.Lock():
        _NONCE_COUNTER += 1
        return _NONCE_COUNTER


def tile_to_document(tile: TileResult, version_id: str) -> dict[str, Any]:
    digest = _content_digest(tile)
    tile.content_digest = digest
    return {
        "format": "offscreen-render-tile",
        "format_version": 1,
        "version_id": version_id,
        "tile_index": tile.tile_index,
        "x0": tile.x0,
        "y0": tile.y0,
        "width": tile.width,
        "height": tile.height,
        "content_digest": digest,
        "stats": {
            "cutoff_count": int(tile.cutoff_count),
            "max_steps_count": int(tile.max_steps_count),
            "max_abs_diff": float(np.max(tile.diff_map)) if tile.diff_map.size else 0.0,
            "first_exceeds": tile.first_exceeds,
        },
        "grayscale": np.ascontiguousarray(tile.grayscale, dtype=np.float64).tolist(),
        "diff_map": np.ascontiguousarray(tile.diff_map, dtype=np.float64).tolist(),
        "steps_map": np.ascontiguousarray(tile.steps_map, dtype=np.int32).tolist(),
        "reason_map": np.ascontiguousarray(tile.reason_map.astype("<U16")).tolist(),
    }


def save_tile(state_dir: str, tile: TileResult, version_id: str) -> str:
    """原子写入分块；若已有分块版本标识不一致，先归档再替换。"""
    path = _tile_path(state_dir, tile.tile_index)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                old = json.load(fh)
            old_version = old.get("version_id")
        except (OSError, json.JSONDecodeError) as exc:
            old_version = None
            old = None
        if old_version is not None and old_version != version_id:
            archive_dir = os.path.join(state_dir, ARCHIVE_DIRNAME)
            os.makedirs(archive_dir, exist_ok=True)
            stamp = hashlib.sha1(f"{old_version}".encode()).hexdigest()[:10]
            archived = os.path.join(
                archive_dir, f"{TILE_PREFIX}{tile.tile_index:05d}.{stamp}{TILE_SUFFIX}")
            shutil.move(path, archived)
        elif old_version is None:
            # 损坏的旧文件不可信任，归档而不是覆盖。
            archive_dir = os.path.join(state_dir, ARCHIVE_DIRNAME)
            os.makedirs(archive_dir, exist_ok=True)
            shutil.move(path, os.path.join(
                archive_dir, f"{TILE_PREFIX}{tile.tile_index:05d}.corrupt{TILE_SUFFIX}"))
    doc = tile_to_document(tile, version_id)
    _atomic_write(path, _canonical_json(doc))
    return path


def document_to_tile(doc: dict[str, Any]) -> TileResult:
    tile = TileResult(
        tile_index=int(doc["tile_index"]), x0=int(doc["x0"]), y0=int(doc["y0"]),
        width=int(doc["width"]), height=int(doc["height"]),
        grayscale=np.array(doc["grayscale"], dtype=np.float64),
        diff_map=np.array(doc["diff_map"], dtype=np.float64),
        steps_map=np.array(doc["steps_map"], dtype=np.int32),
        reason_map=np.array(doc["reason_map"], dtype="<U16"),
        first_exceeds=list(doc["stats"].get("first_exceeds", [])),
        cutoff_count=int(doc["stats"]["cutoff_count"]),
        max_steps_count=int(doc["stats"]["max_steps_count"]),
        content_digest=str(doc["content_digest"]),
    )
    return tile


def load_tile(state_dir: str, tile_index: int, version_id: str) -> TileResult | None:
    """返回版本一致且摘要有效的已持久化分块；否则返回 None（损坏文件归档）。"""
    path = _tile_path(state_dir, tile_index)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        if doc.get("version_id") != version_id:
            return None
        tile = document_to_tile(doc)
        if _content_digest(tile) != doc["content_digest"]:
            raise RenderError(E_STATE, "分块内容摘要校验失败", {"tile": tile_index})
        return tile
    except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        archive_dir = os.path.join(state_dir, ARCHIVE_DIRNAME)
        os.makedirs(archive_dir, exist_ok=True)
        try:
            shutil.move(path, os.path.join(
                archive_dir, f"{TILE_PREFIX}{tile_index:05d}.invalid{TILE_SUFFIX}"))
        except OSError:
            pass
        raise RenderError(E_STATE, f"分块 {tile_index} 状态文件损坏并已归档",
                          {"message": str(exc)}) from exc


def write_version(state_dir: str, version_id: str) -> None:
    _atomic_write(os.path.join(state_dir, VERSION_FILE),
                  _canonical_json({"version_id": version_id}))


def read_version(state_dir: str) -> str | None:
    path = os.path.join(state_dir, VERSION_FILE)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh).get("version_id")
    except (OSError, json.JSONDecodeError):
        return None


def list_persisted_tiles(state_dir: str) -> list[int]:
    out = []
    if not os.path.isdir(state_dir):
        return out
    for name in os.listdir(state_dir):
        if name.startswith(TILE_PREFIX) and name.endswith(TILE_SUFFIX):
            stem = name[len(TILE_PREFIX):-len(TILE_SUFFIX)]
            if stem.isdigit():
                out.append(int(stem))
    return sorted(out)


def write_interrupt_marker(state_dir: str) -> str:
    path = os.path.join(state_dir, "INTERRUPT")
    _atomic_write(path, _canonical_json({"interrupted": True}))
    return path


def interrupt_marker_exists(state_dir: str) -> bool:
    return os.path.exists(os.path.join(state_dir, "INTERRUPT"))


def remove_interrupt_marker(state_dir: str) -> None:
    path = os.path.join(state_dir, "INTERRUPT")
    if os.path.exists(path):
        os.remove(path)


def atomic_write_text(path: str, text: str) -> None:
    _atomic_write(path, text)
