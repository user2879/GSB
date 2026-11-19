"""增量渲染与从空白全量渲染的逐帧/逐潜变量一致性校验。"""
from __future__ import annotations

import shutil
from typing import Any, Mapping

import numpy as np

from .config import validate_config
from .engine import RenderEngine
from .errors import E_STATE_DIVERGENCE, WorkbenchError
from .persistence import SnapshotStore


def _snapshot_frames(store: SnapshotStore, frames: int) -> list[np.ndarray]:
    output = []
    for index in range(frames):
        artifact = store.load_artifact(f"decode_{index:03d}")
        if artifact is None:
            raise WorkbenchError(
                E_STATE_DIVERGENCE,
                f"全量渲染缺少帧产物 decode_{index:03d}",
                stage=f"decode_{index:03d}",
                frame=index,
            )
        output.append(artifact["array"])
    return output


def _snapshot_latents(store: SnapshotStore, steps: int) -> dict[str, np.ndarray]:
    output = {}
    for index in range(steps):
        name = f"denoise_{index:03d}"
        artifact = store.load_artifact(name)
        if artifact is not None:
            output[name] = artifact["array"]
    return output


def assert_snapshots_equal(
    incremental_dir: str,
    full_dir: str,
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """比较两份快照的逐帧像素与各中间潜变量，首个差异即抛 E_STATE_DIVERGENCE。"""
    inc_store = SnapshotStore(incremental_dir)
    full_store = SnapshotStore(full_dir)
    steps = int(cfg["steps"])
    frames = int(cfg["frames"])

    inc_latents = _snapshot_latents(inc_store, steps)
    full_latents = _snapshot_latents(full_store, steps)
    for name in sorted(set(inc_latents) | set(full_latents)):
        if name not in inc_latents or name not in full_latents:
            raise WorkbenchError(
                E_STATE_DIVERGENCE,
                f"阶段 {name} 仅存在于一份快照中",
                stage=name,
            )
        if not np.array_equal(inc_latents[name], full_latents[name]):
            raise WorkbenchError(
                E_STATE_DIVERGENCE,
                f"阶段 {name} 的中间潜变量不一致",
                stage=name,
            )

    inc_frames = _snapshot_frames(inc_store, frames)
    full_frames = _snapshot_frames(full_store, frames)
    for index, (left, right) in enumerate(zip(inc_frames, full_frames)):
        if not np.array_equal(left, right):
            diff = int(np.count_nonzero(left != right))
            raise WorkbenchError(
                E_STATE_DIVERGENCE,
                f"首个差异帧为 frame {index}，差异像素数 {diff}",
                stage=f"decode_{index:03d}",
                frame=index,
            )

    return {
        "equal": True,
        "frames_checked": frames,
        "latent_stages_checked": len(full_latents),
    }


def render_full(
    raw_config: Mapping[str, Any],
    base_dir: str,
    workdir: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """在空白工作区按给定配置一次性全量渲染。"""
    if __import__("os").path.exists(workdir):
        shutil.rmtree(workdir)
    cfg = validate_config(raw_config, base_dir)
    store = SnapshotStore(workdir)
    engine = RenderEngine(cfg, store)
    report = engine.run()
    return cfg, report
