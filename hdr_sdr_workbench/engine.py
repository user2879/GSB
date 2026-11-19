"""工作台引擎：实算管线、增量重算与历史状态重建。

设计要点
--------
* 每次帧输出都由 :func:`hdr_sdr_workbench.pipeline.run_pipeline` 实算，
  输出以 (state_id, source_frame) 内容寻址缓存；缓存命中时仍会重新计算
  SHA-256 并与记录比对，任何近似/损坏缓存都会触发 E_HASH_MISMATCH。
* 一次操作产生的版本戳 = 操作 version。输出文件名内嵌版本戳，历史版本
  产物不会被覆盖，任意历史状态都可从操作序列重放并实算重建。
* 增量重算严格依据 :class:`StateTable` 给出的失效闭包，闭包外帧不触碰，
  其输出哈希保持不变。
"""

from __future__ import annotations

import os

import numpy as np

from .errors import (E_CLOSURE_DIVERGE, E_HASH_MISMATCH, E_RECOVER_ORPHAN,
                     WorkbenchError)
from .persistence import (DIR_OUTPUTS, WorkspaceStore, atomic_write,
                          atomic_write_json, metric_name, output_name,
                          sha256_bytes, sha256_file)
from .pipeline import (M_IDENTITY, M_2020_TO_709, _matmul_pixel,
                       frame_metrics, run_pipeline)
from .state import PipelineState, StateTable
from .synthesis import SequenceConfig, frame_path, read_half_frame


class WorkbenchEngine:
    def __init__(self, workspace: str, source_manifest: dict):
        self.workspace = workspace
        self.store = WorkspaceStore(workspace)
        self.source_manifest = source_manifest
        cfg = source_manifest["config"]
        self.config = SequenceConfig(width=cfg["width"], height=cfg["height"],
                                     frame_count=cfg["frame_count"], seed=cfg["seed"])
        self.config.validate()
        self.frame_count = self.config.frame_count
        self.state_table = self.store.load_state_table() or StateTable(self.frame_count)
        self.frame_entries: dict[int, dict] = self._load_entries()
        # 全新工作区：实算全部基线帧（v0）并完成首次稳定提交
        if not self.frame_entries:
            self._compute_baseline()

    def _compute_baseline(self) -> None:
        for index in range(self.frame_count):
            state = self.state_table.states[index]
            raw, metrics, digest = self.compute_frame(index, state)
            self.frame_entries[index] = self._persist_frame_artifact(
                index, 0, raw, metrics, state)
        self._commit()

    def _load_entries(self) -> dict:
        manifest = self.store.read_manifest()
        if manifest is None:
            return {}
        return {int(k): dict(v) for k, v in manifest["frames"].items()}

    def load_source_frame(self, index: int) -> np.ndarray:
        return read_half_frame(frame_path(os.path.join(self.workspace, "frames"), index),
                               self.config.width, self.config.height)

    # -- 单帧实算 ---------------------------------------------------------
    def compute_frame(self, index: int, state: PipelineState) -> tuple[bytes, dict, str]:
        """实算单帧，返回 (u8 raw bytes, metrics dict, output sha256)。"""
        linear_src = self.load_source_frame(index)
        u8, mapped = run_pipeline(
            linear_src, state.colorspace, float(state.exposure_stops),
            state.tonemap, float(state.tonemap_param), state.oetf)
        # 指标基于“曝光+色域转换后、tonemap 前”的线性数据重算一次
        gain = np.float32(2.0 ** float(state.exposure_stops))
        exposed = (linear_src * gain).astype(np.float32)
        matrix = M_IDENTITY if state.colorspace == "bt2020" else M_2020_TO_709
        converted = np.maximum(_matmul_pixel(exposed, matrix), np.float32(0.0))
        metrics = frame_metrics(converted, mapped, state.colorspace)
        raw = u8.tobytes()
        return raw, metrics, sha256_bytes(raw)

    def _persist_frame_artifact(self, index: int, version: int,
                                raw: bytes, metrics: dict, state: PipelineState) -> dict:
        out_file = output_name(index, version)
        met_file = metric_name(index, version)
        atomic_write(os.path.join(self.store.paths[DIR_OUTPUTS], out_file), raw)
        metric_doc = {
            "frame": index,
            "version": version,
            "state_id": state.state_id(),
            "state": state.to_dict(),
            "metrics": metrics,
            "output_file": out_file,
            "output_sha256": sha256_bytes(raw),
        }
        atomic_write_json(os.path.join(self.store.paths["metrics"], met_file), metric_doc)
        return {
            "frame": index,
            "version": version,
            "state_id": state.state_id(),
            "output_file": out_file,
            "metric_file": met_file,
            "output_hash": metric_doc["output_sha256"],
        }

    # -- 增量操作 ---------------------------------------------------------
    def apply_operation(self, frames: list[int], changes: dict, *,
                        kind: str = "adjust", note: str = "",
                        commit: bool = True) -> dict:
        before_ids = self.state_table.state_ids()
        op = self.state_table.apply(frames, changes, kind=kind, note=note)
        closure = op["closure"]
        version = op["version"]
        recomputed = []
        for index in closure:
            state = self.state_table.states[index]
            raw, metrics, digest = self.compute_frame(index, state)
            entry = self._persist_frame_artifact(index, version, raw, metrics, state)
            # 立即回读校验，防止脏写冒充实算
            back = sha256_file(os.path.join(self.store.paths[DIR_OUTPUTS], entry["output_file"]))
            if back != digest:
                raise WorkbenchError(E_HASH_MISMATCH,
                                     f"帧 {index} 输出回读哈希不符",
                                     details={"frame": index, "expected": digest, "actual": back})
            self.frame_entries[index] = entry
            recomputed.append({"frame": index, "output_hash": digest})
        if commit:
            self._commit()
        # 闭包外状态不变性自检：闭包外帧 state_id 必须与操作前完全一致
        after_ids = self.state_table.state_ids()
        for index in range(self.frame_count):
            if index not in closure and before_ids[index] != after_ids[index]:
                raise WorkbenchError(
                    E_CLOSURE_DIVERGE,
                    f"帧 {index} 不在失效闭包内但状态发生变化",
                    details={"frame": index,
                             "before": before_ids[index], "after": after_ids[index]})
        return {"operation": op, "closure_size": len(closure),
                "closure": closure, "recomputed": recomputed, "version": version}

    def _commit(self) -> dict:
        return self.store.commit(self.state_table, self.frame_entries, self.source_manifest)

    def commit(self) -> dict:
        return self._commit()

    # -- 历史状态重建（禁止近似缓存）--------------------------------------
    def rebuild_historical(self, version: int, frames: list[int] | None = None) -> list[dict]:
        """重放到历史版本并对指定帧实算重建，哈希须与历史产物一致。"""
        historic = self.state_table.replay_to(version)
        targets = sorted(frames) if frames is not None else list(range(self.frame_count))
        results = []
        for index in targets:
            state = historic.states[index]
            raw, _metrics, digest = self.compute_frame(index, state)
            historic_file = output_name(index, version)
            path = os.path.join(self.store.paths[DIR_OUTPUTS], historic_file)
            if os.path.exists(path):
                recorded = sha256_file(path)
                if recorded != digest:
                    raise WorkbenchError(
                        E_HASH_MISMATCH,
                        f"历史重算帧 {index}@v{version} 与当时产物哈希不一致",
                        details={"frame": index, "version": version,
                                 "recomputed": digest, "recorded": recorded})
            results.append({"frame": index, "version": version,
                            "state_id": state.state_id(), "output_hash": digest})
        return results

    # -- 全量校验 ---------------------------------------------------------
    def verify_all(self) -> dict:
        """对所有当前已提交帧做哈希校验，并实算比对（缓存不可信即报错）。"""
        failures = []
        for index, entry in sorted(self.frame_entries.items()):
            path = os.path.join(self.store.paths[DIR_OUTPUTS], entry["output_file"])
            if not os.path.exists(path) or sha256_file(path) != entry["output_hash"]:
                failures.append({"frame": index, "file": entry["output_file"]})
                continue
            state = self.state_table.states[index]
            _raw, _metrics, digest = self.compute_frame(index, state)
            if digest != entry["output_hash"]:
                raise WorkbenchError(E_HASH_MISMATCH,
                                     f"帧 {index} 实算与缓存哈希不符（缓存不可信）",
                                     details={"frame": index})
        if failures:
            raise WorkbenchError(E_HASH_MISMATCH, "产物哈希校验失败",
                                 details={"failures": failures})
        return {"verified_frames": len(self.frame_entries), "failures": 0}

    # -- 受控退出 ---------------------------------------------------------
    def controlled_shutdown(self) -> dict:
        manifest = self._commit()
        return self.store.write_shutdown_report(manifest)

    # -- 恢复（孤儿 -> 记录后继续）----------------------------------------
    def recover(self) -> dict:
        try:
            report = self.store.recover()
        except WorkbenchError as exc:
            if exc.code != E_RECOVER_ORPHAN:
                raise
            # 孤儿已被回收；以 manifest 重建内存状态后继续可操作
            report = dict(exc.details)
        table = self.store.load_state_table()
        if table is not None:
            self.state_table = table
            self.frame_entries = self._load_entries()
        report["recovered_frames"] = len(self.frame_entries)
        return {"recovered": True, **report}

    def hash_listing(self) -> list[dict]:
        return [
            {"frame": index,
             "version": self.frame_entries[index]["version"],
             "state_id": self.frame_entries[index]["state_id"],
             "output_sha256": self.frame_entries[index]["output_hash"]}
            for index in sorted(self.frame_entries)
        ]
