"""增量对齐引擎：编排检测/估计/填补/时间轴/依赖图/快照/持久化。

关键不变量：
- 每次参数修改先计算失效闭包，仅重算闭包内节点；
- 闭包外节点保留原内容哈希（复用并记录 reuse_source）；
- 每步重算后原子提交节点状态与快照，保证强制终止可恢复；
- 闭包清单与实际重算集合由 graph.assert_closure_matches 强校验。
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from . import detection as det
from . import media as media_mod
from .config import (
    PARAM_REFERENCE,
    WorkspaceConfig,
    default_config,
)
from .estimation import estimate_channel
from .gaps import fill_channel
from .graph import (
    NODE_AALIGN,
    NODE_DRIFT,
    NODE_GAPFILL,
    NODE_MAP,
    NODE_TILE,
    NODE_VALIGN,
    Change,
    DependencyGraph,
    node_id,
)
from .hashing import payload_hash
from .persistence import Store
from .snapshot import SnapshotStore, create_snapshot
from .timeline import build_dual_domain


class Engine:
    def __init__(self, ws_dir: str, cfg: WorkspaceConfig | None = None) -> None:
        self.ws_dir = ws_dir
        self.cfg = cfg or default_config()
        os.makedirs(ws_dir, exist_ok=True)
        self.store = Store(ws_dir)
        self.graph = DependencyGraph(n_channels=len(self.cfg.truths))
        self.snapshots = SnapshotStore()
        self.seq = 0
        self.manifest: dict[str, Any] | None = None
        self._audio: dict[int, np.ndarray] = {}
        self._video: dict[int, np.ndarray] = {}
        self._estimations: dict[int, Any] = {}
        self._gaps: dict[int, Any] = {}
        self._duals: dict[int, Any] = {}
        self._audio_gap_runs: dict[int, list] = {}
        self._video_gap_runs: dict[int, list] = {}

    # -- 素材准备 -------------------------------------------------------------
    def prepare_media(self) -> dict[str, Any]:
        manifest_path = os.path.join(self.ws_dir, "manifest.json")
        if os.path.exists(manifest_path):
            import json
            with open(manifest_path, "rb") as fh:
                self.manifest = json.loads(fh.read())
        else:
            self.manifest = media_mod.generate_workspace(self.cfg, self.ws_dir)
        for entry in self.manifest["channels"]:
            ch = entry["channel"]
            self._audio[ch] = det.load_audio(entry, self.ws_dir)
            self._video[ch] = det.load_video(entry, self.ws_dir)
            self._audio_gap_runs[ch] = det.detect_gaps_audio(self._audio[ch])
            self._video_gap_runs[ch] = det.detect_gaps_video(self._video[ch])
        return self.manifest

    def validate_media(self) -> list:
        assert self.manifest is not None
        return det.validate_manifest_media(self.manifest, self.ws_dir)

    # -- 单节点计算 -----------------------------------------------------------
    def _compute_drift(self, ch: int) -> Any:
        audio = self._audio[ch]
        video = self._video[ch]
        ref_pulses = np.array(self.manifest["reference_pulse_samples"], dtype=np.int64)
        ref_frames = np.array(self.manifest["reference_sync_frames"], dtype=np.int64)
        pulse_hits, pulse_errors = det.detect_pulses(audio, ref_pulses, self.cfg)
        sync_hits, frame_errors = det.detect_sync_frames(video, ref_frames)
        truth_ppm = self.cfg.truth(ch).ppm
        result = estimate_channel(ch, pulse_hits, sync_hits, audio,
                                  self.cfg, truth_ppm)
        # 检测类错误并入
        object.__setattr__(result, "errors",
                           result.errors + tuple(pulse_errors) + tuple(frame_errors))
        self._estimations[ch] = result
        return result

    def _compute_gapfill(self, ch: int) -> Any:
        est = self._estimations[ch]
        param = self.cfg.channel_params[ch]
        rate_ratio = est.segments[-1].slope
        result = fill_channel(
            ch, self._audio[ch], self._video[ch],
            self._audio_gap_runs[ch], self._video_gap_runs[ch],
            param.gap_policy,
            drift_version=self.graph.nodes[node_id(NODE_DRIFT, ch)].content_hash or "",
            rate_ratio=rate_ratio, cfg=self.cfg)
        self._gaps[ch] = result
        return result

    def _compute_dual(self, ch: int) -> Any:
        est = self._estimations[ch]
        ref_est = self._estimations[self.cfg.reference_channel]
        reference_slope = ref_est.segments[-1].slope
        # 时间轴版本由上游漂移哈希与参数版本派生（自洽、无自引用循环），
        # 初次计算与恢复重建得到完全相同的版本戳
        drift_hash = self.graph.nodes[node_id(NODE_DRIFT, ch)].content_hash or ""
        # 仅纳入 map 真实依赖：本通道漂移、本通道延迟开关、参考通道。
        # 不含全局 param_version，避免无关通道修改改变版本戳。
        map_version = payload_hash(
            {"drift": drift_hash,
             "reference": self.cfg.reference_channel,
             "delay_comp": self.cfg.channel_params[ch].delay_compensation}, [])
        result = build_dual_domain(est, self.cfg, reference_slope, map_version)
        self._duals[ch] = result
        return result

    def _node_payload(self, kind: str, ch: int) -> Any:
        if kind == NODE_DRIFT:
            return self._estimations[ch].to_dict()
        if kind == NODE_GAPFILL:
            return self._gaps[ch].to_dict()
        if kind in (NODE_MAP, NODE_AALIGN, NODE_VALIGN):
            dual = self._duals[ch]
            if kind == NODE_MAP:
                return dual.timeline.to_dict()
            if kind == NODE_AALIGN:
                return [s.to_dict() for s in dual.audio_segments]
            return [s.to_dict() for s in dual.video_segments]
        if kind == NODE_TILE:
            # 瓦片投影快照在 snapshot 阶段构建；节点记录其引用的上游哈希
            return {
                "drift": self.graph.nodes[node_id(NODE_DRIFT, ch)].content_hash,
                "gapfill": self.graph.nodes[node_id(NODE_GAPFILL, ch)].content_hash,
                "aalign": self.graph.nodes[node_id(NODE_AALIGN, ch)].content_hash,
                "valign": self.graph.nodes[node_id(NODE_VALIGN, ch)].content_hash,
            }
        raise ValueError(kind)  # pragma: no cover

    def _commit_node(self, kind: str, ch: int) -> None:
        nid = node_id(kind, ch)
        payload = self._node_payload(kind, ch)
        digest, blob_path = self.store.write_blob(payload)
        content_hash = payload_hash(payload, [digest])
        self.graph.record_recompute(
            nid, content_hash, self.cfg.param_version(), (digest,), payload)

    # -- 闭包重算 -------------------------------------------------------------
    def _recompute_closure(self, closure: set[str], reused_channels: set[int]) -> None:
        recomputed: set[str] = set()
        # 拓扑序：drift -> gapfill/map -> aalign/valign -> tile
        for ch in range(len(self.cfg.truths)):
            for kind in (NODE_DRIFT, NODE_GAPFILL, NODE_MAP,
                         NODE_AALIGN, NODE_VALIGN, NODE_TILE):
                nid = node_id(kind, ch)
                if nid not in closure:
                    continue
                if kind == NODE_DRIFT:
                    self._compute_drift(ch)
                elif kind == NODE_GAPFILL:
                    self._compute_gapfill(ch)
                elif kind == NODE_MAP:
                    self._compute_dual(ch)
                elif kind in (NODE_AALIGN, NODE_VALIGN):
                    pass  # 与 map 同一 dual 产物，仅提交节点
                elif kind == NODE_TILE:
                    pass
                self._commit_node(kind, ch)
                recomputed.add(nid)
        # 闭包外节点：记录复用来源（哈希保持不变）
        for nid, st in self.graph.nodes.items():
            if nid not in closure and st.status == "ok":
                self.graph.record_reuse(nid, nid)
        self.graph.assert_closure_matches(closure, recomputed)

    # -- 初始全量构建 ----------------------------------------------------------
    def initial_build(self) -> dict[str, Any]:
        self.prepare_media()
        format_errors = self.validate_media()
        closure = {node_id(k, c) for c in range(len(self.cfg.truths))
                   for k in ("drift", "gapfill", "map", "aalign",
                             "valign", "tile")}
        self.graph.mark_stale(closure)
        self._recompute_closure(closure, set())
        self.seq += 1
        state_hash = self.store.commit_node_state(
            self.graph.manifest(), self.cfg.param_version(), self.seq,
            self.cfg.snapshot())
        self.store.append_journal({
            "seq": self.seq, "event": "initial_build",
            "param_version": self.cfg.param_version(),
            "closure_size": len(closure), "state_hash": state_hash,
            "format_errors": [e.to_dict() for e in format_errors],
        })
        snap = self.publish_snapshot()
        return {"state_hash": state_hash, "snapshot": snap,
                "format_errors": format_errors, "closure": sorted(closure)}

    # -- 增量修改 -------------------------------------------------------------
    def apply_change(self, key: str, channel: int | None, value: Any,
                     simulate_crash: bool = False) -> dict[str, Any]:
        old_cfg = self.cfg
        if key == PARAM_REFERENCE:
            new_cfg = old_cfg.with_reference(int(value))
            old_value = old_cfg.reference_channel
        else:
            assert channel is not None
            old_value = getattr(old_cfg.channel_params[channel], key)
            new_cfg = old_cfg.with_channel_param(channel, key, value)
        change = Change(channel, key, old_value, value)
        closure = self.graph.closure(change)
        self.graph.mark_stale(closure)
        self.cfg = new_cfg

        if simulate_crash:
            # 强制终止：只留下未提交的 .tmp，不写关闭标记，直接硬退出
            self._simulate_dirty_write(closure)
            self._hard_exit()

        self._recompute_closure(closure, set())
        self.seq += 1
        state_hash = self.store.commit_node_state(
            self.graph.manifest(), self.cfg.param_version(), self.seq,
            self.cfg.snapshot())
        self.graph.changes.append({
            "seq": self.seq, "key": key, "channel": channel,
            "old": old_value, "new": value,
            "closure": sorted(closure),
        })
        self.store.append_journal({
            "seq": self.seq, "event": "incremental_change",
            "key": key, "channel": channel,
            "old": old_value, "new": value,
            "closure_size": len(closure),
            "closure": sorted(closure),
            "param_version": self.cfg.param_version(),
            "state_hash": state_hash,
        })
        snap = self.publish_snapshot()
        return {"closure": sorted(closure), "state_hash": state_hash,
                "snapshot": snap}

    # -- 快照 -----------------------------------------------------------------
    def publish_snapshot(self) -> dict[str, Any]:
        per_channel = {}
        for ch in range(len(self.cfg.truths)):
            per_channel[ch] = {
                "estimation": self._estimations[ch],
                "gap": self._gaps[ch],
                "dual": self._duals[ch],
                "reused": False,
            }
        version = self.snapshots.current.version + 1 if self.snapshots.current else 1
        prior = self.snapshots.degrade_count_total
        snap = create_snapshot(
            version, self.cfg.param_version(), per_channel, self.graph,
            self.cfg.tile_budget_ms, prior)
        self.snapshots.replace(snap)
        self.snapshots.degrade_count_total = snap.degraded
        path = self.store.commit_snapshot(snap.to_dict())
        self.store.append_journal({
            "seq": self.seq, "event": "snapshot",
            "snapshot_version": snap.version,
            "content_hash": snap.content_hash, "path": path,
        })
        return snap.to_dict()

    # -- 受控退出 / 强制终止 ---------------------------------------------------
    def controlled_shutdown(self) -> dict[str, Any]:
        report = {
            "seq": self.seq,
            "param_version": self.cfg.param_version(),
            "node_count": len(self.graph.nodes),
            "snapshot_version": (self.snapshots.current.version
                                 if self.snapshots.current else 0),
            "journal_events": len(self.store.journal_events()),
        }
        self.store.write_shutdown(report)
        self.store.append_journal({"seq": self.seq, "event": "shutdown", **report})
        return report

    def _simulate_dirty_write(self, closure: set[str]) -> None:
        # 写入一个未完成的 .tmp（模拟提交中途被 kill -9）
        path = os.path.join(self.store.root, "nodes", "state.json.tmp")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(b'{"seq": %d, "uncommitted": true' % (self.seq + 1))
            fh.flush()
            os.fsync(fh.fileno())
        self.store.append_journal({
            "seq": self.seq + 1, "event": "forced_termination",
            "closure": sorted(closure),
        })

    @staticmethod
    def _hard_exit() -> None:
        # 不用 SystemExit（那是受控异常），模拟 SIGKILL 的硬终止
        os._exit(137)

    # -- 恢复 -----------------------------------------------------------------
    def restore_from_store(self) -> dict[str, Any]:
        summary = self.store.recover()
        state = self.store.load_node_state()
        if state is None:
            return summary
        # 按崩溃前最后一次已提交的参数快照重建配置（含所有增量修改）
        self.cfg = self.cfg.with_snapshot(state["config_snapshot"])
        # 重建内存中的节点状态（内容哈希/版本），产物按需要惰性重算
        for node in state["graph"]["nodes"]:
            nid = node["node"]
            st = self.graph.nodes[nid]
            st.version = node["version"]
            st.content_hash = node["content_hash"]
            st.param_version = node["param_version"]
            st.blob_hashes = tuple(node["blob_hashes"])
            st.status = node["status"]
            st.reuse_source = node.get("reuse_source")
            st.recompute_count = node.get("recompute_count", 0)
        self.graph.changes = state["graph"].get("changes", [])
        self.seq = state["seq"]
        # 依据最近已提交状态重建内存中的计算产物，并校验内容哈希未漂移
        self._rebuild_in_memory(state)
        snap = self.store.latest_snapshot()
        if snap is not None:
            from .snapshot import Snapshot, Tile
            tiles = tuple(Tile(**t) for t in snap["tiles"])
            restored = Snapshot(
                version=snap["version"],
                param_version=snap["param_version"],
                tiles=tiles,
                timeline_hashes=snap["timeline_hashes"],
                degraded=snap["degraded"],
                content_hash=snap["content_hash"],
                errors=tuple(snap["errors"]),
            )
            self.snapshots.replace(restored)
            self.snapshots.degrade_count_total = restored.degraded
        return summary

    def _rebuild_in_memory(self, state: dict[str, Any]) -> None:
        """恢复路径：重新执行确定性计算填充内存，节点版本/哈希保持已提交值。"""
        self.prepare_media()
        saved = {n["node"]: n for n in state["graph"]["nodes"]}
        # 先回填已提交的内容哈希/参数版本，使 gapfill 等节点内嵌的
        # 上游版本戳与原始计算时完全一致
        for nid, node in saved.items():
            st = self.graph.nodes[nid]
            st.content_hash = node["content_hash"]
            st.param_version = node["param_version"]
            st.blob_hashes = tuple(node["blob_hashes"])
        # 先算完所有通道漂移，保证参考通道（可能为任意通道）估计已就绪
        for ch in range(len(self.cfg.truths)):
            self._compute_drift(ch)
        for ch in range(len(self.cfg.truths)):
            self._compute_gapfill(ch)
            self._compute_dual(ch)
            for kind in (NODE_DRIFT, NODE_GAPFILL, NODE_MAP,
                         NODE_AALIGN, NODE_VALIGN, NODE_TILE):
                nid = node_id(kind, ch)
                payload = self._node_payload(kind, ch)
                blob = saved[nid]["blob_hashes"][0]
                rebuilt_hash = payload_hash(payload, [blob])
                if rebuilt_hash != saved[nid]["content_hash"]:
                    from .errors import BusinessError, E_DEPENDENCY_STALE
                    raise BusinessError(
                        E_DEPENDENCY_STALE,
                        f"恢复后节点 {nid} 重算哈希与已提交状态不一致",
                        channel=ch,
                        detail={"stored": saved[nid]["content_hash"],
                                "rebuilt": rebuilt_hash})
