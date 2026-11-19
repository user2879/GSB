"""修复工作台编排引擎。

职责：
  1. 解析素材并校验工程 schema（自动迁移 v1）；
  2. 基于素材哈希 + 参数指纹计算失效闭包；
  3. 闭包内重算、闭包外复用缓存，并记录每个阶段的来源与版本戳；
  4. 提交原子快照、产出修复 WAV 与逐段指标；
  5. 重启时按快照哈希校验恢复，禁止盲目复用旧缓存。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import audio_io, config, metrics as metrics_mod
from .cache import StageCache, compute_invalidation_closure
from .errors import E_CHANNEL_MISMATCH, RepairError
from .hashing import sha256_array
from .persistence import Workspace
from .pipeline import STAGE_ORDER, run_stage
from .project import load_and_normalize, project_fingerprint


class RepairEngine:
    def __init__(self, workspace: Workspace) -> None:
        self.ws = workspace
        self.cache: StageCache = workspace.cache

    # ---- 私有：定位需要重算的阶段闭包 ----
    def _plan_execution(
        self,
        content_hash: str,
        project_v2: dict,
        expected_channels: int,
        actual_channels: int,
    ) -> tuple[list[str], list[str], dict[str, str]]:
        if actual_channels != expected_channels:
            raise RepairError(
                E_CHANNEL_MISMATCH,
                "project channel plan does not match parsed audio channels",
                field="channels",
                context={"expected": expected_channels, "actual": actual_channels},
            )
        keys: dict[str, str] = {}
        upstream = None
        changed: list[str] = []
        for stage in STAGE_ORDER:
            key = self.cache.stage_key(stage, content_hash, project_v2, upstream)
            keys[stage] = key
            if not self.cache.has(key):
                changed.append(stage)
            upstream = key
        closure = compute_invalidation_closure(changed)
        return closure, changed, keys

    def execute(
        self,
        audio_path: str,
        project_data: dict[str, Any],
        segments: list[dict],
        expected_channels: int | None = None,
        output_name: str = "repaired.wav",
    ) -> dict[str, Any]:
        parsed = audio_io.parse_audio(audio_path, expected_channels=expected_channels)
        project_v2 = load_and_normalize(project_data)
        content_hash = parsed.content_hash()
        fp = project_fingerprint(project_v2)

        closure, missing, keys = self._plan_execution(
            content_hash, project_v2, parsed.channels, parsed.channels
        )
        closure_set = set(closure)

        # 全部命中缓存（闭包为空）时，必须强制校验既有快照与当前工程/素材/
        # 引擎版本戳严格一致：工作区目录被整体替换或复用了另一工程时，
        # 即使阶段键碰巧存在也禁止盲目复用（E_SNAPSHOT_STALE）。
        # 闭包非空时本就会重算并提交新快照，不依赖旧快照。
        prior_snapshot = self.ws.load_snapshot()
        if prior_snapshot is not None and not closure:
            self.ws.verify_snapshot_freshness(prior_snapshot, content_hash, fp)

        results: dict[str, np.ndarray] = {}
        trace: list[dict[str, Any]] = []
        stage_meta: list[dict[str, Any]] = []
        upstream_key = None
        for stage in STAGE_ORDER:
            key = keys[stage]
            if stage in closure_set:
                data = run_stage(
                    stage,
                    parsed.raw_float,
                    parsed.channels,
                    project_v2,
                    parsed.sample_rate,
                    results,
                )
                meta = self.cache.put(
                    key, stage, data, content_hash, upstream_key, project_v2
                )
                results[stage] = data
                trace.append(
                    {
                        "stage": stage,
                        "action": "recomputed",
                        "key": key,
                        "engine_version": config.ENGINE_VERSION,
                        "output_hash": meta["output_hash"],
                    }
                )
            else:
                data, meta = self.cache.get(key)
                results[stage] = data
                trace.append(
                    {
                        "stage": stage,
                        "action": "reused",
                        "key": key,
                        "engine_version": config.ENGINE_VERSION,
                        "source": "stage_cache",
                        "output_hash": meta["output_hash"],
                    }
                )
            stage_meta.append(
                {
                    "stage": stage,
                    "key": key,
                    "output_hash": meta["output_hash"],
                    "upstream_key": upstream_key,
                }
            )
            upstream_key = key

        # ---- 最终量化输出（16bit PCM；24bit 策略走对应子类型）----
        output_bit_depth = project_v2["bit_depth_policy"]["output_bit_depth"]
        mono_final = results["limit"]
        out_path = self.ws.output_path(output_name)
        audio_io.write_pcm_wav(out_path, mono_final, parsed.sample_rate, output_bit_depth)
        pcm16 = self._read_pcm16_view(mono_final)
        report_rows = metrics_mod.segment_metrics(mono_final, segments)

        snapshot = {
            "engine_version": config.ENGINE_VERSION,
            "content_hash": content_hash,
            "project_fingerprint": fp,
            "schema_version": project_v2["schema_version"],
            "audio_meta": parsed.to_meta(),
            "stages": stage_meta,
            "output": {
                "path": out_path,
                "bit_depth": output_bit_depth,
                "pcm16_sha256": sha256_array(pcm16),
                "float_sha256": sha256_array(mono_final),
            },
            "metrics": report_rows,
            "execution_trace": trace,
            "recomputed_closure": closure,
        }
        self.ws.commit_snapshot(snapshot)
        return snapshot

    @staticmethod
    def _read_pcm16_view(mono_float: np.ndarray) -> np.ndarray:
        max_val = float(2 ** 15 - 1)
        return np.round(np.clip(mono_float, -1.0, 1.0) * max_val).astype(np.int16)

    def restore(self, project_data: dict[str, Any]) -> dict[str, Any]:
        """强制终止后的恢复入口：只依据最近一次已提交快照。"""
        project_v2 = load_and_normalize(project_data)
        snapshot = self.ws.load_snapshot()
        if snapshot is None:
            raise RepairError(
                E_SNAPSHOT_STALE,
                "no committed snapshot exists; nothing to restore",
                field=config.SNAPSHOT_FILE,
            )
        fp = project_fingerprint(project_v2)
        content_hash = snapshot["content_hash"]
        self.ws.verify_snapshot_freshness(snapshot, content_hash, fp)
        live_keys = {entry["key"] for entry in snapshot["stages"]}
        audit = self.ws.recover(live_keys)
        self.ws.verify_stage_hashes(snapshot)
        return {"snapshot": snapshot, "recovery_audit": audit}

    def graceful_shutdown(self, snapshot: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """受控退出：完整清单 + 关闭标记 + 退出报告。"""
        manifest = {
            "stages": [e["stage"] for e in snapshot["stages"]],
            "project_fingerprint": snapshot["project_fingerprint"],
            "engine_version": config.ENGINE_VERSION,
        }
        self.ws.write_close_marker(manifest)
        report = {
            "exit_kind": "graceful",
            "engine_version": config.ENGINE_VERSION,
            "output": snapshot["output"],
            "recomputed_closure": snapshot["recomputed_closure"],
            "metrics_summary": {
                "segments": len(snapshot["metrics"]),
                "final_rms_db": [row["rms_db"] for row in snapshot["metrics"]],
            },
        }
        if extra:
            report.update(extra)
        self.ws.write_exit_report(report)
        return report
