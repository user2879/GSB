"""Frame engine orchestrating depth, Hi-Z, culling, degradation and streaming."""
from __future__ import annotations

import math

import numpy as np

from .config import DEFAULT_CONFIG, WORLD_EXTENT
from .depth import DepthRenderer
from .culling import OcclusionCuller
from .streaming import ResidencyLedger, StreamingScheduler
from .budget import DegradationPolicy, ErrorBookkeeper, base_expansion_margin
from .errors import BlockMissingError


class CullingEngine:
    def __init__(self, dataset, camera, config=DEFAULT_CONFIG):
        self.dataset = dataset
        self.camera = camera
        self.config = config
        self.renderer = DepthRenderer(
            config.depth_width, config.depth_height,
            config.raymarch_steps, config.max_hiz_levels,
            config.max_depth_bias)
        self.culler = OcclusionCuller(camera, config)
        self.ledger = ResidencyLedger(
            dataset, config.memory_budget_bytes, config.vram_budget_bytes)
        self.scheduler = StreamingScheduler(dataset, self.ledger)
        self.policy = DegradationPolicy(cap=config.degradation_cap)
        self.bookkeeper = ErrorBookkeeper(config.decision_budget)
        self.leaves = sorted(dataset.leaf_blocks(), key=lambda b: b.block_id)
        self.cell_world = WORLD_EXTENT / dataset.size
        self.hiz_infos = None

    def _preload_ids(self, visible_ids: list[str], rings: int) -> list[str]:
        if rings <= 0:
            return list(visible_ids)
        keep = set(visible_ids)
        index = {b.block_id: b for b in self.leaves}
        grid = int(math.sqrt(len(self.leaves)))
        coords = {(b.bx, b.by): b.block_id for b in self.leaves}
        seeds = [index[bid] for bid in visible_ids if bid in index]
        for block in seeds:
            for dx in range(-rings, rings + 1):
                for dy in range(-rings, rings + 1):
                    key = (block.bx + dx, block.by + dy)
                    if key in coords:
                        keep.add(coords[key])
        return sorted(keep)

    def run_frame(self, frame_index: int, camera_frame) -> dict:
        # Depth uses last frame's resident set; frame 0 starts with an empty
        # occlusion buffer so nothing can be wrongly culled on the first frame.
        resident_before = set(self.ledger.resident)
        depth = self.renderer.render_depth(self.dataset, camera_frame, resident_before)
        hiz_levels, hiz_infos = self.renderer.build_hiz(
            depth, self.config.fov_rad)
        self.hiz_infos = hiz_infos

        state = self.policy.current()
        base_margin = base_expansion_margin()
        margin = base_margin * state.expansion_multiplier
        hiz_level = min(state.hiz_level_offset, len(hiz_infos) - 1)
        self.renderer.check_level(hiz_level, hiz_infos)

        decisions = []
        for block in self.leaves:
            decisions.append(self.culler.cull_block(
                block, camera_frame, hiz_levels, hiz_infos, margin, hiz_level,
                heights=self.dataset.heights))

        visible = [d for d in decisions if d.state in ("visible", "uncertain")]
        visible_ids = sorted(d.block_id for d in visible)
        uncertain = [d for d in decisions if d.state == "uncertain"]
        occluded = [d for d in decisions if d.state == "occluded"]
        frustum_out = [d for d in decisions if d.state == "frustum_out"]

        in_frustum = [d for d in decisions if d.state != "frustum_out"]
        error_summary = self.bookkeeper.summarize(in_frustum)
        max_util = max(self.ledger.memory_used / max(self.config.memory_budget_bytes, 1),
                       self.ledger.vram_used / max(self.config.vram_budget_bytes, 1))
        self.policy.update(frame_index, max_util, error_summary["over_budget_count"])
        active_state = self.policy.current()
        if active_state.step != state.step:
            state = active_state
            margin = base_margin * state.expansion_multiplier
            hiz_level = min(state.hiz_level_offset, len(hiz_infos) - 1)

        desired = self._preload_ids(visible_ids, state.preload_rings)
        protected = list(visible_ids)

        # First attempt with preload; on pressure drop preload before touching
        # visible blocks, guaranteeing visible blocks are never starved.
        stream_report = None
        try:
            stream_report = self.scheduler.plan(frame_index, desired, protected)
        except Exception:
            minimal = self._preload_ids(visible_ids, 0)
            stream_report = self.scheduler.plan(frame_index, minimal, protected)
            if not all(self.ledger.contains(bid) for bid in visible_ids):
                raise

        self.ledger.verify()
        counts = {"visible": len(visible), "occluded": len(occluded),
                  "uncertain": len(uncertain), "frustum_out": len(frustum_out),
                  "total_leaves": len(self.leaves)}
        return {
            "frame": frame_index,
            "camera": camera_frame.to_param_dict(),
            "counts": counts,
            "decisions": [d.to_dict() for d in decisions],
            "error_summary": error_summary,
            "hiz": {
                "levels": [info.to_dict() for info in hiz_infos],
                "selected_level": hiz_level,
                "expansion_margin": margin,
                "base_expansion_margin": base_margin,
            },
            "degradation": {
                "step": state.step,
                "state": state.to_dict(),
                "sequence": list(self.policy.sequence_log),
            },
            "streaming": stream_report,
            "budget": self.ledger.usage(),
        }
