"""端到端渲染管线：恢复、并发渲染、持久化、稳定性评估与原子输出。"""
from __future__ import annotations

import threading
from dataclasses import dataclass

from .assembly import (
    assemble_supersampled,
    build_report,
    downsample,
    quantize_grayscale,
    write_pgm,
    write_report,
)
from .errors import E_INTERRUPTED, RenderError
from .persistence import (
    compute_version_id,
    interrupt_marker_exists,
    load_tile,
    remove_interrupt_marker,
    save_tile,
    write_version,
)
from .renderer import TileResult
from .scheduler import TileScheduler, build_tile_specs
from .scene import Scene, load_scene
from .stability import evaluate_stability

IMAGE_NAME = "grayscale.pgm"
REPORT_NAME = "report.json"


@dataclass
class RunOutcome:
    status: str                 # completed | resumed | interrupted
    version_id: str
    image_path: str | None
    report_path: str | None
    rendered_tiles: int
    restored_tiles: int
    total_tiles: int


class RenderPipeline:
    def __init__(self, scene: Scene, interrupted: threading.Event | None = None):
        self.scene = scene
        self.interrupted = interrupted or threading.Event()
        self.version_id = compute_version_id(scene)
        self.cfg = scene.config

    def _check_interrupt(self) -> None:
        if self.interrupted.is_set() or interrupt_marker_exists(self.cfg.state_dir):
            self.interrupted.set()
            raise RenderError(E_INTERRUPTED, "在分块边界收到中断标记",
                              {"state_dir": self.cfg.state_dir})

    def execute(self, run_stability: bool = True) -> RunOutcome:
        # 开工前的分块边界检查：预置中断时不渲染任何块。
        self._check_interrupt()
        write_version(self.cfg.state_dir, self.version_id)
        specs = build_tile_specs(self.scene)
        all_indices = {s.index for s in specs}

        # 恢复：仅信任版本一致且摘要有效的已持久化分块。
        restored: list[TileResult] = []
        for index in sorted(all_indices):
            tile = load_tile(self.cfg.state_dir, index, self.version_id)
            if tile is not None:
                restored.append(tile)
        restored_set = {t.tile_index for t in restored}
        missing = sorted(all_indices - restored_set)

        scheduler = TileScheduler(self.scene, self.interrupted)
        fresh: list[TileResult] = []
        interrupted = False
        try:
            for tile in scheduler.run(skip=restored_set):
                self._check_interrupt()
                save_tile(self.cfg.state_dir, tile, self.version_id)
                fresh.append(tile)
                self._check_interrupt()
        except RenderError as exc:
            if exc.code != E_INTERRUPTED:
                raise
            interrupted = True

        if interrupted:
            return RunOutcome(
                status="interrupted", version_id=self.version_id,
                image_path=None, report_path=None,
                rendered_tiles=len(fresh), restored_tiles=len(restored),
                total_tiles=len(specs),
            )

        tiles = restored + fresh
        if {t.tile_index for t in tiles} != all_indices:
            # 理论上不可达；防御性处理，避免拼装半成品。
            raise RenderError(E_INTERRUPTED, "分块不完整，拒绝拼装输出",
                              {"have": sorted(t.tile_index for t in tiles),
                               "need": sorted(all_indices)})

        grid, _, _, _ = assemble_supersampled(tiles, self.cfg)
        pixel_radiance = downsample(grid, self.cfg.supersample)
        image, peak = quantize_grayscale(pixel_radiance)

        stability = evaluate_stability(self.scene, pixel_radiance) if run_stability else None
        report = build_report(tiles, self.cfg, image, peak, self.version_id, stability)

        image_path = f"{self.cfg.output_dir}/{IMAGE_NAME}"
        report_path = f"{self.cfg.output_dir}/{REPORT_NAME}"
        write_pgm(image_path, image)
        write_report(report_path, report)
        remove_interrupt_marker(self.cfg.state_dir)

        return RunOutcome(
            status="resumed" if restored else "completed",
            version_id=self.version_id, image_path=image_path,
            report_path=report_path, rendered_tiles=len(fresh),
            restored_tiles=len(restored), total_tiles=len(specs),
        )


def run_scene_file(path: str, interrupted: threading.Event | None = None,
                   run_stability: bool = True) -> RunOutcome:
    scene = load_scene(path)
    return RenderPipeline(scene, interrupted).execute(run_stability=run_stability)
