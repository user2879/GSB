from __future__ import annotations

import shutil
from pathlib import Path
from typing import Mapping

from PIL import Image

from image_edit_workbench.config import WorkbenchConfig
from image_edit_workbench.exchange import materialize_logical_state
from image_edit_workbench.models import ImageInfo, RegionEdit
from image_edit_workbench.runner import ScenarioState, cold_render_exchange


def cold_render(
    regions: list[RegionEdit],
    *,
    config: WorkbenchConfig,
    images: Mapping[str, ImageInfo],
    work_root: Path,
    version: int = 2,
    drain: bool = True,
):
    shutil.rmtree(work_root, ignore_errors=True)
    exchange = materialize_logical_state(
        regions,
        version=version,
        images=images,
        manifest_keys={region.region_key for region in regions},
    )
    state = ScenarioState(regions=list(exchange.regions), config=config, exchange_version=version)
    return cold_render_exchange(
        exchange=exchange,
        state=state,
        images=images,
        config=config,
        state_dir=work_root / "state",
        output_dir=work_root / "outputs",
        drain=drain,
    )


def read_rgb(path: Path):
    with Image.open(path) as image:
        return image.convert("RGB").tobytes()

