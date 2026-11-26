from __future__ import annotations

import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

from renderdiag.config import Config, load_config
from renderdiag.engine import RenderEngine
from renderdiag.samples import generate_sample_workspace


class SampleWorkspace:
    def __init__(self) -> None:
        self._temp = tempfile.TemporaryDirectory(prefix="renderdiag-test-")
        self.root = Path(self._temp.name)
        self.paths = generate_sample_workspace(self.root / "sample")
        self.config: Config = load_config(self.paths["config"])

    def close(self) -> None:
        self._temp.cleanup()

    def engine(
        self,
        name: str,
        *,
        config: Config | None = None,
        max_concurrency: int | None = None,
        batch_point_budget: int | None = None,
        injections=(),
        prompt_name: str = "p0",
        image_path=...,
        motion_path=None,
    ) -> RenderEngine:
        active = config or self.config
        overrides = {}
        if max_concurrency is not None:
            overrides["max_concurrency"] = max_concurrency
        if batch_point_budget is not None:
            overrides["batch_point_budget"] = batch_point_budget
        if overrides:
            active = replace(active, **overrides)
        kwargs = {}
        if image_path is not ...:
            kwargs["image_path"] = image_path
        if motion_path is not None:
            kwargs["motion_path"] = motion_path
        return RenderEngine(
            active,
            self.root / name,
            store_root=self.root / f"{name}-store",
            output_dir=self.root / f"{name}-output",
            injections=injections,
            prompt_name=prompt_name,
            **kwargs,
        )


def remove_tree(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
