from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from workbench.config import AlignmentConfig
from workbench.generator import generate_dataset
from workbench.injections import InjectionSet
from workbench.pipeline import Workbench


@dataclass
class Workspace:
    root: Path
    samples: Path
    config_dir: Path
    config_path: Path
    injections_path: Path

    def load_config(self) -> AlignmentConfig:
        return AlignmentConfig.load(self.config_path)

    def run(
        self,
        *,
        mode: str = "incremental",
        output_name: str = "output",
        injections: str | None = "injections.json",
        config: AlignmentConfig | None = None,
    ) -> dict[str, Any]:
        injection_set = (
            InjectionSet.load(self.config_dir / injections)
            if injections is not None
            else InjectionSet.empty()
        )
        workbench = Workbench(
            samples_dir=self.samples,
            config=config or self.load_config(),
            injections=injection_set,
            output_dir=self.root / output_name,
            mode=mode,
        )
        return workbench.run()

    def edit_config(self, **updates: Any) -> None:
        raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        raw.update(updates)
        self.config_path.write_text(
            json.dumps(raw, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    samples = tmp_path / "samples"
    config_dir = tmp_path / "config"
    generate_dataset(samples, config_dir, seed=20261122, sample_count=8)
    return Workspace(
        root=tmp_path,
        samples=samples,
        config_dir=config_dir,
        config_path=config_dir / "alignment.json",
        injections_path=config_dir / "injections.json",
    )
