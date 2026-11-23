"""Deterministic edit commands applied to the active exchange workspace."""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import PipelineConfig
from .errors import FailureLog
from .model import Segment
from .pipeline import load_exchange
from .v1 import write_v1
from .v2 import write_v2


@dataclass
class Workspace:
    directory: Path
    config: PipelineConfig
    version: str = "v1"

    @property
    def exchange_path(self) -> Path:
        suffix = "txt" if self.version == "v1" else "json"
        return self.directory / f"active.{self.version}.{suffix}"

    def load(self) -> list[Segment]:
        failures = FailureLog()
        return load_exchange(self.exchange_path, self.version,
                             self.config, failures)

    def save(self, segments: list[Segment]) -> None:
        if self.version == "v1":
            write_v1(self.exchange_path, segments, self.config)
        else:
            write_v2(self.exchange_path, segments)

    def apply(self, command: dict) -> None:
        name = command["op"]
        segments = self.load()
        if name == "switch_version":
            self._switch(command["version"], segments)
            return
        if name == "set_quant_levels":
            # Quant level lives in the config; exchange content is unchanged.
            self.config = PipelineConfig.from_dict(
                {**self.config.to_dict(),
                 "quant_levels": int(command["levels"])})
            self._write_config()
            return
        index = int(command["segment"])
        if name == "set_boundary":
            for seg in segments:
                if seg.index == index:
                    seg.start_frame = float(command["start_frame"])
                    seg.end_frame = float(command["end_frame"])
        elif name == "set_intensity":
            value = float(command["intensity"])
            for seg in segments:
                if seg.index == index:
                    seg.intensity = value
                    seg.intensity_permille = int(round(value * 1000.0))
        elif name == "delete_segment":
            segments = [s for s in segments if s.index != index]
        elif name == "append_segment":
            segments.append(Segment(
                index=index,
                start_frame=float(command["start_frame"]),
                end_frame=float(command["end_frame"]),
                intensity=float(command["intensity"]),
                timbre=str(command["timbre"]),
                source_version=self.version,
                intensity_permille=int(round(
                    float(command["intensity"]) * 1000.0)),
                timbre_code=self.config.name_to_code(command["timbre"]),
            ))
        else:
            raise ValueError(f"unknown edit op {name!r}")
        self.save(segments)

    def _switch(self, target: str, segments: list[Segment]) -> None:
        if target == self.version:
            return
        if target == "v2":
            from .convert import upgrade_v1_to_v2
            upgraded = upgrade_v1_to_v2(segments, self.config)
            self.version = "v2"
            self.save(upgraded)
        else:
            from .convert import downgrade_v2_to_v1, write_sidecar
            downgraded, sidecar = downgrade_v2_to_v1(segments, self.config)
            self.version = "v1"
            self.save(downgraded)
            sidecar_path = self.directory / "active.sidecar.json"
            write_sidecar(sidecar_path, sidecar)

    def _write_config(self) -> None:
        path = self.directory / "acoustic.json"
        path.write_text(
            json.dumps(self.config.to_dict(), sort_keys=True,
                       separators=(",", ":")) + "\n",
            encoding="utf-8")

    def reset_to(self, source_exchange: Path, source_config: Path,
                 version: str) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        shutil.copy(source_config, self.directory / "acoustic.json")
        self.config = PipelineConfig.load(self.directory / "acoustic.json")
        self.version = version
        shutil.copy(source_exchange, self.exchange_path)
