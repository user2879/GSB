"""Shared in-memory data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class LayerTensor:
    name: str
    seed: Optional[int]
    dim: int
    rows: Dict[int, Tuple[float, ...]]
    reference_scale: Tuple[float, ...]
    reference_bias: Tuple[float, ...]
    source_line: int

    def ordered_rows(self) -> list[Tuple[int, Tuple[float, ...]]]:
        return sorted(self.rows.items())


@dataclass
class WeightRow:
    layer: str
    row: int
    scale: float
    bias: float
    scheme: str
    precision: str
    quant_bits: int
    source_version: str
    raw: Dict[str, Any] = field(default_factory=dict)
    sidecar: Dict[str, Any] = field(default_factory=dict)
    source_line: int = 0

    def source_payload(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "row": self.row,
            "scale": self.scale,
            "bias": self.bias,
            "scheme": self.scheme,
            "precision": self.precision,
            "quant_bits": self.quant_bits,
            "source_version": self.source_version,
            "raw": self.raw,
            "sidecar": self.sidecar,
        }

    def logical_payload(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "row": self.row,
            "scale": self.scale,
            "bias": self.bias,
            "scheme": self.scheme,
            "precision": self.precision,
            "quant_bits": self.quant_bits,
            "source_version": self.source_version,
        }

    def clone(self, **changes: Any) -> "WeightRow":
        values = {
            "layer": self.layer,
            "row": self.row,
            "scale": self.scale,
            "bias": self.bias,
            "scheme": self.scheme,
            "precision": self.precision,
            "quant_bits": self.quant_bits,
            "source_version": self.source_version,
            "raw": dict(self.raw),
            "sidecar": dict(self.sidecar),
            "source_line": self.source_line,
        }
        values.update(changes)
        return WeightRow(**values)


@dataclass
class SourceBundle:
    activation_lines: list[str]
    v1_lines: list[str]
    v2_values: list[Any]
    runtime_overrides: Dict[str, Any] = field(default_factory=dict)
    injection_id: Optional[str] = None


@dataclass
class LoadedSources:
    layers: Dict[str, LayerTensor]
    weights_v1: Dict[Tuple[str, int], WeightRow]
    weights_v2: Dict[Tuple[str, int], WeightRow]
    failures: list[Dict[str, Any]]
    load_fingerprint_parts: Dict[str, Any]

