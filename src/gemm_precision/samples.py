from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .model import SampleSpec
from .util import array_sha256, payload_sha256


@dataclass(frozen=True)
class SampleData:
    spec: SampleSpec
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    special_positions: tuple[dict[str, Any], ...]
    generation: dict[str, Any]

    @property
    def digest(self) -> str:
        return array_sha256(self.A, self.B, self.C)

    def metadata(self) -> dict[str, Any]:
        return {
            **self.spec.as_dict(),
            "is_bmm": self.spec.is_bmm,
            "arrays": {
                "A": list(self.A.shape),
                "B": list(self.B.shape),
                "C": list(self.C.shape),
            },
            "dtypes": {"A": str(self.A.dtype), "B": str(self.B.dtype), "C": str(self.C.dtype)},
            "digest_sha256": self.digest,
            "special_positions": [dict(item) for item in self.special_positions],
            "generation": dict(self.generation),
        }


def _float32_bits(value: float) -> str:
    return f"0x{struct.unpack('<I', struct.pack('<f', float(np.float32(value))))[0]:08x}"


def _unravel(index: int, shape: tuple[int, ...]) -> list[int]:
    return [int(item) for item in np.unravel_index(index, shape)]


def _apply_special_values(
    array: np.ndarray,
    matrix_name: str,
    rng: np.random.Generator,
    fraction: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    total = int(array.size)
    count = int(round(total * fraction))
    if fraction > 0.0 and count == 0 and total > 0:
        count = 1
    if count <= 0:
        return array, []
    positions = np.sort(rng.choice(total, size=min(count, total), replace=False))
    subnormal_positive = np.nextafter(np.float32(0.0), np.float32(1.0), dtype=np.float32)
    subnormal_negative = np.float32(-float(subnormal_positive))
    records: list[dict[str, Any]] = []
    flat = array.ravel()
    for ordinal, flat_index in enumerate(positions.tolist()):
        if ordinal % 2 == 0:
            kind = "zero"
            value = np.float32(0.0)
        elif ordinal % 4 == 1:
            kind = "positive_subnormal"
            value = subnormal_positive
        else:
            kind = "negative_subnormal"
            value = subnormal_negative
        flat[flat_index] = value
        coordinate = _unravel(flat_index, array.shape)
        records.append(
            {
                "matrix": matrix_name,
                "flat_index": int(flat_index),
                "index": coordinate,
                "kind": kind,
                "value": float(value),
                "float32_bits": _float32_bits(value),
                "selection_ordinal": ordinal,
            }
        )
    return flat.reshape(array.shape), records


def generate_sample(spec: SampleSpec) -> SampleData:
    spec.validate()
    rng = np.random.default_rng(int(spec.seed))
    if spec.is_bmm:
        a_shape = (spec.batch_size, spec.M, spec.K)
        b_shape = (spec.batch_size, spec.K, spec.N)
        c_shape = (spec.batch_size, spec.M, spec.N)
    else:
        a_shape = (spec.M, spec.K)
        b_shape = (spec.K, spec.N)
        c_shape = (spec.M, spec.N)

    A = rng.uniform(-1.0, 1.0, size=a_shape).astype(np.float32)
    B = rng.uniform(-1.0, 1.0, size=b_shape).astype(np.float32)
    C = rng.uniform(-0.5, 0.5, size=c_shape).astype(np.float32)

    special_positions: list[dict[str, Any]] = []
    if spec.category == "special_values" or spec.special_fraction > 0.0:
        if spec.category == "special_values":
            A, records_a = _apply_special_values(A, "A", rng, spec.special_fraction)
            B, records_b = _apply_special_values(B, "B", rng, spec.special_fraction)
            special_positions.extend(records_a)
            special_positions.extend(records_b)

    generation = {
        "generator": "numpy.random.Generator(PCG64)",
        "seed": int(spec.seed),
        "distribution": {
            "A": "uniform[-1,1)->float32",
            "B": "uniform[-1,1)->float32",
            "C": "uniform[-0.5,0.5)->float32",
        },
        "special_selection": {
            "policy": "rng.choice(flat_size, count, replace=False), sorted by flat index",
            "kind_policy": "even ordinal=+0.0; ordinal%4==1=+smallest_subnormal; otherwise=-smallest_subnormal",
            "smallest_positive_float32_subnormal": float(
                np.nextafter(np.float32(0.0), np.float32(1.0), dtype=np.float32)
            ),
            "fraction": float(spec.special_fraction),
            "position_count": len(special_positions),
        },
        "sample_payload_digest": payload_sha256(
            {
                "spec": spec.as_dict(),
                "special_positions": special_positions,
                "arrays_digest": array_sha256(A, B, C),
            }
        ),
    }
    return SampleData(
        spec=spec,
        A=A,
        B=B,
        C=C,
        special_positions=tuple(special_positions),
        generation=generation,
    )


def generate_samples(specs: tuple[SampleSpec, ...]) -> dict[str, SampleData]:
    return {spec.name: generate_sample(spec) for spec in specs}


def save_sample_npz(sample: SampleData, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        A=sample.A,
        B=sample.B,
        C=sample.C,
        metadata=np.asarray([payload_sha256(sample.metadata())], dtype="U64"),
    )
