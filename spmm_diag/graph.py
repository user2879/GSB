from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import SpMMConfig
from .csr import CSRMatrix
from .fingerprint import array_bytes, digest_bytes, digest_json


@dataclass(frozen=True)
class BlockDependency:
    block_id: int
    row_start: int
    row_end: int
    a_rows: tuple[int, ...]
    b_rows: tuple[int, ...]
    a_row_fingerprints: tuple[str, ...]
    b_row_fingerprints: tuple[str, ...]
    c_row_fingerprints: tuple[str, ...]
    coefficient_fingerprint: str
    precision_fingerprint: str
    layout_fingerprint: str
    input_fingerprint: str
    output_fingerprint: str | None = None

    def with_output(self, output_fingerprint: str) -> "BlockDependency":
        return BlockDependency(
            block_id=self.block_id,
            row_start=self.row_start,
            row_end=self.row_end,
            a_rows=self.a_rows,
            b_rows=self.b_rows,
            a_row_fingerprints=self.a_row_fingerprints,
            b_row_fingerprints=self.b_row_fingerprints,
            c_row_fingerprints=self.c_row_fingerprints,
            coefficient_fingerprint=self.coefficient_fingerprint,
            precision_fingerprint=self.precision_fingerprint,
            layout_fingerprint=self.layout_fingerprint,
            input_fingerprint=self.input_fingerprint,
            output_fingerprint=output_fingerprint,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "row_start": self.row_start,
            "row_end": self.row_end,
            "a_rows": list(self.a_rows),
            "b_rows": list(self.b_rows),
            "a_row_fingerprints": list(self.a_row_fingerprints),
            "b_row_fingerprints": list(self.b_row_fingerprints),
            "c_row_fingerprints": list(self.c_row_fingerprints),
            "coefficient_fingerprint": self.coefficient_fingerprint,
            "precision_fingerprint": self.precision_fingerprint,
            "layout_fingerprint": self.layout_fingerprint,
            "input_fingerprint": self.input_fingerprint,
            "output_fingerprint": self.output_fingerprint,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BlockDependency":
        return cls(
            block_id=int(value["block_id"]),
            row_start=int(value["row_start"]),
            row_end=int(value["row_end"]),
            a_rows=tuple(int(item) for item in value["a_rows"]),
            b_rows=tuple(int(item) for item in value["b_rows"]),
            a_row_fingerprints=tuple(value["a_row_fingerprints"]),
            b_row_fingerprints=tuple(value["b_row_fingerprints"]),
            c_row_fingerprints=tuple(value["c_row_fingerprints"]),
            coefficient_fingerprint=str(value["coefficient_fingerprint"]),
            precision_fingerprint=str(value["precision_fingerprint"]),
            layout_fingerprint=str(value["layout_fingerprint"]),
            input_fingerprint=str(value["input_fingerprint"]),
            output_fingerprint=value.get("output_fingerprint"),
        )


def derive_dependency(
    csr: CSRMatrix,
    b_matrix: np.ndarray,
    c_matrix: np.ndarray,
    config: SpMMConfig,
    block_id: int,
    row_start: int,
    row_end: int,
) -> BlockDependency:
    a_rows = tuple(range(row_start, row_end))
    b_rows = tuple(
        sorted(
            {
                int(column)
                for row in a_rows
                for column in csr.col_indices[
                    int(csr.row_offsets[row]) : int(csr.row_offsets[row + 1])
                ]
            }
        )
    )
    a_row_fingerprints = tuple(csr.row_fingerprint(row) for row in a_rows)
    b_row_fingerprints = tuple(
        digest_bytes(array_bytes(np.asarray(b_matrix[row], dtype="<f8")))
        for row in b_rows
    )
    c_row_fingerprints = tuple(
        digest_bytes(array_bytes(np.asarray(c_matrix[row], dtype="<f8")))
        for row in a_rows
    )
    coefficient_fingerprint = config.coefficient_fingerprint()
    precision_fingerprint = config.precision_fingerprint()
    layout_fingerprint = config.layout_fingerprint()
    input_fingerprint = digest_json(
        {
            "a_rows": [
                {"row": row, "fingerprint": fingerprint}
                for row, fingerprint in zip(a_rows, a_row_fingerprints)
            ],
            "b_rows": [
                {"row": row, "fingerprint": fingerprint}
                for row, fingerprint in zip(b_rows, b_row_fingerprints)
            ],
            "c_rows": [
                {"row": row, "fingerprint": fingerprint}
                for row, fingerprint in zip(a_rows, c_row_fingerprints)
            ],
            "alpha_beta": coefficient_fingerprint,
            "precision": precision_fingerprint,
            "layout": layout_fingerprint,
            "order": config.order,
        }
    )
    return BlockDependency(
        block_id=block_id,
        row_start=row_start,
        row_end=row_end,
        a_rows=a_rows,
        b_rows=b_rows,
        a_row_fingerprints=a_row_fingerprints,
        b_row_fingerprints=b_row_fingerprints,
        c_row_fingerprints=c_row_fingerprints,
        coefficient_fingerprint=coefficient_fingerprint,
        precision_fingerprint=precision_fingerprint,
        layout_fingerprint=layout_fingerprint,
        input_fingerprint=input_fingerprint,
    )


def dependency_fingerprint(dependency: BlockDependency) -> str:
    return dependency.input_fingerprint
