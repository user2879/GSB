from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .rope import PositionEncoding


@dataclass(frozen=True)
class EncodedSample:
    scheme: str
    sample_id: str
    query_encoding: PositionEncoding
    key_encodings: tuple[PositionEncoding, ...]
    query_rotated: np.ndarray
    key_rotated: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheme": self.scheme,
            "sample_id": self.sample_id,
            "query_encoding": self.query_encoding.to_dict(),
            "key_encodings": [encoding.to_dict() for encoding in self.key_encodings],
            "query_rotated": [float(value) for value in self.query_rotated.tolist()],
            "key_rotated": [
                [float(value) for value in row] for row in self.key_rotated.tolist()
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EncodedSample":
        return cls(
            scheme=str(payload["scheme"]),
            sample_id=str(payload["sample_id"]),
            query_encoding=PositionEncoding.from_dict(payload["query_encoding"]),
            key_encodings=tuple(
                PositionEncoding.from_dict(item) for item in payload["key_encodings"]
            ),
            query_rotated=np.asarray(payload["query_rotated"], dtype=np.float64),
            key_rotated=np.asarray(payload["key_rotated"], dtype=np.float64),
        )

