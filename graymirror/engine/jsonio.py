"""JSON helpers that keep exact rational values as explicit strings."""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any


class RationalEncoder(json.JSONEncoder):
    def default(self, value: Any) -> Any:
        if isinstance(value, Fraction):
            return str(value)
        if isinstance(value, Path):
            return str(value)
        return super().default(value)


def dumps(payload: Any, **kwargs: Any) -> str:
    return json.dumps(payload, cls=RationalEncoder, ensure_ascii=False, **kwargs)


def dump_file(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
