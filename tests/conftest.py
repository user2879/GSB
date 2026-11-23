from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from quant_workbench.samples import generate_samples


@pytest.fixture()
def sample_paths(tmp_path: Path) -> dict[str, Path]:
    return generate_samples(tmp_path / "samples")


def write_jsonl(path: Path, records: list[Any]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, separators=(",", ":"), allow_nan=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
