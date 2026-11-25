from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path

import pytest

from voice_diag.synth import generate_demo


@pytest.fixture()
def demo(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    generate_demo(corpus, seed=20261126)
    return corpus


def update_config(corpus: Path, mutate) -> dict:
    path = corpus / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    mutate(config)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return config


def append_label(corpus: Path, label: dict) -> None:
    with (corpus / "labels.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(label, ensure_ascii=False, sort_keys=True) + "\n")


def write_float_nan_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = struct.pack("<f", float("nan"))
    fmt = struct.pack("<HHIIHH", 3, 1, 16000, 64000, 4, 32)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(payload)) + payload
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


def copy_corpus(source: Path, target: Path) -> Path:
    shutil.copytree(source, target)
    return target
