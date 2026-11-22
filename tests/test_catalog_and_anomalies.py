from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path

import numpy as np
import pytest

from pageagg.canonical import sha256_bytes
from pageagg.catalog import canonical_catalog_text, load_catalog
from pageagg.errors import EBudget, ENullMap, EPage
from pageagg.page_format import HEADER
from pageagg.sample import generate_sample, inject_fault


def test_catalog_and_page_validation(tmp_path) -> None:
    sample = generate_sample(tmp_path / "sample", force=True)
    catalog_path = Path(sample["catalog_path"])
    raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    raw["pages"][1]["page_id"] = raw["pages"][0]["page_id"]
    catalog_path.write_text(canonical_catalog_text(raw), encoding="utf-8")
    with pytest.raises(EPage) as caught:
        load_catalog(catalog_path)
    assert caught.value.field == "page_id"
    assert caught.value.page_id == raw["pages"][0]["page_id"]


def test_missing_page_file_is_e_page(tmp_path) -> None:
    sample = generate_sample(tmp_path / "sample", force=True)
    catalog_path = Path(sample["catalog_path"])
    raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    (catalog_path.parent / raw["pages"][0]["file"]).unlink()
    with pytest.raises(EPage) as caught:
        load_catalog(catalog_path)
    assert caught.value.page_id == 1
    assert caught.value.field == "file"


def test_nonfinite_float_is_e_page(tmp_path) -> None:
    sample = generate_sample(tmp_path / "sample", force=True)
    catalog_path = Path(sample["catalog_path"])
    raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    entry = raw["pages"][0]
    page_path = catalog_path.parent / entry["file"]
    payload = bytearray(page_path.read_bytes())
    # First column is group_id; second is value_int; third is value_float.
    _, _, _, rows, ncols = HEADER.unpack_from(payload, 0)
    offset = HEADER.size
    for _ in range(ncols):
        name_len = struct.unpack_from("<H", payload, offset)[0]
        offset += 2
        name = payload[offset : offset + name_len].decode("utf-8")
        offset += name_len + 1
        if name == "value_float":
            payload[offset : offset + 8] = struct.pack("<d", float("inf"))
            break
        offset += rows * 8 + rows + 4
    page_path.write_bytes(payload)
    entry["content_sha256"] = sha256_bytes(bytes(payload))
    catalog_path.write_text(canonical_catalog_text(raw), encoding="utf-8")
    with pytest.raises(EPage) as caught:
        load_catalog(catalog_path)
    assert caught.value.page_id == 1
    assert caught.value.field == "columns[value_float].values"


def test_injected_catalog_truncation_is_e_page(tmp_path) -> None:
    sample = generate_sample(tmp_path / "sample", force=True)
    fault = inject_fault(sample["catalog_path"], tmp_path / "fault", "catalog_truncate")
    with pytest.raises(EPage) as caught:
        load_catalog(fault["catalog_path"])
    assert caught.value.field == "catalog.json"


def test_injected_nullmap_misalignment_is_e_nullmap(tmp_path) -> None:
    sample = generate_sample(tmp_path / "sample", force=True)
    fault = inject_fault(sample["catalog_path"], tmp_path / "fault", "nullmap_misalign")
    with pytest.raises(ENullMap) as caught:
        load_catalog(fault["catalog_path"])
    assert caught.value.code == "E_NULLMAP"
    assert caught.value.category == "E_PAGE"
    assert caught.value.page_id == 2
    assert "null_bitmap" in caught.value.field


def test_truncated_nullmap_is_e_nullmap(tmp_path) -> None:
    sample = generate_sample(tmp_path / "sample", force=True)
    catalog_path = Path(sample["catalog_path"])
    raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    entry = raw["pages"][0]
    page_path = catalog_path.parent / entry["file"]
    payload = page_path.read_bytes()
    _, _, _, rows, _ = HEADER.unpack_from(payload, 0)
    offset = HEADER.size
    name_len = struct.unpack_from("<H", payload, offset)[0]
    offset += 2 + name_len + 1
    bitmap_start = offset + rows * 8
    page_path.write_bytes(payload[: bitmap_start + 5])
    with pytest.raises(ENullMap):
        load_catalog(catalog_path)
