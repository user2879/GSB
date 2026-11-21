"""Fault-injection isolation: apply real input faults to a copied dataset and
map each injection id to its expected business code in the report."""

import json
import shutil
from pathlib import Path

from .errors import (
    E_BOX_OUT_OF_RANGE,
    E_LABEL_ENUM,
    E_ORDER_GAP,
    E_SAMPLE_INVALID,
)
from .pipeline import run_pipeline

FAULT_TO_CODE = {
    "truncate_png": E_SAMPLE_INVALID,
    "desc_category_enum": E_LABEL_ENUM,
    "label_order_gap": E_ORDER_GAP,
    "box_out_of_range": E_BOX_OUT_OF_RANGE,
}

LABEL_FILES = {"label_a": "labels_a.jsonl", "label_b": "labels_b.jsonl"}


def _apply_one(spec, data_dir):
    fault = spec["fault"]
    params = spec.get("params", {})
    sample_id = spec["sample_id"]

    if fault == "truncate_png":
        path = data_dir / "images" / f"{sample_id}.png"
        raw = path.read_bytes()
        path.write_bytes(raw[: params["keep_bytes"]])
        return

    if fault == "desc_category_enum":
        path = data_dir / "descriptions" / f"{sample_id}.json"
        desc = json.loads(path.read_text(encoding="utf-8"))
        desc["assertions"][params["index"]]["category"] = params["value"]
        path.write_text(json.dumps(desc, ensure_ascii=False, indent=1), encoding="utf-8")
        return

    if fault == "label_order_gap":
        path = data_dir / LABEL_FILES[spec["source"]]
        entries = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        owned = [e for e in entries if e.get("sample_id") == sample_id]
        owned.sort(key=lambda e: e["order"])
        target = owned[params["entry_index"]]
        for e in entries:
            if e is target or (
                e.get("sample_id") == sample_id and e.get("order") == target["order"]
            ):
                e["order"] = e["order"] + params.get("skip", 2)
                break
        path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries),
                        encoding="utf-8")
        return

    if fault == "box_out_of_range":
        path = data_dir / LABEL_FILES[spec["source"]]
        entries = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        owned = [e for e in entries if e.get("sample_id") == sample_id]
        owned.sort(key=lambda e: e["order"])
        target = owned[params["entry_index"]]
        for e in entries:
            if e.get("sample_id") == sample_id and e.get("order") == target["order"]:
                e["bbox"][2] = e["bbox"][2] + params.get("dx", 500)
                break
        path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries),
                        encoding="utf-8")
        return

    raise ValueError(f"unknown fault: {fault}")


def run_injections(config_path, data_dir, injections_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = json.loads(Path(injections_path).read_text(encoding="utf-8"))["injections"]

    work = out_dir / "injected_data"
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(data_dir, work)
    for junk in ("edits.json", "injections.json"):
        junk_path = work / junk
        if junk_path.exists():
            junk_path.unlink()

    for spec in specs:
        _apply_one(spec, work)

    report = run_pipeline(config_path, work, out_dir)
    hits = []
    for spec in specs:
        code = FAULT_TO_CODE[spec["fault"]]
        hit = any(
            f["sample_id"] == spec["sample_id"] and f["code"] == code
            for f in report["failures"]
        )
        hits.append(
            {
                "id": spec["id"],
                "sample_id": spec["sample_id"],
                "source": spec["source"],
                "fault": spec["fault"],
                "expected_code": code,
                "hit": hit,
            }
        )
    report["injections"] = hits
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return report
