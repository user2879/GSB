from __future__ import annotations

import json
import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any

from PIL import Image

from .glyphs import GLYPH_ORDER, render_glyph_image
from .ledger import atomic_write_json


PAGE_W = 720
PAGE_H = 960
BLOCK = 72
X_CENTERS = [630, 470, 310, 150]
Y_START = 92
Y_PITCH = 118
DEFAULT_BLOCKS = 6


def default_plan() -> dict[str, Any]:
    pages = []
    stable_text = [char for char in GLYPH_ORDER if char not in {"一", "二", "三", "川"}]
    text = stable_text * 10
    char_cursor = 0
    for image_no in range(1, 4):
        columns = []
        for col_index, x_center in enumerate(X_CENTERS[:3]):
            blocks = []
            for i in range(DEFAULT_BLOCKS):
                y = Y_START + i * Y_PITCH
                char = text[char_cursor % len(text)]
                char_cursor += 1
                blocks.append({
                    "block_id": f"p{image_no}-c{col_index}-b{i}",
                    "char": char,
                    "x_center": x_center,
                    "y": y,
                    "bbox": [x_center - BLOCK // 2, y, x_center + BLOCK // 2, y + BLOCK],
                    "keypoint": [x_center, y + BLOCK // 2],
                })
            columns.append({
                "column_id": f"p{image_no}-col-{col_index}",
                "x_center": x_center,
                "bbox": [x_center - 48, 72, x_center + 48, 790],
                "blocks": blocks,
            })
        pages.append({
            "image_no": image_no,
            "name": f"page_{image_no:03d}.png",
            "seed": 319000 + image_no * 97,
            "columns": columns,
        })
    plan = {
        "schema": "vertical-layout-plan/v1",
        "seed": 319,
        "global_version": 1,
        "width": PAGE_W,
        "height": PAGE_H,
        "mode": "L",
        "pages": pages,
    }
    renumber_plan(plan)
    return plan


def renumber_plan(plan: dict[str, Any]) -> None:
    for page in plan["pages"]:
        columns = sorted(page["columns"], key=lambda c: -c["x_center"])
        reading_no = 1
        for col_index, column in enumerate(columns):
            column["column_id"] = column.get("column_id") or f"{page['name']}-col-{col_index}"
            column["rtl_index"] = col_index
            column["bbox"] = [column["x_center"] - 48, 72, column["x_center"] + 48, 790]
            blocks = sorted(column["blocks"], key=lambda b: b["y"])
            column["blocks"] = blocks
            for i, block in enumerate(blocks):
                x = block.get("x_center", column["x_center"])
                block["x_center"] = x
                block["bbox"] = [x - BLOCK // 2, block["y"], x + BLOCK // 2, block["y"] + BLOCK]
                block["keypoint"] = [x, block["y"] + BLOCK // 2]
                block["reading_no"] = reading_no
                block["top_down_index"] = i
                reading_no += 1


def write_plan(root: str | Path, plan: dict[str, Any]) -> Path:
    path = Path(root) / "input" / "sample_plan.json"
    atomic_write_json(path, plan)
    return path


def load_plan(root: str | Path) -> dict[str, Any]:
    return json.loads((Path(root) / "input" / "sample_plan.json").read_text(encoding="utf-8"))


def render_page(page: dict[str, Any], plan_seed: int) -> Image.Image:
    img = Image.new("L", (PAGE_W, PAGE_H), 255)
    for column in page["columns"]:
        for block in column["blocks"]:
            reading_no = block.get("reading_no", 0)
            stable_id_seed = int(hashlib.sha256(block["block_id"].encode("utf-8")).hexdigest()[:8], 16)
            seed = int(page.get("seed", plan_seed)) * 17 + stable_id_seed % 100003
            tile = render_glyph_image(block["char"], jitter_seed=seed)
            cx = block["x_center"]
            y = block["y"]
            img.paste(tile, (cx - tile.width // 2, y + BLOCK // 2 - tile.height // 2))
    return img


def _truth_from_plan(plan: dict[str, Any]) -> dict[str, Any]:
    images = []
    for page in plan["pages"]:
        img_entry = {
            "image_no": page["image_no"],
            "image": page["name"],
            "seed": page.get("seed", plan["seed"]),
            "columns": [],
        }
        for column in sorted(page["columns"], key=lambda c: -c["x_center"]):
            col_entry = {
                "column_id": column["column_id"],
                "reading_direction": "right_to_left",
                "bbox": column["bbox"],
                "blocks": [],
            }
            for block in sorted(column["blocks"], key=lambda b: b["y"]):
                col_entry["blocks"].append({
                    "block_id": block["block_id"],
                    "char": block["char"],
                    "bbox": block["bbox"],
                    "keypoint": block["keypoint"],
                    "reading_no": block["reading_no"],
                })
            img_entry["columns"].append(col_entry)
        images.append(img_entry)
    return {
        "schema": "vertical-layout-truth/v1",
        "width": plan["width"],
        "height": plan["height"],
        "mode": plan["mode"],
        "images": images,
    }


def keypoint_submissions_from_plan(plan: dict[str, Any]) -> dict[str, Any]:
    submissions = []
    base_ms = 1763200000000
    for page in plan["pages"]:
        for column in page["columns"]:
            for block in column["blocks"]:
                x, y = block["keypoint"]
                no = block["reading_no"]
                common = {"image": page["name"], "block_id": block["block_id"]}
                submissions.extend([
                    {**common, "version_no": 1, "version_name": "kpv-v1",
                     "annotated_ms": base_ms + page["image_no"] * 100000 + no * 1000,
                     "annotator": "zz_old", "x": x - 2, "y": y - 1},
                    {**common, "version_no": 2, "version_name": "kpv-v2-b",
                     "annotated_ms": base_ms + page["image_no"] * 100000 + no * 1000 + 200,
                     "annotator": "b_sync", "x": x + 1, "y": y},
                    {**common, "version_no": 2, "version_name": "kpv-v2-a",
                     "annotated_ms": base_ms + page["image_no"] * 100000 + no * 1000 + 500,
                     "annotator": "a_field", "x": x, "y": y + 1},
                    {**common, "version_no": 3, "version_name": "kpv-v3-authority",
                     "annotated_ms": base_ms + page["image_no"] * 100000 + no * 1000 + 900,
                     "annotator": "m_lead", "x": x, "y": y},
                ])
    return {"schema": "keypoint-submissions/v1", "submissions": submissions}


def generate_workspace(root: str | Path, plan: dict[str, Any] | None = None) -> dict[str, Path]:
    root = Path(root)
    input_dir = root / "input"
    config_dir = root / "config"
    ledger_dir = root / "ledger"
    for directory in (input_dir, config_dir, ledger_dir, root / "snapshots", root / "reports", root / "exchange"):
        directory.mkdir(parents=True, exist_ok=True)
    plan = plan or default_plan()
    renumber_plan(plan)
    write_plan(root, plan)
    for page in plan["pages"]:
        image = render_page(page, int(plan.get("seed", 319)))
        image.save(input_dir / page["name"], format="PNG")
    atomic_write_json(input_dir / "layout_truth.json", _truth_from_plan(plan))
    atomic_write_json(input_dir / "keypoint_versions.json", keypoint_submissions_from_plan(plan))
    return {
        "input": input_dir,
        "config": config_dir,
        "ledger": ledger_dir,
        "plan": input_dir / "sample_plan.json",
    }


def apply_plan_commands(plan: dict[str, Any], commands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    effects = []
    for cmd in commands:
        op = cmd["op"]
        if op == "set_keypoint_version" or op == "set_config":
            effects.append({"op": op, "deferred": True})
            continue
        page = next((p for p in plan["pages"] if p["name"] == cmd.get("image")), None)
        if page is None:
            raise KeyError(f"image not found: {cmd.get('image')}")
        if op == "replace_image":
            page["seed"] = int(cmd.get("seed", page["seed"] + 101))
            effects.append({"op": op, "image": page["name"], "seed": page["seed"]})
        elif op == "remove_column":
            before = len(page["columns"])
            page["columns"] = [c for c in page["columns"] if c["column_id"] != cmd["column_id"]]
            if len(page["columns"]) == before:
                raise KeyError(f"column not found: {cmd['column_id']}")
            effects.append({"op": op, "image": page["name"], "column_id": cmd["column_id"]})
        elif op == "add_column":
            x_center = cmd.get("x_center", X_CENTERS[3])
            column = {"column_id": cmd["column_id"], "x_center": x_center, "blocks": []}
            chars = cmd.get("chars", GLYPH_ORDER[:DEFAULT_BLOCKS])
            for i, char in enumerate(chars[:DEFAULT_BLOCKS]):
                y = Y_START + i * Y_PITCH
                column["blocks"].append({
                    "block_id": f"{cmd['column_id']}-b{i}",
                    "char": char,
                    "x_center": x_center,
                    "y": y,
                })
            page["columns"].append(column)
            effects.append({"op": op, "image": page["name"], "column_id": cmd["column_id"]})
        elif op == "add_block":
            column = next((c for c in page["columns"] if c["column_id"] == cmd["column_id"]), None)
            if column is None:
                raise KeyError(f"column not found: {cmd['column_id']}")
            y = cmd.get("y", Y_START + len(column["blocks"]) * Y_PITCH)
            column["blocks"].append({
                "block_id": cmd["block_id"],
                "char": cmd["char"],
                "x_center": column["x_center"],
                "y": y,
            })
            effects.append({"op": op, "block_id": cmd["block_id"]})
        elif op == "remove_block":
            for column in page["columns"]:
                before = len(column["blocks"])
                column["blocks"] = [b for b in column["blocks"] if b["block_id"] != cmd["block_id"]]
                if len(column["blocks"]) != before:
                    effects.append({"op": op, "block_id": cmd["block_id"]})
                    break
            else:
                raise KeyError(f"block not found: {cmd['block_id']}")
        plan["global_version"] = int(plan.get("global_version", 1)) + 1
    renumber_plan(plan)
    return effects


def append_keypoint_command(root: str | Path, cmd: dict[str, Any]) -> dict[str, Any]:
    path = Path(root) / "input" / "keypoint_versions.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    submission = {
        "image": cmd["image"],
        "block_id": cmd["block_id"],
        "version_no": int(cmd["version_no"]),
        "version_name": cmd["version_name"],
        "annotated_ms": int(cmd["annotated_ms"]),
        "annotator": cmd["annotator"],
        "x": int(cmd["x"]),
        "y": int(cmd["y"]),
    }
    data["submissions"].append(submission)
    atomic_write_json(path, data)
    return submission


def clone_plan(plan: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(plan)
