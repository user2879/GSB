"""Edit-sequence runner: applies JSON edit ops to a working copy of inputs."""

import json
from pathlib import Path

import yaml

from .config import Config
from .samplegen import generate_image

LABEL_FILES = {"label_a": "labels_a.jsonl", "label_b": "labels_b.jsonl"}


def _read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _write_jsonl(path, entries):
    path.write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries),
        encoding="utf-8",
    )


def apply_edit(edit, data_dir, config_path):
    data_dir = Path(data_dir)
    config_path = Path(config_path)
    op = edit["op"]

    if op in ("desc_modify", "desc_add", "desc_remove"):
        path = data_dir / "descriptions" / f"{edit['sample_id']}.json"
        desc = json.loads(path.read_text(encoding="utf-8"))
        if op == "desc_modify":
            desc["assertions"][edit["index"]].update(edit["patch"])
        elif op == "desc_add":
            desc["assertions"].append(edit["assertion"])
        else:
            desc["assertions"].pop(edit["index"])
        path.write_text(json.dumps(desc, ensure_ascii=False, indent=1), encoding="utf-8")
        return

    if op == "label_modify":
        path = data_dir / LABEL_FILES[edit["source"]]
        entries = _read_jsonl(path)
        for entry in entries:
            if entry.get("sample_id") == edit["sample_id"] and entry.get("order") == edit["order"]:
                entry.update(edit["patch"])
        _write_jsonl(path, entries)
        return

    if op == "replace_labels":
        _write_jsonl(data_dir / LABEL_FILES[edit["source"]], edit["entries"])
        return

    if op == "replace_image":
        config = Config.load(config_path)
        img, _ = generate_image(edit["seed"], config)
        img.save(data_dir / "images" / f"{edit['sample_id']}.png")
        return

    if op in ("set_tolerance", "set_priority"):
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if op == "set_tolerance":
            raw["count_tolerance"] = edit["value"]
        else:
            raw["arbitration_priority"] = edit["value"]
        config_path.write_text(
            yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        return

    raise ValueError(f"unknown edit op: {op}")


def apply_edits(edits_path, data_dir, config_path):
    edits = json.loads(Path(edits_path).read_text(encoding="utf-8"))
    for edit in edits:
        apply_edit(edit, data_dir, config_path)
    return len(edits)
