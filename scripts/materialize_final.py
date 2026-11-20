"""把交互命令流物化到一份最终输入目录，用于独立全量重建比对。"""

from __future__ import annotations

import argparse
import json
import shutil
from fractions import Fraction
from pathlib import Path

from rerank_core.commands import parse_commands


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--commands", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    src = Path(args.input)
    out = Path(args.out)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    config = json.loads((src / "rerank_config.json").read_text(encoding="utf-8"))
    records: dict[str, dict[str, dict]] = {}
    order: list[str] = []
    with (src / "candidates.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            rid = record["request_id"]
            records.setdefault(rid, {})
            if rid not in order:
                order.append(rid)
            records[rid][record["item_id"]] = record

    commands = parse_commands(json.loads(
        Path(args.commands).read_text(encoding="utf-8"))["commands"])
    for command in commands:
        payload = command.payload
        if command.type == "set_weight":
            config["weights"][payload["feature_index"]] = payload["weight"]
        elif command.type == "set_normalization":
            config["normalization"] = payload["normalization"]
        elif command.type == "set_quant":
            config["frac_bits"] = payload.get("frac_bits", config["frac_bits"])
            config["quant_levels"] = payload.get(
                "quant_levels", config["quant_levels"])
        elif command.type == "set_rerank_window":
            config["rerank_window"] = payload["window"]
        elif command.type == "set_penalty":
            config["diversity_penalty"] = payload["penalty"]
        elif command.type == "add_candidate":
            rid = payload["request_id"]
            record = payload["record"]
            records.setdefault(rid, {})
            records[rid][record["item_id"]] = record
        elif command.type == "delete_candidate":
            rid, item_id = payload["request_id"], payload["item_id"]
            records.get(rid, {}).pop(item_id, None)

    (out / "rerank_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy(src / "feature_groups.json", out / "feature_groups.json")
    with (out / "candidates.jsonl").open("w", encoding="utf-8") as handle:
        for rid in sorted(records):
            for item_id in sorted(records[rid]):
                handle.write(json.dumps(records[rid][item_id],
                                        ensure_ascii=False) + "\n")
    print(f"最终输入已物化到 {out}")


if __name__ == "__main__":
    main()
