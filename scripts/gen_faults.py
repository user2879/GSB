"""Generate fault-injection sample directories for the validation tests.

Each scenario copies the healthy sample tree and applies one corruption:
  nonfinite_weight      NaN in a weight binary        -> E_BASE_INVALID
  shape_mismatch        truncated weight binary       -> E_BASE_INVALID
  layer_not_whitelisted task targets unlisted layer   -> E_TASK_INVALID
  zero_tolerance        atol = 0 in config            -> E_QUANT_INVALID
  bad_quant_tier        quant_tier outside enum       -> E_QUANT_INVALID
  negative_rank         rank = -1 in a task           -> E_TASK_INVALID
  huge_alpha            alpha so large the merge saturates -> E_IRREVERSIBLE_MERGE
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np
import yaml

SCENARIOS = (
    "nonfinite_weight",
    "shape_mismatch",
    "layer_not_whitelisted",
    "zero_tolerance",
    "bad_quant_tier",
    "negative_rank",
    "huge_alpha",
)


def _load(scn_dir, rel):
    with open(os.path.join(scn_dir, rel), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _dump(scn_dir, rel, payload):
    with open(os.path.join(scn_dir, rel), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def apply_scenario(base_dir: str, scn_dir: str, scenario: str) -> None:
    shutil.copytree(base_dir, scn_dir)
    manifest = _load(scn_dir, "base_pack/layers.json")
    first = manifest["layers"][0]

    if scenario == "nonfinite_weight":
        path = os.path.join(scn_dir, "base_pack", first["weight_file"])
        arr = np.fromfile(path, dtype="<f8").copy()
        arr[0] = np.nan
        arr.tofile(path)
    elif scenario == "shape_mismatch":
        path = os.path.join(scn_dir, "base_pack", first["weight_file"])
        with open(path, "r+b") as fh:
            fh.truncate(os.path.getsize(path) - 8)
    elif scenario == "layer_not_whitelisted":
        tasks = _load(scn_dir, "tasks.json")
        tasks["tasks"][0]["target_layer"] = "not.a.whitelisted.layer"
        _dump(scn_dir, "tasks.json", tasks)
    elif scenario == "zero_tolerance":
        cfg_path = os.path.join(scn_dir, "peft_config.yaml")
        cfg = yaml.safe_load(open(cfg_path))
        cfg["atol"] = 0.0
        yaml.safe_dump(cfg, open(cfg_path, "w"), sort_keys=True)
    elif scenario == "bad_quant_tier":
        cfg_path = os.path.join(scn_dir, "peft_config.yaml")
        cfg = yaml.safe_load(open(cfg_path))
        cfg["quant_tier"] = 9
        yaml.safe_dump(cfg, open(cfg_path, "w"), sort_keys=True)
    elif scenario == "negative_rank":
        tasks = _load(scn_dir, "tasks.json")
        tasks["tasks"][0]["rank"] = -1
        _dump(scn_dir, "tasks.json", tasks)
    elif scenario == "huge_alpha":
        tasks = _load(scn_dir, "tasks.json")
        tasks["tasks"][0]["alpha"] = 1.0e18
        _dump(scn_dir, "tasks.json", tasks)
    else:
        raise ValueError(f"unknown scenario {scenario}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir")
    parser.add_argument("out_dir")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for scenario in SCENARIOS:
        apply_scenario(args.base_dir, os.path.join(args.out_dir, scenario), scenario)
        print(f"fault scenario generated: {scenario}")


if __name__ == "__main__":
    main()
