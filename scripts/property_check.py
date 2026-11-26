from __future__ import annotations

import copy
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rope_lab.commands import apply_command
from rope_lab.engine import LabEngine
from rope_lab.generator import default_config, generate_samples, make_sample
from rope_lab.hashing import canonical_json


def _fixed_point(rng: random.Random) -> dict[str, object]:
    signed_bits = rng.choice([16, 32, 64])
    fraction_bits = rng.choice([4, 8, 12])
    if fraction_bits >= signed_bits:
        fraction_bits = 8
    return {
        "signed_bits": signed_bits,
        "fraction_bits": fraction_bits,
        "accumulator_bits": 64,
        "quant_level": rng.choice(["truncate", "nearest", "ties_even"]),
    }


def main(sequence_count: int = 12) -> int:
    for sequence in range(sequence_count):
        rng = random.Random(9000 + sequence)
        config = default_config()
        samples = generate_samples(config)
        incremental = LabEngine(copy.deepcopy(config), copy.deepcopy(samples))
        commands = []

        for step in range(12):
            names = [scheme["name"] for scheme in incremental.config["schemes"]]
            op = rng.choice(["add", "remove", "base", "scale", "train", "target", "replace", "fixed", "schedule"])
            command_id = f"R{sequence}-{step}"
            if op == "add":
                command = {
                    "command_id": command_id,
                    "op": "add_scheme",
                    "scheme": {
                        "name": f"rand_{sequence}_{step}",
                        "base_frequency": float(rng.choice([1000, 10000, 50000])),
                        "angle_scale": rng.choice([0.5, 1.0, 1.5]),
                        "block_size": rng.choice([2, 4, 8]),
                        "frequency_correction": rng.choice([0, 1, 2, 3]),
                    },
                }
            elif op == "remove" and len(names) > 1:
                removable = names[1:] if len(names) > 1 else names
                command = {"command_id": command_id, "op": "remove_scheme", "name": rng.choice(removable)}
            elif op == "base":
                command = {"command_id": command_id, "op": "update_scheme", "name": rng.choice(names), "params": {"base_frequency": float(rng.choice([5000, 10000, 30000]))}}
            elif op == "scale":
                command = {"command_id": command_id, "op": "update_scheme", "name": rng.choice(names), "params": {"angle_scale": rng.choice([0.5, 0.75, 1.0, 1.25])}}
            elif op == "train":
                command = {"command_id": command_id, "op": "set_train_length", "train_length": rng.choice([4, 8, 12, 16])}
            elif op == "target":
                command = {"command_id": command_id, "op": "set_target_length", "target_length": rng.choice([18, 24, 30])}
            elif op == "replace":
                sample_id = rng.choice([sample["sample_id"] for sample in incremental.samples])
                index = int(sample_id[1:])
                replacement = make_sample(incremental.config, sample_id, index, replacement=True)
                command = {"command_id": command_id, "op": "replace_sample", "sample": replacement}
            elif op == "fixed":
                command = {"command_id": command_id, "op": "set_fixed_point", "fixed_point": _fixed_point(rng)}
            else:
                command = {
                    "command_id": command_id,
                    "op": "set_scheduling",
                    "point_budget": rng.choice([100, 240, 1000]),
                    "sample_quota": rng.choice([80, 160]),
                    "concurrency": rng.choice([1, 2, 4]),
                }
            commands.append(command)
            manifest = incremental.apply_command(command)
            incremental.run(manifest, parallel=rng.choice([True, False]))

        incremental.config["point_budget"] = 10000
        incremental.config["sample_quota"] = 10000
        incremental.run({"op": "final"}, parallel=True)
        incremental_semantic = incremental.semantic_report()

        final_config = copy.deepcopy(config)
        final_samples = copy.deepcopy(samples)
        for command in commands:
            final_config, final_samples, _ = apply_command(final_config, final_samples, command)
        final_config["point_budget"] = 10000
        final_config["sample_quota"] = 10000
        blank = LabEngine(final_config, final_samples, global_version=incremental.global_version)
        blank_report = blank.run(parallel=False)

        field_equal = canonical_json(incremental_semantic) == canonical_json(blank.semantic_report())
        order_equal = [entry["stable_key"] for entry in incremental.last_report["ledger"]] == [
            entry["stable_key"] for entry in blank_report["ledger"]
        ]
        print(f"sequence={sequence:02d} field_equal={field_equal} ledger_order_equal={order_equal}")
        if not field_equal or not order_equal:
            return 1
    print("ALL RANDOM COMMAND SEQUENCES MATCH FULL REBUILD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
