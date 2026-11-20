"""Scale-up helper.

Generates every preset tier (or a custom tier) from the fixed seed and prints
the resulting entry counts. Custom tiers let reviewers sweep data size
continuously; the generator is deterministic and network-free.

Examples
--------
Regenerate all preset tiers::

    python3 scripts/scale_up.py --data-dir data/generated

Add a custom tier of 12k queries / 12k vocab / 18k entities / 220 rules::

    python3 scripts/scale_up.py --data-dir data/custom \\
        --custom custom:12000:12000:18000:220
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qu.config import DEFAULT_CONFIG
from qu.synthetic import DEFAULT_TIERS, SyntheticGenerator, TierSpec


def parse_custom(value: str) -> TierSpec:
    parts = value.split(":")
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            "expected tier:queries:vocab:entities:rules")
    tier_id, queries, vocab, entities, rules = parts
    return TierSpec(tier_id, int(queries), int(vocab), int(entities),
                    int(rules))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/generated")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
    parser.add_argument("--custom", type=parse_custom, default=None,
                        help="tier:queries:vocab:entities:rules")
    args = parser.parse_args()

    generator = SyntheticGenerator(args.seed)
    specs = (args.custom,) if args.custom else DEFAULT_TIERS
    manifest = {}
    if os.path.exists(os.path.join(args.data_dir, "manifest.json")):
        with open(os.path.join(args.data_dir, "manifest.json"),
                  encoding="utf-8") as handle:
            manifest = json.load(handle)
    for spec in specs:
        manifest[spec.tier_id] = generator.generate_tier(args.data_dir, spec)
        print(json.dumps({
            "tier": spec.tier_id,
            "queries": spec.queries,
            "vocab": spec.vocab,
            "entities": spec.entities,
            "rules": spec.rules,
        }, ensure_ascii=False))
    with open(os.path.join(args.data_dir, "manifest.json"),
              "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2,
                  sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
