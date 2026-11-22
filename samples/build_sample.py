"""Generate the deterministic snapshot-chain sample used by every demo."""
import json
import os


def keys(prefix, start, count):
    return [f"{prefix}{index:03d}" for index in range(start, start + count)]


RANGES = [
    ("R1", "k0000", "k0100"),
    ("R2", "k0100", "k0200"),
    ("R3", "k0200", "k0300"),
    ("R4", "k0300", "k0400"),
]


def range_decl(generation, range_id, lower, upper, key_list, action,
               tombstone=False):
    entry = {"range_id": range_id, "lower": lower, "upper": upper,
             "keys": key_list, "tombstone": tombstone}
    return entry, {"range_id": range_id, "action": action}


def main():
    generations = []

    # G0: R1 6 keys (split+merge), R2 5, R3 4, R4 3 -> demonstrates split & merge
    g0_ranges, g0_rules = [], []
    for rid, lo, hi in RANGES:
        n = {"R1": 6, "R2": 5, "R3": 4, "R4": 3}[rid]
        entry, rule = range_decl(
            "G0", rid, lo, hi, keys(lo, 0, n), "new")
        g0_ranges.append(entry)
        g0_rules.append(rule)
    generations.append({
        "generation_id": "G0", "seq": 0, "parent_id": None,
        "ranges": g0_ranges, "rules": g0_rules})

    # G1: R1 inherited (byte-identical), R2 discarded, R3 rewritten (5),
    # R4 new with 8 keys (8 -> 4+4 balanced split)
    g1_ranges, g1_rules = [], []
    for rid, lo, hi in RANGES:
        if rid == "R1":
            entry, rule = range_decl("G1", rid, lo, hi, keys(lo, 0, 6),
                                     "inherit")
        elif rid == "R2":
            entry, rule = range_decl("G1", rid, lo, hi, [], "discard",
                                     tombstone=True)
        elif rid == "R3":
            entry, rule = range_decl("G1", rid, lo, hi, keys(lo, 10, 5),
                                     "rewrite")
        else:
            entry, rule = range_decl("G1", rid, lo, hi, keys(lo, 20, 8),
                                     "new")
        g1_ranges.append(entry)
        g1_rules.append(rule)
    generations.append({
        "generation_id": "G1", "seq": 1, "parent_id": "G0",
        "ranges": g1_ranges, "rules": g1_rules})

    # G2: inherits G1 active ranges verbatim; tombstones retained
    g2_ranges, g2_rules = [], []
    for rid, lo, hi in RANGES:
        if rid == "R2":
            entry, rule = range_decl("G2", rid, lo, hi, [], "discard",
                                     tombstone=True)
        else:
            n = {"R1": 6, "R3": 5, "R4": 8}[rid]
            start = {"R1": 0, "R3": 10, "R4": 20}[rid]
            entry, rule = range_decl("G2", rid, lo, hi,
                                     keys(lo, start, n), "inherit")
        g2_ranges.append(entry)
        g2_rules.append(rule)
    generations.append({
        "generation_id": "G2", "seq": 2, "parent_id": "G1",
        "ranges": g2_ranges, "rules": g2_rules})

    # G3: R4 rewritten to 11 keys (11 -> 4+4+3), others inherit
    g3_ranges, g3_rules = [], []
    for rid, lo, hi in RANGES:
        if rid == "R2":
            entry, rule = range_decl("G3", rid, lo, hi, [], "discard",
                                     tombstone=True)
        elif rid == "R4":
            entry, rule = range_decl("G3", rid, lo, hi, keys(lo, 40, 11),
                                     "rewrite")
        else:
            n = {"R1": 6, "R3": 5}[rid]
            start = {"R1": 0, "R3": 10}[rid]
            entry, rule = range_decl("G3", rid, lo, hi,
                                     keys(lo, start, n), "inherit")
        g3_ranges.append(entry)
        g3_rules.append(rule)
    generations.append({
        "generation_id": "G3", "seq": 3, "parent_id": "G2",
        "ranges": g3_ranges, "rules": g3_rules})

    chain = {"title": "deterministic-snapshot-chain",
             "limits": {"lower": 3, "upper": 5},
             "generations": generations}
    out = os.path.join(os.path.dirname(__file__), "snapshot_chain.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(chain, handle, ensure_ascii=False, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
