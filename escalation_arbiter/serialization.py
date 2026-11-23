"""Three-version serialization, conversion and loss auditing."""

import json
import os

from .fingerprint import fingerprint
from .versions import v1, v2, v3
from .versions.common import VERSIONS
from .diagnostics import deviation_histogram, reason_restore_stats

_WRITERS = {"v1": v1.write_bundle, "v2": v2.write_bundle,
            "v3": v3.write_bundle}
_READERS = {"v1": v1.read_bundle, "v2": v2.read_bundle,
            "v3": v3.read_bundle}


def serialize(version, decisions, rules, config, work_dir):
    if version not in VERSIONS:
        raise ValueError(f"unknown exchange version {version}")
    data_path = os.path.join(work_dir, f"bundle_{version}")
    side_path = os.path.join(work_dir, f"sidecar_{version}.json")
    suffix = ".txt" if version == "v1" else ".json" if version == "v2" \
        else ".bin"
    data_path += suffix
    sidecar = _WRITERS[version](data_path, decisions, rules, config)
    sidecar["data_path"] = os.path.basename(data_path)
    sidecar["fingerprint"] = fingerprint(
        [d.to_dict() for d in sorted(decisions, key=lambda d: d.tid)])
    with open(side_path, "w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, ensure_ascii=False, indent=2, sort_keys=True)
    return data_path, side_path


def deserialize(version, work_dir):
    suffix = ".txt" if version == "v1" else ".json" if version == "v2" \
        else ".bin"
    data_path = os.path.join(work_dir, f"bundle_{version}{suffix}")
    side_path = os.path.join(work_dir, f"sidecar_{version}.json")
    with open(side_path, "r", encoding="utf-8") as fh:
        sidecar = json.load(fh)
    return _READERS[version](data_path, sidecar)


def convert(source_version, target_version, work_dir):
    decisions = deserialize(source_version, work_dir)
    rules = _load_rules_hint(work_dir)
    config = _load_config_hint(work_dir)
    return serialize(target_version, decisions, rules, config, work_dir)


def attach_hints(rules, config, work_dir):
    with open(os.path.join(work_dir, "_rules_hint.json"), "w",
              encoding="utf-8") as fh:
        json.dump({
            "type_weights": rules.type_weights,
            "role_credibility": rules.role_cred,
            "reason_codes": [{"code": k, "name": v} for k, v in
                             sorted(rules.reason_dict.items())],
        }, fh, ensure_ascii=False, sort_keys=True)
    with open(os.path.join(work_dir, "_config_hint.json"), "w",
              encoding="utf-8") as fh:
        json.dump(config.to_dict(), fh, sort_keys=True)


def _load_rules_hint(work_dir):
    from .loader import load_rule_package
    return load_rule_package(os.path.join(work_dir, "_rules_hint.json"))[0]


def _load_config_hint(work_dir):
    from .loader import load_config
    return load_config(os.path.join(work_dir, "_config_hint.json"))[0]


def round_trip_stats(decisions, rules, config, work_dir):
    """Serialize+deserialize every version; score deviation histograms."""
    baseline = {d.tid: d for d in decisions}
    report = {}
    for version in VERSIONS:
        serialize(version, decisions, rules, config, work_dir)
        restored = deserialize(version, work_dir)
        deviations = []
        reason_ok = 0
        reason_total = 0
        for decision in restored:
            original = baseline[decision.tid]
            deviations.append(decision.decision_score
                              - original.decision_score)
            reason_total += 1
            if decision.reason_codes == original.reason_codes:
                reason_ok += 1
        report[version] = {
            "score_deviation_histogram": deviation_histogram(deviations),
            "reason_restore": reason_restore_stats(
                reason_ok, reason_total),
        }
    return report
