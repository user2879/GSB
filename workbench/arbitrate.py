"""Consistency judgement and conflict arbitration per (sample, category, color)."""

from .errors import E_ARBITRATION_DEADLOCK
from .ledger import SOURCES


def _pick_by_priority(counts, priority):
    for rule in priority:
        if rule == "detection":
            return counts["detection"], "detection"
        if rule == "description":
            return counts["description"], "description"
        if rule == "label_a":
            return counts["label_a"], "label_a"
        if rule == "label_majority" and counts["label_a"] == counts["label_b"]:
            return counts["label_a"], "label_majority"
    return counts["detection"], "detection_fallback"


def arbitrate(sample_id, category, color, counts, config):
    """counts: {source: int} for the four sources. Returns an arbitration record."""
    values = list(counts.values())
    spread = max(values) - min(values)
    tolerance = config.count_tolerance
    labels_agree = counts["label_a"] == counts["label_b"]
    distinct = len({counts["detection"], counts["description"], counts["label_a"]})

    if spread <= tolerance:
        status = "consistent"
    elif labels_agree and distinct == 3:
        status = "deadlock"
    else:
        status = "conflict"

    record = {
        "sample_id": sample_id,
        "category": category,
        "color": color,
        "status": status,
        "counts": {source: counts[source] for source in SOURCES},
    }
    if status == "deadlock":
        record.update(
            {
                "arbitrated_count": None,
                "rule": "deadlock",
                "code": E_ARBITRATION_DEADLOCK,
                "deviations": {source: None for source in SOURCES},
                "rejected": [],
            }
        )
        return record

    arbitrated, rule = _pick_by_priority(counts, config.arbitration_priority)
    deviations = {source: counts[source] - arbitrated for source in SOURCES}
    rejected = [
        {"source": source, "count": counts[source], "deviation": deviations[source]}
        for source in SOURCES
        if counts[source] != arbitrated
    ]
    record.update(
        {
            "arbitrated_count": arbitrated,
            "rule": rule,
            "code": None,
            "deviations": deviations,
            "rejected": rejected,
        }
    )
    return record
