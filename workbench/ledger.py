"""Unified reconciliation ledger across the four count sources."""

from .fingerprint import sha256_json

SOURCES = ("detection", "description", "label_a", "label_b")


def _entry(sample_id, source, category, color, items):
    if source == "detection":
        count = len(items)
        details = [
            {"instance_id": it["instance_id"], "bbox": it["bbox"]} for it in items
        ]
    elif source == "description":
        count = sum(it["count"] for it in items)
        details = [
            {"index": it["index"], "count": it["count"], "position": it["position"]}
            for it in items
        ]
    else:
        count = len(items)
        details = [{"order": it["order"], "bbox": it["bbox"]} for it in items]
    return {
        "sample_id": sample_id,
        "source": source,
        "category": category,
        "color": color,
        "count": count,
        "zero": count == 0,
        "merged": len(items) > 1,
        "merge_details": details,
        "source_fingerprint": sha256_json(items),
    }


def build_category_payload(sample_id, category, det_instances, description,
                           label_a, label_b, config):
    """Aggregate counts per (category, color, source) for one sample.

    Arbitration records are produced by the caller.
    """
    pools = {
        "detection": [i for i in det_instances if i["category"] == category],
        "description": [a for a in description if a["category"] == category],
        "label_a": [e for e in label_a if e["category"] == category],
        "label_b": [e for e in label_b if e["category"] == category],
    }
    colors = sorted({it["color"] for items in pools.values() for it in items})
    entries = []
    for color in colors:
        for source in SOURCES:
            items = [it for it in pools[source] if it["color"] == color]
            entries.append(_entry(sample_id, source, category, color, items))
    refs = {
        "detection_instances": [i["instance_id"] for i in pools["detection"]],
        "description_assertions": [a["index"] for a in pools["description"]],
        "label_a_orders": [e["order"] for e in pools["label_a"]],
        "label_b_orders": [e["order"] for e in pools["label_b"]],
    }
    return {"ledger": entries, "refs": refs}
