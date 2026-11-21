"""Deterministic synthetic corpus generator (no randomness, no network).

Generates:
  data/vocab/vocabulary.json
  data/config/hierarchy_config.json
  data/samples/batch_0.jsonl (+ invalid rows in batch_3)
  data/plans/edit_plan.json
  data/injections/injection_manifest.json

All ordering is explicit; repeated runs produce byte-identical files.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from workbench.io_utils import atomic_write, dumps  # noqa: E402

VOCAB = {
    "presale": [
        "consult", "price", "discount", "promotion", "stock", "recommend",
        "feature", "compare", "new_arrival", "warranty",
    ],
    "aftersale": [
        "repair", "broken", "fault", "refund", "return", "exchange",
        "quality", "complain", "service", "maintain",
    ],
    "billing": [
        "invoice", "receipt", "amount", "tax", "duplicate", "reissue",
        "company", "personal", "settle", "deduct",
    ],
    "logistics": [
        "ship", "track", "delivery", "delay", "address", "courier",
        "dispatch", "warehouse", "sign", "parcel",
    ],
    "shared": [
        "please", "want", "need", "hello", "thanks", "today", "tomorrow",
        "urgent", "my", "the", "a", "for", "to", "of", "and", "with",
        "change",
    ],
    "products": ["phone", "laptop", "headphone", "tablet", "keyboard"],
    "cities": ["beijing", "shanghai", "guangzhou", "shenzhen", "hangzhou"],
    "issues": ["screen", "battery", "keyboard_fault", "camera", "speaker"],
    "invoice_types": ["special_invoice", "ordinary_invoice", "electronic_invoice"],
    "amounts": ["100", "299", "1299", "4999", "8999"],
    "time_slots": ["morning", "afternoon", "evening", "weekend"],
}

DOMAIN_WORD_BAGS = {
    "presale": {
        "consult": 3, "price": 3, "discount": 3, "promotion": 2, "stock": 2,
        "recommend": 2, "feature": 1, "compare": 1, "new_arrival": 1,
        "warranty": 1, "phone": 1, "laptop": 1, "headphone": 1, "tablet": 1,
        "keyboard": 1,
    },
    "aftersale": {
        "repair": 3, "broken": 3, "fault": 2, "refund": 3, "return": 2,
        "exchange": 2, "quality": 1, "complain": 1, "service": 1,
        "maintain": 1, "screen": 1, "battery": 1, "keyboard_fault": 1,
        "camera": 1, "speaker": 1,
    },
    "billing": {
        "invoice": 3, "receipt": 2, "amount": 2, "tax": 2, "duplicate": 2,
        "reissue": 2, "company": 1, "personal": 1, "settle": 1, "deduct": 1,
        "special_invoice": 1, "ordinary_invoice": 1, "electronic_invoice": 1,
    },
    "logistics": {
        "ship": 3, "track": 3, "delivery": 2, "delay": 2, "address": 2,
        "courier": 2, "dispatch": 1, "warehouse": 1, "sign": 1, "parcel": 1,
        "beijing": 1, "shanghai": 1, "guangzhou": 1, "shenzhen": 1, "hangzhou": 1,
    },
}

INTENT_TEMPLATES = {
    "price_inquiry": {
        "triggers": ["price", "discount", "promotion"],
        "negations": [],
        "negation_window": 1,
    },
    "product_consult": {
        "triggers": ["consult", "recommend", "feature", "compare"],
        "negations": [],
        "negation_window": 1,
    },
    "repair_request": {
        "triggers": ["repair", "broken", "fault", "maintain"],
        "negations": [],
        "negation_window": 1,
    },
    "refund_request": {
        "triggers": ["refund", "return", "exchange", "complain"],
        "negations": ["not"],
        "negation_window": 1,
    },
    "invoice_issue": {
        "triggers": ["invoice", "reissue", "duplicate", "tax"],
        "negations": [],
        "negation_window": 1,
    },
    "receipt_settlement": {
        "triggers": ["receipt", "settle", "deduct", "amount"],
        "negations": [],
        "negation_window": 1,
    },
    "track_shipment": {
        "triggers": ["track", "ship", "parcel", "courier"],
        "negations": [],
        "negation_window": 1,
    },
    "delivery_change": {
        "triggers": ["delivery", "delay", "address", "dispatch"],
        "negations": [],
        "negation_window": 1,
    },
}

# "not" must live in the controlled vocabulary to act as a negator.
NEGATOR = "not"

SLOT_DEFINITIONS = {
    "product": {"values": [[p] for p in VOCAB["products"]]},
    "city": {"values": [[c] for c in VOCAB["cities"]]},
    "issue_part": {"values": [[i] for i in VOCAB["issues"]]},
    "invoice_type": {"values": [[t] for t in VOCAB["invoice_types"]]},
    "invoice_amount": {"values": [[a] for a in VOCAB["amounts"]]},
    "time_window": {"values": [[t] for t in VOCAB["time_slots"]]},
}

INTENT_REQUIRED_SLOTS = {
    "price_inquiry": ["product"],
    "product_consult": ["product"],
    "repair_request": ["product", "issue_part"],
    "refund_request": ["product", "issue_part"],
    "invoice_issue": ["invoice_type", "invoice_amount"],
    "receipt_settlement": ["invoice_amount"],
    "track_shipment": ["city"],
    "delivery_change": ["city", "time_window"],
}

DOMAIN_INTENTS = {
    "presale": ["price_inquiry", "product_consult"],
    "aftersale": ["repair_request", "refund_request"],
    "billing": ["invoice_issue", "receipt_settlement"],
    "logistics": ["track_shipment", "delivery_change"],
}


def build_vocab():
    tokens = []
    for group in VOCAB.values():
        tokens.extend(group)
    tokens.append(NEGATOR)
    # deterministic de-dup preserving first-seen order
    seen = set()
    ordered = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            ordered.append(token)
    return {"version": 1, "tokens": ordered, "domain_word_bags": DOMAIN_WORD_BAGS}


def build_config():
    return {
        "version": 1,
        "domains": ["aftersale", "billing", "logistics", "presale"],
        "domain_word_bags": DOMAIN_WORD_BAGS,
        "domain_intents": DOMAIN_INTENTS,
        "intent_templates": INTENT_TEMPLATES,
        "slot_definitions": SLOT_DEFINITIONS,
        "intent_required_slots": INTENT_REQUIRED_SLOTS,
        "thresholds": {"domain": "0.15", "intent": "0.25"},
        "slot_match_window": 2,
        "fixed_point_bits": 16,
        "quantization_levels": {
            "coarse": {"grid": "0.1", "bits": 1},
            "medium": {"grid": "0.01", "bits": 2},
            "fine": {"grid": "0.0001", "bits": 4},
        },
        "active_quantization_level": "medium",
        "recalibration": {
            "completeness_weight": "0.2",
            "suppression_weight": "0.05",
            "floor": "0.02",
        },
        "max_workers": 4,
        "slot_budget_per_batch": 16,
        "snapshot_commit_interval": 1,
    }

# Explicit deterministic sample rows. Each row: id, tokens, domain, intent,
# gold triples [slot, value tokens, start index].
SAMPLE_ROWS = [
    ("U001", ["hello", "want", "price", "phone", "today"], "presale",
     "price_inquiry", [["product", ["phone"], 3]]),
    ("U002", ["please", "consult", "feature", "laptop", "and", "tablet"],
     "presale", "product_consult", [["product", ["laptop"], 3]]),
    ("U003", ["my", "phone", "screen", "broken", "need", "repair"],
     "aftersale", "repair_request",
     [["product", ["phone"], 1], ["issue_part", ["screen"], 2]]),
    ("U004", ["battery", "fault", "want", "refund", "laptop"],
     "aftersale", "refund_request",
     [["product", ["laptop"], 4], ["issue_part", ["battery"], 0]]),
    ("U005", ["need", "invoice", "reissue", "special_invoice", "amount", "1299"],
     "billing", "invoice_issue",
     [["invoice_type", ["special_invoice"], 3],
      ["invoice_amount", ["1299"], 5]]),
    ("U006", ["please", "settle", "amount", "4999", "with", "receipt"],
     "billing", "receipt_settlement", [["invoice_amount", ["4999"], 3]]),
    ("U007", ["track", "parcel", "to", "shanghai", "courier"],
     "logistics", "track_shipment", [["city", ["shanghai"], 3]]),
    ("U008", ["delivery", "delay", "please", "change", "address", "beijing",
              "morning"],
     "logistics", "delivery_change",
     [["city", ["beijing"], 5], ["time_window", ["morning"], 6]]),
]

BATCH1_ROWS = [
    ("U101", ["discount", "promotion", "headphone", "please"],
     "presale", "price_inquiry", [["product", ["headphone"], 2]]),
    ("U102", ["camera", "broken", "repair", "tablet", "urgent"],
     "aftersale", "repair_request",
     [["product", ["tablet"], 3], ["issue_part", ["camera"], 0]]),
]

BATCH2_ROWS = [
    ("U201", ["duplicate", "invoice", "electronic_invoice", "amount", "299"],
     "billing", "invoice_issue",
     [["invoice_type", ["electronic_invoice"], 2],
      ["invoice_amount", ["299"], 4]]),
    ("U202", ["ship", "parcel", "dispatch", "guangzhou", "afternoon"],
     "logistics", "delivery_change",
     [["city", ["guangzhou"], 3], ["time_window", ["afternoon"], 4]]),
]

# Multi-candidate suppression: two products mentioned -> two slot candidates.
BATCH3_ROWS = [
    ("U301", ["broken", "phone", "and", "laptop", "fault", "repair"],
     "aftersale", "repair_request",
     [["product", ["phone"], 1], ["issue_part", ["battery"], 99]]),
    ("U302", ["track", "ship", "shenzhen", "courier", "parcel"],
     "logistics", "track_shipment", [["city", ["shenzhen"], 2]]),
]

# Deliberately invalid rows for the failure list.
INVALID_ROWS = [
    {"utterance_id": "U901",
     "tokens": ["price", "phone", "ZZZ_NOT_A_TOKEN"],
     "domain": "presale", "intent": "price_inquiry", "gold_slots": []},
    {"utterance_id": "U902", "tokens": ["repair"], "domain": "mars",
     "intent": "repair_request", "gold_slots": []},
    {"utterance_id": "U903", "tokens": ["repair"], "domain": "aftersale",
     "intent": "price_inquiry", "gold_slots": []},
    {"utterance_id": "U001", "tokens": ["price", "phone"], "domain": "presale",
     "intent": "price_inquiry", "gold_slots": []},
    {"utterance_id": "U904", "tokens": ["price", "phone"], "domain": "presale",
     "intent": "price_inquiry",
     "gold_slots": [["ghost_slot", ["phone"], 1]]},
]


def row_to_json(row):
    uid, tokens, domain, intent, gold = row
    return {"utterance_id": uid, "tokens": tokens, "domain": domain,
            "intent": intent, "gold_slots": gold}


def build_edit_plan():
    return {
        "plan_id": "plan-001",
        "batches": [
            {
                "batch_id": "B1",
                "add_samples": [row_to_json(r) for r in BATCH1_ROWS],
                "intent_migrations": [],
                "slot_pattern_edits": [],
                "threshold_edits": {},
            },
            {
                "batch_id": "B2",
                "add_samples": [row_to_json(r) for r in BATCH2_ROWS],
                # product_consult moves presale -> aftersale; only utterances
                # referencing that intent template may invalidate.
                "intent_migrations": [
                    {"intent": "product_consult", "source_domain": "presale",
                     "target_domain": "aftersale"}],
                "slot_pattern_edits": [],
                "threshold_edits": {},
            },
            {
                "batch_id": "B3",
                "add_samples": [row_to_json(r) for r in BATCH3_ROWS] + INVALID_ROWS,
                "intent_migrations": [],
                "slot_pattern_edits": [
                    {"intent": "track_shipment",
                     "required_slots": ["city", "time_window"],
                     "add_slot_values": []}],
                "threshold_edits": {"intent": "0.3"},
            },
            {
                "batch_id": "B4",
                "add_samples": [],
                "intent_migrations": [
                    {"intent": "product_consult", "source_domain": "aftersale",
                     "target_domain": "presale"}],
                "slot_pattern_edits": [
                    {"intent": "track_shipment",
                     "required_slots": ["city"],
                     "add_slot_values": [
                         {"slot": "city",
                          "values": [["hangzhou"], ["shenzhen"]]}]}],
                "threshold_edits": {"active_quantization_level": "fine"},
            },
            {
                "batch_id": "B5",
                "add_samples": [
                    {"utterance_id": "U401",
                     "tokens": ["track", "parcel", "hangzhou", "courier"],
                     "domain": "logistics", "intent": "track_shipment",
                     "gold_slots": [["city", ["hangzhou"], 2]]}],
                "intent_migrations": [],
                "slot_pattern_edits": [],
                "threshold_edits": {},
            },
        ],
    }


def build_injection_manifest():
    return {
        "manifest_id": "injection-001",
        "injections": [
            {"id": "INJ-0001", "kind": "illegal_token",
             "tokens": ["price", "phone", "HACKED_TOKEN"]},
            {"id": "INJ-0002", "kind": "migration_conflict",
             "intent": "repair_request", "source_domain": "aftersale",
             "target_domain": "unknown_realm"},
            {"id": "INJ-0003", "kind": "migration_conflict",
             "intent": "ghost_intent", "target_domain": "presale"},
            {"id": "INJ-0004", "kind": "non_finite_threshold",
             "level": "intent", "value": "NaN"},
            {"id": "INJ-0005", "kind": "non_finite_threshold",
             "level": "domain", "value": "Infinity"},
            {"id": "INJ-0006", "kind": "non_finite_threshold",
             "level": "intent", "value": "1.7"},
        ],
    }

def write_json(path, obj):
    atomic_write(path, dumps(obj) + "\n")


def write_jsonl(path, rows):
    text = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False,
                              separators=(",", ":")) + "\n" for row in rows)
    atomic_write(path, text)


def main():
    vocab = build_vocab()
    config = build_config()
    write_json(os.path.join(ROOT, "data/vocab/vocabulary.json"), vocab)
    write_json(os.path.join(ROOT, "data/config/hierarchy_config.json"), config)
    write_jsonl(os.path.join(ROOT, "data/samples/batch_0.jsonl"),
                [row_to_json(r) for r in SAMPLE_ROWS])
    write_json(os.path.join(ROOT, "data/plans/edit_plan.json"), build_edit_plan())
    write_json(os.path.join(ROOT, "data/injections/injection_manifest.json"),
               build_injection_manifest())
    print("corpus generated under data/")


if __name__ == "__main__":
    main()
