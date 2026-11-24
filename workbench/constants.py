from __future__ import annotations

SCHEMA_VERSION = "workbench.v1"
CONFIG_SCHEMA_VERSION = "workbench.config.v1"

EVENT_APPLY = "APPLY"
EVENT_WRITE = "WRITE"
EVENT_READ = "READ"
EVENT_RELEASE = "RELEASE"
EVENT_LEASE_RENEW = "LEASE_RENEW"
EVENT_TICK = "TICK"
EVENT_QUOTA_CHANGED = "QUOTA_CHANGED"
EVENT_POLICY_CHANGED = "POLICY_CHANGED"
EVENT_RUN_EVICTION = "RUN_EVICTION"

EVENT_TYPES = {
    EVENT_APPLY,
    EVENT_WRITE,
    EVENT_READ,
    EVENT_RELEASE,
    EVENT_LEASE_RENEW,
    EVENT_TICK,
    EVENT_QUOTA_CHANGED,
    EVENT_POLICY_CHANGED,
    EVENT_RUN_EVICTION,
}

E_NODE_INVALID = "E_NODE_INVALID"
E_MANIFEST_INVALID = "E_MANIFEST_INVALID"
E_EVENT_INVALID = "E_EVENT_INVALID"
E_PROTOCOL_VIOLATION = "E_PROTOCOL_VIOLATION"
E_CAPACITY_EXCEEDED = "E_CAPACITY_EXCEEDED"
E_LEASE_LIMIT_EXCEEDED = "E_LEASE_LIMIT_EXCEEDED"

ERROR_CODE_ORDER = [
    E_NODE_INVALID,
    E_MANIFEST_INVALID,
    E_EVENT_INVALID,
    E_PROTOCOL_VIOLATION,
    E_CAPACITY_EXCEEDED,
    E_LEASE_LIMIT_EXCEEDED,
]

WATERMARK_RULE_METADATA = {
    "scale": 1000,
    "ratio_unit": "permille",
    "threshold_rounding": "half_up",
    "exact_half_tie": "round_up",
    "boundary_rule": "at_or_above_threshold",
    "admission_rule": "reject_when_projected_occupancy_exceeds_hard_limit",
    "hard_rejection_code": E_CAPACITY_EXCEEDED,
    "victim_order": [
        "priority_desc",
        "expires_ms_asc",
        "artifact_key_asc",
    ],
    "authority_order": [
        "capacity_version_desc",
        "write_elapsed_ms_asc",
        "content_digest_asc_present_first",
        "node_id_asc_when_digest_absent_or_final_tie",
    ],
    "pending_order": ["artifact_key_asc", "arrival_seq_asc", "application_id_asc"],
    "cache_ledger_order": ["node_id_asc", "artifact_key_asc", "stage_asc"],
}
