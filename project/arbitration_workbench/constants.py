"""Stable constants shared by loaders, arbitration and exchange codecs."""

from __future__ import annotations

PROJECT_SCHEMA = "local-upgrade-arbitration/v1"
FIXED_ALGORITHM_VERSION = "fixed-arbitration-v1"
CODEC_VERSION = "exchange-codec-v1"

STAGE_ORDER = ("score", "relations", "decision", "convert")
STAGE_RANK = {name: index for index, name in enumerate(STAGE_ORDER)}

VERSIONS = ("v1", "v2", "v3")

ERROR_JSON_INVALID = "E_JSON_INVALID"
ERROR_EVIDENCE_TYPE_UNKNOWN = "E_EVIDENCE_TYPE_UNKNOWN"
ERROR_ROLE_UNKNOWN = "E_ROLE_UNKNOWN"
ERROR_TIMESTAMP_INVALID = "E_TIMESTAMP_INVALID"
ERROR_DUPLICATE_EVIDENCE = "E_DUPLICATE_EVIDENCE"
ERROR_STAGE_EXCEPTION = "E_STAGE_EXCEPTION"
ERROR_VERSION_LOSSY = "E_VERSION_LOSSY"
ERROR_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
ERROR_INTERACT_BUDGET_PENDING = "E_INTERACT_BUDGET_PENDING"
ERROR_SIDECAR_STALE = "E_SIDECAR_STALE"
ERROR_TEMP_VOLUME_MISMATCH = "E_TEMP_VOLUME_MISMATCH"
ERROR_SNAPSHOT_INVALID = "E_SNAPSHOT_INVALID"

REASON_ORDER = (
    "R_UPGRADE",
    "R_NO_UPGRADE",
    "R_THRESHOLD_BOUNDARY",
    "R_EXCLUSIVE_APPLIED",
    "R_SUPPORT_APPLIED",
)

V1_MAGIC = None
V3_MAGIC = b"ARBV3\x00\x00\x00"
V3_VERSION = 3
V3_FEATURES = (
    "ticket_id",
    "upgrade_code",
    "priority_level",
    "priority_name",
    "precision",
    "reason_codes",
    "missing",
    "boundary",
    "score_fixed",
)
