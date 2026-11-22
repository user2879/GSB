"""Shared constants: plugin identifiers, taint effects and truncation labels."""
from __future__ import annotations

# Taint effects accepted by the specification.
TAINT_EFFECTS = frozenset(
    {"NoSchedule", "PreferNoSchedule", "NoExecute"}
)

# The four score plugins, in canonical computation order. The order is part of
# the stable tie-break contract; it is never derived from input ordering.
PLUGIN_NODE_AFFINITY = "node_affinity_soft"
PLUGIN_POD_ANTI_AFFINITY = "pod_anti_affinity_soft"
PLUGIN_RESOURCE_BALANCE = "resource_balance"
PLUGIN_LABEL_LOCALITY = "label_locality"

PLUGIN_ORDER = (
    PLUGIN_NODE_AFFINITY,
    PLUGIN_POD_ANTI_AFFINITY,
    PLUGIN_RESOURCE_BALANCE,
    PLUGIN_LABEL_LOCALITY,
)

PLUGIN_LABELS = {
    PLUGIN_NODE_AFFINITY: "NodeAffinitySoft",
    PLUGIN_POD_ANTI_AFFINITY: "PodAntiAffinitySoft",
    PLUGIN_RESOURCE_BALANCE: "ResourceBalance",
    PLUGIN_LABEL_LOCALITY: "LabelLocality",
}

# Every plugin emits scores on this integer scale.
SCORE_MIN = 0
SCORE_MAX = 100

# Weight basis: weights are basis points (1/10000) and must sum to this value.
WEIGHT_BASIS = 10_000

# Every integer division truncates toward negative infinity (floor) because all
# operands are non-negative; this is the explicitly declared truncation rule.
TRUNCATION = "floor_div"

# SQLite ledger format version written by this build.
LEDGER_FORMAT_VERSION = 1

# Status markers for persisted pod records.
STATUS_COMMITTED = "COMMITTED"
STATUS_UNKNOWN = "UNKNOWN"
