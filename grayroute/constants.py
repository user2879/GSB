"""Supported enumerations and shared constants."""
from __future__ import annotations

SUPPORTED_QUANTIZATIONS = ("none", "int8", "int4", "fp16")
SUPPORTED_PARAM_FORMATS = ("r1", "r2", "r3")
SUPPORTED_RECEIPT_FORMATS = (1, 2, 3)

LOOPBACK_HOST = "127.0.0.1"
PORT_MIN = 1024
PORT_MAX = 65535
WEIGHT_MIN = 0
WEIGHT_MAX = 10000
BUCKETS = 10000

POOL_STABLE = "stable"
POOL_GRAY = "gray"
POOLS = (POOL_STABLE, POOL_GRAY)

LEDGER_FORMAT_VERSION = 1
SNAPSHOT_FORMAT_VERSION = 1
UNKNOWN_BATCH = "unknown"
