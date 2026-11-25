"""Multi-environment deployment configuration arbitration platform.

Deterministic, order-independent arbitration across local environments using
real TCP subprocess services, fixed-point quantization, adaptive full vs
incremental arbitration, invalidation propagation, quota-constrained thread
scheduling and a versioned SQLite arbitration ledger.
"""

__version__ = "1.0.0"
LEDGER_FORMAT_VERSION = 1
