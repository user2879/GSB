"""Multimodal evidence collaborative reasoning pipeline.

A fully local, deterministic fixed-point pipeline that fuses visual region
evidence with structured text assertions under an explicit evidence relation
table. All conclusions are content-addressed so incremental runs converge to
the exact same ledger as a clean full rebuild.
"""

__version__ = "1.0.0"
