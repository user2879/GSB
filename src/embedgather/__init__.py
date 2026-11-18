"""Incremental Embedding + Gather operator component.

Deterministic, local-only implementation for recommendation/retrieval
embedding tables with block layout, fixed-point gather aggregation,
staged snapshots, incremental invalidation and fault injection.
"""

__version__ = "1.0.0"
