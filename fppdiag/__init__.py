"""fppdiag - fixed-point feature pipeline drift diagnostics.

Pure-stdlib, deterministic, integer-only fixed-point feature pipeline.
All numeric transformations use explicit integer arithmetic with declared
rounding, saturation and null propagation rules (see fppdiag.fixedpoint and
fppdiag.stages).
"""

__version__ = "1.0.0"
LEDGER_FORMAT_VERSION = 1
SNAPSHOT_FORMAT_VERSION = 1
