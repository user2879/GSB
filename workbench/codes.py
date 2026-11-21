"""Business codes emitted by the workbench.

Every abnormal condition surfaces as one of these codes; nothing is
silently replaced by a default value.
"""

# Load / validation layer (requirement 1)
E_LOAD_JSON = "E_LOAD_JSON"                        # malformed JSON line / file
E_LOAD_TURN_SCHEMA = "E_LOAD_TURN_SCHEMA"          # turn record missing/invalid fields
E_LOAD_TURN_DUP = "E_LOAD_TURN_DUP"                # duplicate turn id
E_LOAD_VOCAB = "E_LOAD_VOCAB"                      # term outside controlled vocabulary
E_LOAD_TOOL_PARAMS = "E_LOAD_TOOL_PARAMS"          # tool call params mismatch schema
E_LOAD_ACOUSTIC_SCHEMA = "E_LOAD_ACOUSTIC_SCHEMA"  # acoustic record invalid
E_LOAD_ACOUSTIC_NONFINITE = "E_LOAD_ACOUSTIC_NONFINITE"  # non-finite acoustic stat
E_LOAD_ACOUSTIC_ORPHAN = "E_LOAD_ACOUSTIC_ORPHAN"  # acoustic row without turn
E_LOAD_ACOUSTIC_MISSING = "E_LOAD_ACOUSTIC_MISSING"  # turn without acoustic row
E_LOAD_TOOL_SCHEMA = "E_LOAD_TOOL_SCHEMA"          # tool manifest entry invalid
E_LOAD_CONFIG = "E_LOAD_CONFIG"                    # config file invalid
E_LOAD_TOL = "E_LOAD_TOL"                          # tolerance not positive finite
E_LOAD_PRECISION = "E_LOAD_PRECISION"              # frac bits / quantum outside allowed set
E_LOAD_BUDGET = "E_LOAD_BUDGET"                    # budgets not positive

# Fusion chain (requirement 3)
E_FUSE_WEIGHT_SUM_ZERO = "E_FUSE_WEIGHT_SUM_ZERO"
E_FUSE_DIM_INCOMPAT = "E_FUSE_DIM_INCOMPAT"
E_FUSE_EVIDENCE_MISSING = "E_FUSE_EVIDENCE_MISSING"
E_FUSE_CLIPPED = "E_FUSE_CLIPPED"

# Scheduler (requirement 6)
E_FUSE_BUDGET_PENDING = "E_FUSE_BUDGET_PENDING"

# Interactive / snapshot (requirement 8)
E_PREVIEW_BUDGET = "E_PREVIEW_BUDGET"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
