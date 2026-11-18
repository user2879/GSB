"""Business error codes shared by every stage of the pipeline."""

#: A document / query / command record failed structural validation.
E_REC_INVALID = "E_REC_INVALID"
#: No citable snippet could be produced for a turn.
E_CITATION_MISSING = "E_CITATION_MISSING"
#: A fresh answer could not be produced inside the first-response budget.
E_ANSWER_BUDGET_PENDING = "E_ANSWER_BUDGET_PENDING"
#: Snapshot manifest version / config hash does not match current inputs.
E_CONFIG_MISMATCH = "E_CONFIG_MISMATCH"

ALL_CODES = (
    E_REC_INVALID,
    E_CITATION_MISSING,
    E_ANSWER_BUDGET_PENDING,
    E_CONFIG_MISMATCH,
)

OK = "OK"

