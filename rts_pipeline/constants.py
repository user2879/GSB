"""Stable identifiers shared by the generator, engine, and reports."""

PIPELINE_VERSION = "1.0.0"
SCHEMA_VERSION = 1
LEDGER_SCHEMA_VERSION = 1
ENGINE_VERSION = "rts-engine-sha256-v1"
RNG_VERSION = "sha256-u64-v1"
CHUNK_VERSION = "tick-range-v1"
SCHEDULER_VERSION = "budgeted-threadpool-v1"

ACTIONS = ("collect", "produce", "dispatch")
UNIT_TYPES = ("infantry", "scout", "tank")

ERROR_CODES = (
    "E_SEQ_GAP",
    "E_RNG_DRIFT",
    "E_RESOURCE_OVERFLOW",
    "E_BUDGET_DENIED",
    "E_MALFORMED_EVENT",
    "E_LEDGER_HASH",
    "E_CHUNK_BOUNDARY",
    "E_SCHEMA",
    "E_IO",
    "E_INTERNAL",
)

ERROR_EXIT_CODES = {
    "OK": 0,
    "E_MALFORMED_EVENT": 20,
    "E_SEQ_GAP": 21,
    "E_RNG_DRIFT": 22,
    "E_RESOURCE_OVERFLOW": 23,
    "E_BUDGET_DENIED": 24,
    "E_LEDGER_HASH": 25,
    "E_CHUNK_BOUNDARY": 26,
    "E_SCHEMA": 27,
    "E_IO": 28,
    "E_INTERNAL": 29,
}

LEDGER_FILENAME = "parallel_ledger.json"
SERIAL_LEDGER_FILENAME = "serial_ledger.json"
EXIT_REPORT_FILENAME = "exit_report.json"

