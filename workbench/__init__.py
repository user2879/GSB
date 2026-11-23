"""Gateway rate-limit & auth diagnostic workbench (Windows local, stdlib only)."""

LEDGER_FORMAT_VERSION = 1
SNAPSHOT_FORMAT_VERSION = 1
SCALE_SEMANTICS_VERSION = 1

# Business codes
OK = "OK"
E_SPEC = "E_SPEC"
E_RATE = "E_RATE"
E_AUTH = "E_AUTH"
E_UPSTREAM = "E_UPSTREAM"
E_TIMEOUT = "E_TIMEOUT"
E_RECEIPT = "E_RECEIPT"
E_REPLAY = "E_REPLAY"
UNKNOWN = "UNKNOWN"
