"""Cross-component platform configuration distribution and recovery system."""

__version__ = "1.0.0"
CURRENT_FORMAT_VERSION = 3

STATE_PREPARED = "prepared"
STATE_DISTRIBUTING = "distributing"
STATE_COMMITTED = "committed"
STATE_FAILED = "failed"
STATE_MACHINE_STATES = (
    STATE_PREPARED,
    STATE_DISTRIBUTING,
    STATE_COMMITTED,
    STATE_FAILED,
)
