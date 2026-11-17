"""Subscriber state machine: idle / processing / committed / failed."""

from .errors import BusError, E_STATE

ALLOWED = {
    "idle": {"processing"},
    "processing": {"committed", "failed"},
    "committed": {"processing"},
    "failed": {"processing"},
}


class SubscriberMachine:
    """In-memory view of one subscriber; always rebuilt from the ledger."""

    def __init__(self, subscriber_id):
        self.subscriber_id = subscriber_id
        self.state = "idle"
        self.transition_count = 0

    def transition(self, to_state, event_seq):
        if to_state not in ALLOWED[self.state]:
            raise BusError(
                E_STATE,
                "illegal transition %s -> %s" % (self.state, to_state),
                subscriber_id=self.subscriber_id, event_seq=event_seq)
        from_state = self.state
        self.state = to_state
        self.transition_count += 1
        return from_state

    def apply_recorded(self, from_state, to_state, event_seq):
        """Apply a ledger row during recovery/replay without re-validating."""
        self.state = to_state
        self.transition_count += 1
