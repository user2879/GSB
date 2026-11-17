"""Business error codes and the unified BusError exception."""

E_SPEC = "E_SPEC"
E_STATE = "E_STATE"
E_PROGRESS = "E_PROGRESS"
E_LEDGER = "E_LEDGER"
E_REPLAY = "E_REPLAY"

EXIT_CODES = {
    E_SPEC: 2,
    E_STATE: 3,
    E_PROGRESS: 4,
    E_LEDGER: 5,
    E_REPLAY: 6,
}


class BusError(Exception):
    """Business error carrying a stable code and locating context."""

    def __init__(self, code, message, *, field=None, subscriber_id=None, event_seq=None):
        super().__init__("[%s] %s" % (code, message))
        self.code = code
        self.message = message
        self.field = field
        self.subscriber_id = subscriber_id
        self.event_seq = event_seq

    def to_dict(self):
        return {
            "code": self.code,
            "message": self.message,
            "field": self.field,
            "subscriber_id": self.subscriber_id,
            "event_seq": self.event_seq,
        }
