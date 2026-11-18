"""Handleable exit signals (SIGTERM/SIGINT) as a cooperative flag."""

import signal


class ExitFlag:
    def __init__(self):
        self.requested = False
        self.signum = None

    def install(self):
        def handler(signum, _frame):
            self.requested = True
            self.signum = signum

        for name in ("SIGTERM", "SIGINT"):
            sig = getattr(signal, name, None)
            if sig is None:
                continue
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass
        return self
