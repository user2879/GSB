"""Structured pipeline error with a stable business code."""


class PipelineError(Exception):
    """Error carrying a business code (E_FORMAT/E_LOG/...) and details."""

    def __init__(self, code, message, **details):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.details = details

    def to_dict(self):
        return {"code": self.code, "message": self.message, "details": self.details}
