"""Business error type carrying a stable business code."""


class DiagError(Exception):
    """Error with a business code, optional field/line/seq localization."""

    def __init__(self, code, message, field=None, line=None, seq=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.field = field
        self.line = line
        self.seq = seq

    def __str__(self):
        parts = [self.code, self.message]
        if self.field is not None:
            parts.append(f"field={self.field}")
        if self.line is not None:
            parts.append(f"line={self.line}")
        if self.seq is not None:
            parts.append(f"seq={self.seq}")
        return " | ".join(parts)
