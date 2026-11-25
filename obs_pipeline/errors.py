"""Typed pipeline errors.

Error codes:
  E_SPEC    - spec/declaration validation failure (reports stage id + field)
  E_FIELD   - record field decode failure (keeps raw line number)
  E_VERSION - record format/scale version failure (keeps raw line number)
"""


class PipelineError(Exception):
    code = "E_UNKNOWN"

    def __init__(self, message, **context):
        super().__init__(message)
        self.message = message
        self.context = context

    def to_dict(self):
        return {"code": self.code, "message": self.message, **self.context}


class SpecError(PipelineError):
    code = "E_SPEC"

    def __init__(self, stage_id, field, message):
        super().__init__(message, stage_id=stage_id, field=field)
        self.stage_id = stage_id
        self.field = field


class FieldError(PipelineError):
    code = "E_FIELD"

    def __init__(self, line_no, field, message):
        super().__init__(message, line_no=line_no, field=field)
        self.line_no = line_no
        self.field = field


class VersionError(PipelineError):
    code = "E_VERSION"

    def __init__(self, line_no, field, message):
        super().__init__(message, line_no=line_no, field=field)
        self.line_no = line_no
        self.field = field
