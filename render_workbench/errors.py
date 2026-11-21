"""Business error codes used across the rendering workbench."""

E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_TOKEN_OUT_OF_VOCAB = "E_TOKEN_OUT_OF_VOCAB"
E_PATCH_OUT_OF_RANGE = "E_PATCH_OUT_OF_RANGE"
E_INPUT_MISSING = "E_INPUT_MISSING"
E_INIT_IMAGE_INVALID = "E_INIT_IMAGE_INVALID"


class WorkbenchError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message

    def to_dict(self):
        return {"code": self.code, "message": self.message}
