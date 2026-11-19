class BenchError(Exception):
    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def to_json(self):
        return {
            "ok": False,
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            },
        }


ERROR_EXIT_CODES = {
    "E_INPUT": 2,
    "E_BONE": 3,
    "E_HIERARCHY": 4,
    "E_LENGTH": 5,
    "E_CURVE": 6,
    "E_QUATERNION": 7,
    "E_LIMIT": 8,
    "E_TARGET": 9,
    "E_CONTACT": 10,
    "E_PARAM": 11,
    "E_DIGEST": 12,
    "E_SOLVE": 13,
    "E_STATE": 14,
    "E_INTERRUPTED": 130,
}


def exit_code_for(code):
    return ERROR_EXIT_CODES.get(code, 1)
