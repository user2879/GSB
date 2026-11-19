"""Business error codes for the terrain workbench.

Every abnormal path raises TerrainError carrying a stable business code and,
where applicable, the first violating block coordinate. Nothing is silently
skipped and nothing crashes with a bare exception.
"""

E_HEIGHT_RANGE = "E_HEIGHT_RANGE"        # illegal elevation range requested
E_LOD_LEVEL = "E_LOD_LEVEL"              # LOD / density level out of bounds
E_CRACK_GAP = "E_CRACK_GAP"              # seam gap detected between blocks
E_BUDGET_DENIED = "E_BUDGET_DENIED"      # budget cannot be satisfied
E_TEXTURE_RESIDENT = "E_TEXTURE_RESIDENT"  # single texture exceeds memory budget
E_TRACK_DIVERGE = "E_TRACK_DIVERGE"      # camera track leaves valid domain
E_GL_CONTEXT = "E_GL_CONTEXT"            # no GL context; software path used

ALL_CODES = (
    E_HEIGHT_RANGE,
    E_LOD_LEVEL,
    E_CRACK_GAP,
    E_BUDGET_DENIED,
    E_TEXTURE_RESIDENT,
    E_TRACK_DIVERGE,
    E_GL_CONTEXT,
)


class TerrainError(Exception):
    """Business exception with a stable code and structured context."""

    def __init__(self, code, message, block=None, **context):
        if code not in ALL_CODES:
            raise ValueError(f"unknown terrain error code: {code}")
        self.code = code
        if isinstance(block, (list, tuple)):
            block = tuple(block)
        self.block = block
        self.context = dict(context)
        super().__init__(f"[{code}] {message} (block={self.block}, ctx={self.context})")

    def to_dict(self):
        return {
            "code": self.code,
            "message": str(self.args[0]) if self.args else "",
            "block": list(self.block) if self.block is not None else None,
            "context": {k: _jsonable(v) for k, v in sorted(self.context.items())},
        }


def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)
