"""Domain error hierarchy for the graph traversal diagnostic workbench.

Error codes are stable identifiers used by the CLI, GUI and replay layer:

* ``E_GRAPH``  - invalid graph / query declaration (whole load is rejected)
* ``E_SCALE``  - fixed-point scale makes the integer domain truncate totals
* ``E_REPLAY`` - post-restore deterministic replay diverges from the snapshot
* ``E_STATE``  - stale or incompatible session / snapshot state
"""


class GraphDiagError(Exception):
    """Base class carrying a stable machine-readable ``code``."""

    code = "E_INTERNAL"

    def __init__(self, message: str, *, detail: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}


class GraphValidationError(GraphDiagError):
    code = "E_GRAPH"


class ScaleError(GraphDiagError):
    code = "E_SCALE"


class ReplayError(GraphDiagError):
    code = "E_REPLAY"


class StateError(GraphDiagError):
    code = "E_STATE"
