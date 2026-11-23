"""Domain error hierarchy for the alert noise-reduction pipeline.

Error codes required by the specification:
    E_SPEC      - spec / sample-level invariant violation (graph cycle, dangling
                  dependency, non-monotonic timestamps, duplicate service id,
                  invalid fingerprint, negative silence window at spec level,
                  injected graph cycle).
    E_FIELD     - single-record codec violation (unknown schema version, missing
                  required field, wrong type, negative silence window, dangling
                  root-cause reference).
    E_REPLAY    - first mismatch during post-recovery deterministic replay.
    E_SNAPSHOT  - SQLite ledger content digest mismatch.
Every error carries ``line`` (1-based line / record number, 0 for file-level
issues) and ``field`` so the *first* offending position is always reported.
"""


class PipelineError(Exception):
    code = "E_PIPELINE"

    def __init__(self, message, line=0, field="", detail=""):
        self.line = int(line)
        self.field = field
        self.detail = detail
        loc = []
        if line:
            loc.append(f"line={line}")
        if field:
            loc.append(f"field={field}")
        prefix = f"[{self.code}] " + ("<" + ", ".join(loc) + "> " if loc else "")
        super().__init__(prefix + message + (f" ({detail})" if detail else ""))


class SpecError(PipelineError):
    code = "E_SPEC"


class FieldError(PipelineError):
    code = "E_FIELD"


class ReplayError(PipelineError):
    code = "E_REPLAY"


class SnapshotError(PipelineError):
    code = "E_SNAPSHOT"
