"""Versioned processing state and fixed pipeline constants."""

from dataclasses import asdict, dataclass, field

from .errors import StateVersionError

STATE_SCHEMA_VERSION = 1
LEDGER_SCHEMA_VERSION = 1

STAGES = (
    "load",        # fp16 file -> float32 linear RGB
    "normalize",   # negative clamp
    "colorspace",  # source primaries -> target primaries
    "exposure",    # linear exposure compensation
    "tonemap",     # Reinhard / ACES fit
    "transfer",    # OETF encode to [0,1] float32
    "quantize",    # -> uint8
)
STAGE_INDEX = {name: idx for idx, name in enumerate(STAGES)}

# Each parameter change invalidates the stage at which it first takes effect
# and every stage downstream.
PARAM_ENTRY_STAGE = {
    "colorspace": "colorspace",
    "exposure_ev": "exposure",
    "tonemap_operator": "tonemap",
    "tonemap_white": "tonemap",
    "tonemap_a": "tonemap",
    "tonemap_b": "tonemap",
    "tonemap_c": "tonemap",
    "tonemap_d": "tonemap",
    "tonemap_e": "tonemap",
    "transfer": "transfer",
    "transfer_gamma": "transfer",
    "dither": "quantize",
}

VALID_COLORSPACES = ("bt709", "bt2020")
VALID_TONEMAPS = ("reinhard", "aces")
VALID_TRANSFERS = ("srgb", "gamma")

# Fixed ACES fitted-curve constants (Narkowicz 2015 RRTAndODTFit).
ACES_A = 2.51
ACES_B = 0.03
ACES_C = 2.43
ACES_D = 0.59
ACES_E = 0.14

# Tone mapping domain. Inputs outside [0, TONEMAP_INPUT_MAX] are clamped
# (deterministic highlight rolloff/saturation); non-finite input raises
# E_TONEMAP_DOMAIN instead of producing undefined output.
TONEMAP_INPUT_MAX = 64.0
TONEMAP_OUTPUT_MIN = 0.0
TONEMAP_OUTPUT_MAX = 1.0

# Transfer gamma legal domain (inclusive).
GAMMA_MIN = 0.1
GAMMA_MAX = 10.0

# Quantization step for the 8-bit encoding, dither off => deterministic.
QUANT_STEP = 1.0 / 255.0

# Tolerances used by analytical reference checks (declared acceptance band).
TRANSFER_MONO_TOL = 1e-6
TRANSFER_ENDPOINT_TOL = 1e-6
TONEMAP_MONO_TOL = 1e-6


@dataclass(frozen=True)
class ProcessingState:
    """Fully parameterized, versioned color pipeline state."""

    colorspace: str = "bt709"
    exposure_ev: float = 0.0
    tonemap_operator: str = "reinhard"
    tonemap_white: float = 1.0
    tonemap_a: float = ACES_A
    tonemap_b: float = ACES_B
    tonemap_c: float = ACES_C
    tonemap_d: float = ACES_D
    tonemap_e: float = ACES_E
    transfer: str = "srgb"
    transfer_gamma: float = 2.2
    dither: bool = False
    schema_version: int = STATE_SCHEMA_VERSION

    def validate(self):
        if self.schema_version != STATE_SCHEMA_VERSION:
            raise StateVersionError(
                f"unsupported state schema version {self.schema_version}",
                details={"expected": STATE_SCHEMA_VERSION},
            )
        if self.colorspace not in VALID_COLORSPACES:
            from .errors import InjectBoundsError

            raise InjectBoundsError(
                f"illegal colorspace {self.colorspace!r}",
                details={"allowed": list(VALID_COLORSPACES)},
            )
        if self.tonemap_operator not in VALID_TONEMAPS:
            from .errors import InjectBoundsError

            raise InjectBoundsError(
                f"illegal tonemap operator {self.tonemap_operator!r}",
                details={"allowed": list(VALID_TONEMAPS)},
            )
        if self.transfer not in VALID_TRANSFERS:
            from .errors import InjectBoundsError

            raise InjectBoundsError(
                f"illegal transfer {self.transfer!r}",
                details={"allowed": list(VALID_TRANSFERS)},
            )
        if not GAMMA_MIN <= float(self.transfer_gamma) <= GAMMA_MAX:
            from .errors import InjectBoundsError

            raise InjectBoundsError(
                f"transfer gamma {self.transfer_gamma} outside "
                f"[{GAMMA_MIN}, {GAMMA_MAX}]",
                code="E_INJECT_BOUNDS",
                details={"gamma": self.transfer_gamma},
            )
        if float(self.tonemap_white) <= 0.0:
            from .errors import InjectBoundsError

            raise InjectBoundsError(
                f"tonemap white point must be positive, got {self.tonemap_white}",
                details={"white": self.tonemap_white},
            )
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        allowed = {f for f in cls.__dataclass_fields__}  # noqa: F841
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise StateVersionError(
                f"unknown state fields {sorted(unknown)}",
                details={"unknown": sorted(unknown)},
            )
        obj = cls(**data)
        return obj

    def changed_parameters(self, other):
        """Return parameter names whose values differ between two states."""
        a, b = self.to_dict(), other.to_dict()
        return sorted(k for k in a if a[k] != b[k])
