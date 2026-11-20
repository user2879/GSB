"""定点卷积与饱和截断算子组件。"""

from .attribution import AttributionReport, PositionAttribution, attribute_position
from .calibration import load_calibration
from .cache import PositionCache
from .errors import ALL_BUSINESS_CODES, FailureLedger
from .injections import InjectionEngine, load_injections
from .kernel import PositionResult, compute_position
from .models import OpDesc, load_descriptor
from .pipeline import RunConfig, run_pipeline
from .quantize import dequantize, truncate
from .reference import reference_position
from .tensorpack import TensorPack, expected_main_shapes, load_tensor_pack

__all__ = [
    "ALL_BUSINESS_CODES",
    "AttributionReport",
    "FailureLedger",
    "InjectionEngine",
    "OpDesc",
    "PositionAttribution",
    "PositionCache",
    "PositionResult",
    "RunConfig",
    "TensorPack",
    "attribute_position",
    "compute_position",
    "dequantize",
    "expected_main_shapes",
    "load_calibration",
    "load_descriptor",
    "load_injections",
    "load_tensor_pack",
    "reference_position",
    "run_pipeline",
    "truncate",
]
