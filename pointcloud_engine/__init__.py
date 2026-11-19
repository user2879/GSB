"""Local point-cloud offscreen reconstruction renderer.

Pure standard-library + NumPy implementation. Projects a sparse colored
point cloud (positions, grayscale, normals) into grayscale and depth images
with deterministic tiled scheduling, block-level persistence, digest based
invalidation, interrupt/resume support and fault injection.
"""

from .errors import EngineError, ERROR_CODES

__all__ = ["EngineError", "ERROR_CODES"]
__version__ = "1.0.0"
