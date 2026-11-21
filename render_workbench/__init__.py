"""Deterministic local text-to-image / img2img rendering workbench."""

# Force single-threaded native math before numpy is imported anywhere. The
# pipeline parallelizes across pipeline nodes with a thread pool; inside each
# node the math itself must be deterministic and independent of pool size.
import os as _os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
             "BLIS_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")

__version__ = "1.0.0"
