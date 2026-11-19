"""Optional NumPy/PyTorch CPU environment smoke check."""

from __future__ import annotations

import json
import platform

import numpy
import torch


report = {
    "python": platform.python_version(),
    "numpy": numpy.__version__,
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "torch_threads": torch.get_num_threads(),
}
print(json.dumps(report, sort_keys=True, separators=(",", ":")))

