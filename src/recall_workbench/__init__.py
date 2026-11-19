"""Deterministic multi-channel recall and fixed-point fusion workbench."""

from .fusion import run_fusion_for_user
from .models import CHANNEL_ORDER
from .recall import run_recall_for_user

__all__ = ["CHANNEL_ORDER", "run_recall_for_user", "run_fusion_for_user"]
__version__ = "1.0.0"
