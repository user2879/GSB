"""Deterministic multi-band remote-sensing tile segmentation.

The public entry points are intentionally small:

* :class:`remote_seg.engine.SegmentationEngine`
* :func:`remote_seg.samplegen.generate_sample_workspace`
* :func:`remote_seg.edits.apply_edit_sequence`
* :func:`remote_seg.injections.materialize_injections`
"""

__all__ = ["RunOptions", "SegmentationEngine"]
__version__ = "1.0.0"


def __getattr__(name: str):
    if name in __all__:
        from .engine import RunOptions, SegmentationEngine

        return {"RunOptions": RunOptions, "SegmentationEngine": SegmentationEngine}[name]
    raise AttributeError(name)
