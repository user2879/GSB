"""Parameterized per-frame color pipeline with stage-level content hashes.

Stage order is fixed (see config.STAGES):
    load -> normalize -> colorspace -> exposure -> tonemap -> transfer
    -> quantize
Every float32 intermediate and the final uint8 frame are hashed (SHA-256 over
dtype+shape+raw C-order bytes). Injection hooks only replace values *inside*
this function; injected inputs/parameters enter through the state, manifest
or the injection context, never via hidden side channels.
"""

import os
from dataclasses import dataclass, field

import numpy as np

from .colorspace import apply_matrix
from .config import STAGES, STAGE_INDEX
from .hashing import hash_array
from .quantize import quantize_array
from .synth import FRAME_HEIGHT, FRAME_WIDTH, load_frame
from .tonemap import tonemap_array
from .transfer import encode_array


@dataclass
class InjectionContext:
    """All pipeline-layer injections that can hit a given frame."""

    stage_replacements: dict = field(default_factory=dict)
    # map (frame, stage) -> replacement ndarray (float32 or uint8 for quantize)

    def replacement_for(self, frame_index, stage):
        return self.stage_replacements.get((frame_index, stage))


def process_frame(frame_index, frames_dir, state, *, context=None,
                  width=FRAME_WIDTH, height=FRAME_HEIGHT):
    """Process one frame; return (uint8 frame, ordered stage hashes, stages).

    stage_hashes: dict stage_name -> sha256 hex of that stage's output.
    stage_arrays: dict stage_name -> ndarray (kept in memory by the engine
                  only for stages that need recomputation or localization).
    """
    state.validate()
    context = context or InjectionContext()
    name = "frame_{:04d}.fp16".format(frame_index)
    path = os.path.join(frames_dir, name)
    stage_arrays = {}
    stage_hashes = {}

    def record(stage, arr):
        stage_arrays[stage] = arr
        stage_hashes[stage] = hash_array(arr)

    # 1. load (fp16 -> exact float32 widening)
    loaded = load_frame(path, width=width, height=height)
    record("load", loaded)

    # 2. normalize: deterministic negative clamp to zero
    normalized = np.maximum(loaded, np.float32(0.0))
    record("normalize", normalized)

    # 3. colorspace conversion (fixed matrix, explicit float32 order)
    converted = apply_matrix(normalized, state.colorspace)
    converted = _maybe_replace(frame_index, "colorspace", converted, context,
                               stage_hashes)
    record("colorspace", converted)

    # 4. exposure compensation in stops: multiply by 2**ev in float32
    factor = np.float32(2.0) ** np.float32(state.exposure_ev)
    exposed = converted * factor
    exposed = _maybe_replace(frame_index, "exposure", exposed, context,
                             stage_hashes)
    record("exposure", exposed)

    # 5. tone mapping (frozen formulas, clamp policy, domain check)
    tonemapped = tonemap_array(exposed, state, frame_index)
    tonemapped = _maybe_replace(frame_index, "tonemap", tonemapped, context,
                                stage_hashes)
    record("tonemap", tonemapped)

    # 6. output transfer function
    encoded = encode_array(tonemapped, state)
    encoded = _maybe_replace(frame_index, "transfer", encoded, context,
                             stage_hashes)
    record("transfer", encoded)

    # 7. quantize to uint8 (dither off => deterministic)
    quantized = quantize_array(encoded, state)
    quantized = _maybe_replace(frame_index, "quantize", quantized, context,
                               stage_hashes)
    record("quantize", quantized)

    assert tuple(stage_hashes) == STAGES
    return quantized, stage_hashes, stage_arrays


def _maybe_replace(frame_index, stage, original, context, stage_hashes):
    replacement = context.replacement_for(frame_index, stage)
    if replacement is None:
        return original
    if replacement.shape != original.shape or replacement.dtype != original.dtype:
        from .errors import FrameFormatError

        raise FrameFormatError(
            "injected replacement has incompatible shape/dtype",
            details={"frame": frame_index, "stage": stage,
                     "expected_shape": list(original.shape),
                     "actual_shape": list(replacement.shape),
                     "expected_dtype": str(original.dtype),
                     "actual_dtype": str(replacement.dtype)},
        )
    return replacement
