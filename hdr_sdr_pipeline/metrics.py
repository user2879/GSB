"""Quality metrics with an independent recomputation contract.

Every metric below is computed solely from:
  * the delivered 8-bit frame (uint8 HxWx3), and
  * the truth manifest (reference linear sample values, frame luma stats,
    fixed region bounds and sample points).

No internal pipeline intermediate is consulted, so a third party holding only
``frames_out`` and ``truth.json`` can reproduce every number byte-for-byte.

Declared conventions:
  * code values are normalized as code / 255.0 (float64).
  * reference linear values are mapped through the *same documented
    Reinhard/ACES + sRGB/gamma formulas* as scalar analytic references, which
    only need the processing state parameters (part of the ledger), never
    pipeline intermediates.
  * reductions use float64, row-major order, equal per-pixel weight within a
    region and equal per-frame weight within an interval.
  * relative errors use a fixed epsilon floor to remain finite near black.
"""

import numpy as np

from .colorspace import luma_coeffs
from .tonemap import aces_scalar, reinhard_scalar
from .transfer import gamma_scalar, srgb_scalar

REL_EPS = 1e-3


def decode_codes(uint8_frame):
    return uint8_frame.astype(np.float64) / 255.0


def _region_mask(shape, bounds):
    r0, r1, c0, c1 = bounds
    mask = np.zeros(shape[:2], dtype=bool)
    mask[r0:r1, c0:c1] = True
    return mask


def reference_code(rgb_linear, state):
    """Analytic reference code value for one linear RGB triple (float64)."""
    rgb = [max(0.0, float(v)) * (2.0 ** float(state.exposure_ev))
           for v in rgb_linear]
    out = []
    for v in rgb:
        if state.tonemap_operator == "reinhard":
            t = reinhard_scalar(v, state.tonemap_white)
        else:
            t = aces_scalar(v, state.tonemap_a, state.tonemap_b,
                            state.tonemap_c, state.tonemap_d, state.tonemap_e)
        t = min(1.0, max(0.0, t))
        if state.transfer == "srgb":
            out.append(min(1.0, max(0.0, srgb_scalar(t))))
        else:
            out.append(min(1.0, max(0.0,
                                    gamma_scalar(t, state.transfer_gamma))))
    return out


def _luma64(rgb_norm, colorspace):
    kr, kg, kb = luma_coeffs(colorspace)
    return rgb_norm[..., 0] * kr + rgb_norm[..., 1] * kg + rgb_norm[..., 2] * kb


def compute_frame_metrics(uint8_frame, frame_truth, state, regions):
    """Compute the per-frame metric dictionary from outputs + truth only."""
    norm = decode_codes(uint8_frame)
    luma_out = _luma64(norm, state.colorspace)

    # 1. sampled-pixel luma bias against analytic references.
    sample_bias = []
    sample_rows = []
    for sample in frame_truth["samples"]:
        r, c = int(sample["row"]), int(sample["col"])
        ref_rgb = reference_code(sample["rgb"], state)
        kr, kg, kb = luma_coeffs(state.colorspace)
        ref_y = ref_rgb[0] * kr + ref_rgb[1] * kg + ref_rgb[2] * kb
        obs_y = float(luma_out[r, c])
        sample_bias.append(abs(obs_y - ref_y))
        sample_rows.append(
            {
                "row": r,
                "col": c,
                "observed_luma": obs_y,
                "reference_luma": ref_y,
                "abs_bias": abs(obs_y - ref_y),
            }
        )
    sample_luma_bias = float(
        np.sum(np.asarray(sample_bias, dtype=np.float64)) / len(sample_bias)
    )

    # 2. shadow / highlight region relative error, pixels weighted equally.
    def region_relative_error(bounds):
        mask = _region_mask(norm.shape, bounds)
        observed = luma_out[mask]
        # Reference luma per pixel cannot be rebuilt without the linear frame,
        # so region truth anchors use the manifest's frame-level luma stats:
        # the analytic tonemap+OETF of the stored reference peak/mean luma.
        return observed

    shadow_obs = region_relative_error(regions["shadow"])
    highlight_obs = region_relative_error(regions["highlight"])

    ref_peak_code = reference_code(
        [frame_truth["peak_luma"], 0.0, 0.0], state
    )[0]
    ref_mean_code = reference_code(
        [frame_truth["mean_luma"], 0.0, 0.0], state
    )[0]

    shadow_rel = float(
        np.sum(np.abs(shadow_obs - ref_mean_code) /
               (np.abs(ref_mean_code) + REL_EPS)) / shadow_obs.size
    )
    highlight_rel = float(
        np.sum(np.abs(highlight_obs - ref_peak_code) /
               (np.abs(ref_peak_code) + REL_EPS)) / highlight_obs.size
    )

    # 3. full-frame delivered luma peak / mean, with bias vs analytic truth.
    peak_obs = float(np.max(luma_out))
    mean_obs = float(np.sum(luma_out, dtype=np.float64) / luma_out.size)

    return {
        "frame": int(frame_truth["frame"]),
        "pixel_range": "all %dx%d pixels, region masks from manifest"
                       % (norm.shape[0], norm.shape[1]),
        "weights": "equal per-pixel weight inside a region",
        "sample_luma_bias": sample_luma_bias,
        "shadow_rel_error": shadow_rel,
        "highlight_rel_error": highlight_rel,
        "peak_luma_observed": peak_obs,
        "peak_luma_reference": ref_peak_code,
        "peak_luma_bias": abs(peak_obs - ref_peak_code),
        "mean_luma_observed": mean_obs,
        "mean_luma_reference": ref_mean_code,
        "mean_luma_bias": abs(mean_obs - ref_mean_code),
        "sample_detail": sample_rows,
    }


def compute_interval_metrics(frame_metric_list):
    """Pool per-frame metrics over a fixed interval, equal per-frame weight."""
    if not frame_metric_list:
        raise ValueError("empty interval")
    keys = (
        "sample_luma_bias",
        "shadow_rel_error",
        "highlight_rel_error",
        "peak_luma_bias",
        "mean_luma_bias",
    )
    pooled = {
        "frames": [m["frame"] for m in frame_metric_list],
        "weights": "equal per-frame weight",
    }
    for key in keys:
        vals = np.asarray([m[key] for m in frame_metric_list], dtype=np.float64)
        pooled[key + "_mean"] = float(np.sum(vals) / vals.size)
        pooled[key + "_max"] = float(np.max(vals))
    return pooled


# Fixed reporting intervals (half, quarter, whole sequence) used by examples.
def default_intervals(frame_count):
    return {
        "whole": [0, frame_count],
        "first_half": [0, frame_count // 2],
        "second_half": [frame_count // 2, frame_count],
        "first_quarter": [0, frame_count // 4],
    }
