"""Fixed-point error bound derivation and per-metric attribution (requirement 5).

Error components, composed in topological stage order using exact Fraction
arithmetic (all quantities in the *output ulp* of the final stage):

  quantization      - single-value quantization, half ulp of the record scale
                      (ROUND_HALF_EVEN), introduced at ingest
  window_boundary   - sliding-window boundary attribution error: a boundary
                      record may enter/leave the window, contributing at most
                      the per-record input error divided by the window width
  bucket_boundary   - quantile-bucket boundary attribution error: a value may
                      fall on either side of a bucket edge, half a bucket

Per-stage propagation rules (e_in = cumulative bound at stage input, in that
stage's input ulp; results expressed in that stage's output ulp):
  std_diff:        e_out = 2 * e_in            (exact integer subtraction)
  window_avg:      e_out = e_in + 1 + e_in / w (mean of errors + floor
                   truncation <= 1 ulp + window boundary term)
  quantile_bucket: e_out = e_in * b / (hi-lo) * in_ulp / out_ulp + 1/2
  threshold_norm:  e_out = e_in * scale / span + 1

Attribution: each stage's *introduced* component (truncation / boundary terms
it adds, converted to final-stage ulp) divided by the final cumulative bound
gives its contribution share. The dominant stage is the max-share stage whose
share reaches spec.dominance_threshold_ppm; otherwise dominance is "none".
The threshold is explicit in every report.

Error models are computed per spec_version; the ledger keeps every version so
old and new bounds can be compared side by side.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

COMPONENT_LABELS = ("quantization", "window_boundary", "bucket_boundary", "truncation")


@dataclass(frozen=True)
class StageErrorNode:
    stage_id: str
    kind: str
    introduced_ulp: Fraction  # error introduced here, in this stage's output ulp
    cumulative_ulp: Fraction  # total bound at this stage's output, in its output ulp
    components: tuple  # (label, Fraction in output ulp) pairs introduced here


@dataclass(frozen=True)
class MetricErrorReport:
    metric_key: str
    final_bound_ulp: Fraction  # cumulative bound at final stage output, final ulp
    shares: tuple  # (stage_id, Fraction of final bound) in topo order
    dominant_stage: str | None
    dominance_threshold_ppm: int
    stage_nodes: tuple


def _stage_ulp_chain(spec, metric_record_scale):
    """Propagate the error bound along the topo-ordered chain.

    Returns (nodes, cumulative bound in final-stage output ulp, per-stage
    introduced error converted to final-stage output ulp).
    """
    from .spec import topo_order

    ordered = topo_order(spec.stages)
    # ulp[value-unit] conversions: a stage's output ulp in units of its input
    # ulp depends on the transform gain; we track cumulative error in each
    # stage's own output ulp and the gain to convert to the next stage's ulp.
    nodes = []
    # cumulative error at current point, in current ulp units
    cumulative = Fraction(1, 2 * metric_record_scale)  # ingest quantization, record-scale ulp
    introduced_here = []
    for stage in ordered:
        kind = stage.kind
        if kind == "std_diff":
            introduced = Fraction(0)
            cumulative = 2 * cumulative
            components = ()
        elif kind == "window_avg":
            width = stage.params["window"]
            boundary = cumulative / width
            truncation = Fraction(1)
            introduced = truncation + boundary
            cumulative = cumulative + truncation + boundary
            components = (("window_boundary", boundary), ("truncation", truncation))
        elif kind == "quantile_bucket":
            buckets = stage.params["buckets"]
            span = stage.params["hi"] - stage.params["lo"]
            # input error in value units -> bucket-index units
            cumulative = cumulative * buckets / span
            boundary = Fraction(1, 2)
            introduced = boundary
            cumulative = cumulative + boundary
            components = (("bucket_boundary", boundary),)
        elif kind == "threshold_norm":
            span = stage.params["span"]
            cumulative = cumulative * stage.scale / span
            truncation = Fraction(1)
            introduced = truncation
            cumulative = cumulative + truncation
            components = (("truncation", truncation),)
        else:  # pragma: no cover
            raise ValueError(kind)
        nodes.append(StageErrorNode(stage.id, kind, introduced, cumulative, components))
        introduced_here.append((stage.id, introduced))
    return nodes, cumulative, introduced_here


def _gain_after(spec, stage_id):
    """Multiplicative gain converting one output ulp of stage_id into final
    stage output ulps (for attribution in common units)."""
    from .spec import topo_order

    ordered = topo_order(spec.stages)
    gain = Fraction(1)
    active = False
    for stage in ordered:
        if active:
            if stage.kind == "std_diff":
                gain *= 2
            elif stage.kind == "quantile_bucket":
                gain *= Fraction(stage.params["buckets"], stage.params["hi"] - stage.params["lo"])
            elif stage.kind == "threshold_norm":
                gain *= Fraction(stage.scale, stage.params["span"])
        if stage.id == stage_id:
            active = True
    return gain


def derive_error_model(spec, metric_record_scales):
    """Derive per-metric error reports for a spec.

    metric_record_scales: {metric_key: record scale} from the normalized
    records (v3 scale or filled default).
    """
    reports = {}
    for metric in sorted(metric_record_scales):
        nodes, final_bound, introduced = _stage_ulp_chain(spec, metric_record_scales[metric])
        shares = []
        for stage_id, introduced_ulp in introduced:
            converted = introduced_ulp * _gain_after(spec, stage_id)
            share = converted / final_bound if final_bound > 0 else Fraction(0)
            shares.append((stage_id, share))
        dominant = None
        if shares:
            best_stage, best_share = max(shares, key=lambda item: (item[1], -_stage_order(spec, item[0])))
            if best_share * 1000000 >= spec.dominance_threshold_ppm:
                dominant = best_stage
        reports[metric] = MetricErrorReport(
            metric_key=metric,
            final_bound_ulp=final_bound,
            shares=tuple(shares),
            dominant_stage=dominant,
            dominance_threshold_ppm=spec.dominance_threshold_ppm,
            stage_nodes=tuple(nodes),
        )
    return reports


def _stage_order(spec, stage_id):
    for stage in spec.stages:
        if stage.id == stage_id:
            return stage.order
    return -1


def report_to_dict(report):
    return {
        "metric_key": report.metric_key,
        "final_bound_ulp": str(report.final_bound_ulp),
        "shares": [(sid, str(share)) for sid, share in report.shares],
        "dominant_stage": report.dominant_stage,
        "dominance_threshold_ppm": report.dominance_threshold_ppm,
    }


def compare_versions(old_reports, new_reports):
    """Side-by-side comparison of two versioned error models."""
    diff = []
    for metric in sorted(set(old_reports) | set(new_reports)):
        old = old_reports.get(metric)
        new = new_reports.get(metric)
        diff.append(
            {
                "metric_key": metric,
                "old_bound": str(old.final_bound_ulp) if old else None,
                "new_bound": str(new.final_bound_ulp) if new else None,
                "old_dominant": old.dominant_stage if old else None,
                "new_dominant": new.dominant_stage if new else None,
            }
        )
    return diff
