"""Headless text renderer used when no windowing system is available.

The GUI and this renderer consume exactly the same Session data, so rendering
never mutates the ledger.
"""
from __future__ import annotations

from dataclasses import asdict

from .session import Session


def render_session_text(session: Session) -> str:
    lines: list[str] = []
    bounds = session.engine.fp.describe_bounds()
    lines.append("== Fixed-point error budget ==")
    for key in sorted(bounds):
        lines.append(f"  {key}: {bounds[key]}")
    for view_id in sorted(session.conclusions):
        vc = session.conclusions[view_id]
        lines.append("")
        lines.append(f"== View {view_id} (version {vc.version}) ==")
        if vc.added_keys or vc.removed_keys or vc.changed_keys:
            lines.append(
                f"  diff added={vc.added_keys} removed={vc.removed_keys} "
                f"changed={vc.changed_keys}"
            )
        for key in sorted(vc.keys):
            kc = vc.keys[key]
            lines.append(f"  [{key}] policy={kc.policy}")
            lines.append(
                f"    authority_q={kc.authority_q} real={kc.authority_real} "
                f"bound_q={kc.bound_q} adjacent_diff_q={kc.adjacent_diff_q}"
            )
            lines.append(f"    error_influenced={kc.error_influenced} attribution={kc.attribution}")
            lines.append(f"    accepted={kc.accepted_obs_ids}")
            for verdict in kc.rejected:
                lines.append(
                    f"    rejected {verdict.obs_id} reason={verdict.reject_reason} "
                    f"q={verdict.quantized_value}"
                )
            for env_id, rows in sorted(kc.distribution_q.items()):
                vals = ", ".join(f"{row['obs_id']}={row['q']}" for row in rows)
                lines.append(f"    distribution {env_id}: {vals}")
    unknown = session.unknown_views()
    if unknown:
        lines.append("")
        lines.append(f"UNKNOWN (must re-run): {unknown}")
    return "\n".join(lines)
