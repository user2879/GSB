from __future__ import annotations

from typing import Any

from .engine import ArbiterEngine, normalized_environment
from .ledger import Ledger
from .spec import Specs


def build_diagnostic_model(specs: Specs, ledger: Ledger) -> dict[str, Any]:
    """Build a deterministic, read-only projection for the UI and tests."""
    engine = ArbiterEngine(specs, ledger)
    status = engine.status_report()
    status_by_view = {item["view_id"]: item for item in status["views"]}

    environments = [
        normalized_environment(specs.environments[environment_id])
        for environment_id in sorted(specs.environments)
    ]
    views: list[dict[str, Any]] = []
    for view_id in sorted(specs.views):
        view_status = status_by_view[view_id]
        current = ledger.current_view(view_id)
        keys: list[dict[str, Any]] = []
        if view_status["status"] == "complete" and current is not None:
            result = current["result"]
            for key in sorted(result["keys"]):
                key_result = result["keys"][key]
                observations = []
                chains = []
                for chain in key_result["chains"]:
                    chains.append(
                        {
                            "environment_id": chain["environment_id"],
                            "depth": chain["depth"],
                            "chain_length": chain["chain_length"],
                            "chain_signature": chain["chain_signature"],
                            "binding": chain["binding"],
                            "steps": [
                                {
                                    "rank": step["rank"],
                                    "source_id": step["source_id"],
                                    "source_type": step["source_type"],
                                    "priority_bias": step["priority_bias"],
                                    "value_text": step["value_text"],
                                    "value_fixed": step["value_fixed"],
                                    "observed_seq": step["observed_seq"],
                                    "captured_at_ms": step["captured_at_ms"],
                                    "binding_eligible": step["binding_eligible"],
                                    "binding_failures": step["binding_failures"],
                                    "decision": step["decision"],
                                }
                                for step in chain["steps"]
                            ],
                        }
                    )
                    effective = chain["effective"]
                    observations.append(
                        {
                            "environment_id": chain["environment_id"],
                            "source_id": None if effective is None else effective["source_id"],
                            "source_type": None if effective is None else effective["source_type"],
                            "value_text": None if effective is None else effective["value_text"],
                            "value_fixed": None if effective is None else effective["value_fixed"],
                            "observed_seq": None if effective is None else effective["observed_seq"],
                            "captured_at_ms": None if effective is None else effective["captured_at_ms"],
                        }
                    )
                keys.append(
                    {
                        "key": key,
                        "strategy": key_result["strategy"],
                        "policy": key_result["policy"],
                        "authoritative": key_result["authoritative"],
                        "accepted": key_result["accepted"],
                        "rejected": key_result["rejected"],
                        "chains": sorted(chains, key=lambda item: item["environment_id"]),
                        "observations": sorted(observations, key=lambda item: item["environment_id"]),
                        "error_model": key_result["error_model"],
                        "view_aggregation_error_upper_bound_fixed_units": result[
                            "view_aggregation_error_upper_bound_fixed_units"
                        ],
                        "error_influenced": key_result["error_influenced"],
                        "error_attribution": key_result["error_attribution"],
                        "key_aggregation_error_upper_fixed_units": key_result[
                            "key_aggregation_error_upper_fixed_units"
                        ],
                    }
                )
        history = ledger.view_history(view_id)
        stale = ledger.stale_findings(view_id, int(view_status["version"]))
        views.append(
            {
                "view_id": view_id,
                "status": view_status["status"],
                "version": view_status["version"],
                "requires_rerun": view_status["requires_rerun"],
                "reason": view_status["reason"],
                "fingerprint": view_status["fingerprint"],
                "result_sha256": view_status["result_sha256"],
                "keys": keys,
                "history": history,
                "stale_findings": stale,
            }
        )
    return {
        "ledger_format_version": status["ledger_format_version"],
        "ledger_content_digest": status["ledger_content_digest"],
        "completed_views": status["completed_views"],
        "unknown_views": status["unknown_views"],
        "changed_views": status["changed_views"],
        "environments": environments,
        "views": views,
    }
