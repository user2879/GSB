from __future__ import annotations

import os
import signal
from typing import Any

from .arbitration import arbitrate_key
from .ledger import Ledger
from .spec import Environment, Specs, View
from .util import canonical_sha256, canonical_text


def _normalized_binding(environment: Environment, key: str) -> dict[str, Any] | None:
    binding = environment.bindings.get(key)
    if binding is None:
        return None
    return {
        "required_source_types": list(binding.required_source_types),
        "min_observed_seq": binding.min_observed_seq,
    }


def normalized_environment(environment: Environment) -> dict[str, Any]:
    return {
        "environment_id": environment.environment_id,
        "depth": environment.depth,
        "available_sources": sorted(environment.available_sources),
        "bindings": {key: _normalized_binding(environment, key) for key in sorted(environment.bindings)},
        "sources": {
            source_id: {
                "source_id": source.source_id,
                "source_type": source.source_type,
                "priority_bias": source.priority_bias,
                "observations": {
                    key: {
                        "value_text": observation.value_text,
                        "observed_seq": observation.observed_seq,
                        "captured_at_ms": observation.captured_at_ms,
                    }
                    for key, observation in sorted(source.observations.items())
                },
            }
            for source_id, source in sorted(environment.sources.items())
        },
    }


def normalized_view(view: View) -> dict[str, Any]:
    return {
        "view_id": view.view_id,
        "environments": sorted(view.environments),
        "environment_priority": list(view.environment_priority),
        "environment_weights": {env_id: view.environment_weights[env_id] for env_id in sorted(view.environment_weights)},
        "key_policies": {
            key: {
                "strategy": policy.strategy,
                "tolerance": policy.tolerance,
            }
            for key, policy in sorted(view.key_policies.items())
        },
    }


def session_token(specs: Specs) -> dict[str, Any]:
    return {
        "environment_set_version": specs.environment_set_version,
        "source_priority_version": specs.source_priority_version,
        "source_type_order": list(specs.source_type_order),
        "scale_power": specs.session.scale_power,
        "scale_policy_version": specs.session.scale_policy_version,
        "arbitration_policy_version": specs.session.arbitration_policy_version,
        "truncation": specs.session.truncation,
        "overflow": specs.session.overflow,
    }


def view_input_fingerprint(specs: Specs, view: View) -> tuple[str, dict[str, Any]]:
    relevant = {
        "session": session_token(specs),
        "view": normalized_view(view),
        "environments": {
            env_id: normalized_environment(specs.environments[env_id])
            for env_id in sorted(view.environments)
        },
    }
    return canonical_sha256(relevant), relevant


def key_dependency(specs: Specs, view: View, key: str) -> dict[str, Any]:
    policy = view.key_policies[key]
    environments: dict[str, Any] = {}
    for environment_id in sorted(view.environments):
        environment = specs.environments[environment_id]
        sources: dict[str, Any] = {}
        for source_id, source in sorted(environment.sources.items()):
            observation = source.observations.get(key)
            if observation is None:
                continue
            sources[source_id] = {
                "source_type": source.source_type,
                "priority_bias": source.priority_bias,
                "observation": {
                    "value_text": observation.value_text,
                    "observed_seq": observation.observed_seq,
                    "captured_at_ms": observation.captured_at_ms,
                },
            }
        environments[environment_id] = {
            "depth": environment.depth,
            "available_sources": sorted(environment.available_sources),
            "binding": _normalized_binding(environment, key),
            "sources": sources,
        }
    return {
        "session": session_token(specs),
        "view_scope": {
            "view_id": view.view_id,
            "environments": sorted(view.environments),
            "environment_priority": list(view.environment_priority),
            "environment_weights": {
                env_id: view.environment_weights[env_id] for env_id in sorted(view.environment_weights)
            },
        },
        "view_key_policy": {
            "strategy": policy.strategy,
            "tolerance": policy.tolerance,
        },
        "environments": environments,
    }


def _diff_paths(left: Any, right: Any, path: str = "$", limit: int = 24) -> list[str]:
    if left == right:
        return []
    if isinstance(left, dict) and isinstance(right, dict):
        paths: list[str] = []
        for key in sorted(set(left) | set(right)):
            if key not in left:
                paths.append(f"{path}.{key}<added>")
            elif key not in right:
                paths.append(f"{path}.{key}<removed>")
            else:
                paths.extend(_diff_paths(left[key], right[key], f"{path}.{key}", limit))
            if len(paths) >= limit:
                return sorted(paths)[:limit]
        return sorted(paths)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [f"{path}<length:{len(left)}->{len(right)}>"]
        paths = []
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            paths.extend(_diff_paths(left_item, right_item, f"{path}[{index}]", limit))
            if len(paths) >= limit:
                return sorted(paths)[:limit]
        return sorted(paths)
    return [f"{path}<{canonical_text(left)}->{canonical_text(right)}>"]


def _classify_change(changed_paths: list[str]) -> str:
    for path in changed_paths:
        if "view_key_policy" in path:
            return "arbitration_policy_changed"
    for path in changed_paths:
        if "view_scope" in path:
            return "view_scope_changed"
    for path in changed_paths:
        if "binding" in path:
            return "binding_changed"
    for path in changed_paths:
        if ".sources." in path:
            return "source_chain_changed"
    for path in changed_paths:
        if "available_sources" in path or ".depth" in path:
            return "environment_changed"
    return "session_version_changed"


class ArbiterEngine:
    def __init__(self, specs: Specs, ledger: Ledger) -> None:
        self.specs = specs
        self.ledger = ledger
        self._committed_counter = 0

    def _view_fingerprint(self, view: View) -> str:
        return view_input_fingerprint(self.specs, view)[0]

    def _dependencies(self, view: View) -> dict[str, dict[str, Any]]:
        return {key: key_dependency(self.specs, view, key) for key in sorted(view.key_policies)}

    def _view_summary(self, key_results: dict[str, dict[str, Any]], version: int) -> dict[str, Any]:
        error_influenced = sorted(key for key, result in key_results.items() if result["error_influenced"])
        accepted_counts = {key: len(result["accepted"]) for key, result in sorted(key_results.items())}
        rejected_counts = {key: len(result["rejected"]) for key, result in sorted(key_results.items())}
        aggregation_bounds = {
            key: result["key_aggregation_error_upper_fixed_units"]
            for key, result in sorted(key_results.items())
        }
        return {
            "view_version": version,
            "key_count": len(key_results),
            "error_influenced_keys": error_influenced,
            "error_influenced_key_count": len(error_influenced),
            "accepted_source_counts": accepted_counts,
            "rejected_source_counts": rejected_counts,
            "authoritative_by_key": {
                key: {
                    "value_fixed": result["authoritative"]["value_fixed"],
                    "value_text": result["authoritative"]["value_text"],
                    "origin": result["authoritative"]["origin"],
                }
                for key, result in sorted(key_results.items())
            },
        }

    def _assemble_view_result(
        self,
        view: View,
        version: int,
        fingerprint: str,
        key_results: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        aggregation_bounds = {
            key: result["key_aggregation_error_upper_fixed_units"]
            for key, result in sorted(key_results.items())
        }
        view_bound = max(aggregation_bounds.values()) if aggregation_bounds else 0
        summary = self._view_summary(key_results, version)
        summary["view_aggregation_error_upper_bound_fixed_units"] = view_bound
        return {
            "view_id": view.view_id,
            "view_version": version,
            "fingerprint": fingerprint,
            "status": "complete",
            "environment_set_version": self.specs.environment_set_version,
            "source_priority_version": self.specs.source_priority_version,
            "scale_policy_version": self.specs.session.scale_policy_version,
            "arbitration_policy_version": self.specs.session.arbitration_policy_version,
            "scale_policy": {
                "scale_power": self.specs.session.scale_power,
                "scale_factor": 10**self.specs.session.scale_power,
                "truncation": self.specs.session.truncation,
                "overflow": self.specs.session.overflow,
            },
            "view_aggregation_error_upper_bound_fixed_units": view_bound,
            "per_key_comparison_error_bounds": {
                key: result["error_model"]["comparison_error_bound_fixed_units"]
                for key, result in sorted(key_results.items())
            },
            "per_key_aggregation_error_upper_bounds": aggregation_bounds,
            "keys": key_results,
            "summary": summary,
        }

    def recompute_view(self, view_id: str, version: int) -> dict[str, Any]:
        """Recompute every key without carrying values from a previous version."""
        view = self.specs.views[view_id]
        key_results = {key: arbitrate_key(self.specs, view, key) for key in sorted(view.key_policies)}
        return self._assemble_view_result(view, version, self._view_fingerprint(view), key_results)

    def plan_view(self, view: View) -> dict[str, Any]:
        fingerprint = self._view_fingerprint(view)
        previous = self.ledger.current_view(view.view_id)
        if previous is not None and previous["fingerprint"] == fingerprint:
            return {
                "view_id": view.view_id,
                "action": "unchanged",
                "version": int(previous["current_version"]),
                "fingerprint": fingerprint,
                "previous": previous,
            }
        version = int(previous["current_version"]) + 1 if previous is not None else 1
        return {
            "view_id": view.view_id,
            "action": "recompute",
            "version": version,
            "fingerprint": fingerprint,
            "previous": previous,
        }

    def compute_planned_view(self, plan: dict[str, Any]) -> dict[str, Any]:
        view = self.specs.views[plan["view_id"]]
        dependencies = self._dependencies(view)
        previous = plan["previous"]
        old_results: dict[str, dict[str, Any]] = {}
        old_dependencies: dict[str, dict[str, Any]] = {}
        old_hashes: dict[str, str] = {}
        old_version: int | None = None
        if previous is not None:
            old_version = int(previous["current_version"])
            old_dependencies = previous["dependencies"]
            for row in self.ledger.key_rows(view.view_id, old_version):
                old_results[row["key"]] = row["result"]
                old_hashes[row["key"]] = row["result_sha256"]

        affected: dict[str, dict[str, Any]] = {}
        all_keys = sorted(set(view.key_policies) | set(old_results) | set(old_dependencies))
        for key in all_keys:
            if key not in old_dependencies:
                if key in view.key_policies:
                    affected[key] = {
                        "reason": "key_added",
                        "changed_paths": ["$.keys<added>"],
                        "old_dependency_sha256": None,
                        "new_dependency_sha256": canonical_sha256(dependencies.get(key)),
                        "old_version": old_version,
                        "old_result_sha256": None,
                    }
                continue
            if key not in dependencies or key not in view.key_policies:
                affected[key] = {
                    "reason": "key_removed",
                    "changed_paths": ["$.keys<removed>"],
                    "old_dependency_sha256": canonical_sha256(old_dependencies[key]),
                    "new_dependency_sha256": None,
                    "old_version": old_version,
                    "old_result_sha256": old_hashes.get(key),
                }
                continue
            if old_dependencies[key] != dependencies[key]:
                changed_paths = _diff_paths(old_dependencies[key], dependencies[key])
                affected[key] = {
                    "reason": _classify_change(changed_paths),
                    "changed_paths": changed_paths,
                    "old_dependency_sha256": canonical_sha256(old_dependencies[key]),
                    "new_dependency_sha256": canonical_sha256(dependencies[key]),
                    "old_version": old_version,
                    "old_result_sha256": old_hashes.get(key),
                }

        key_results: dict[str, dict[str, Any]] = {}
        for key in sorted(view.key_policies):
            if previous is not None and key not in affected and key in old_results:
                key_results[key] = old_results[key]
            else:
                key_results[key] = arbitrate_key(self.specs, view, key)

        result = self._assemble_view_result(view, int(plan["version"]), plan["fingerprint"], key_results)
        stale_events = {
            key: {
                **trace,
                "old_version": old_version if old_version is not None else int(plan["version"]),
                "new_version": int(plan["version"]),
            }
            for key, trace in sorted(affected.items())
            if previous is not None and old_version is not None
        }
        return {
            "result": result,
            "key_results": key_results,
            "dependencies": dependencies,
            "stale_events": stale_events,
            "affected_keys": sorted(affected),
        }

    def _test_crash_hook(self, committed_count: int) -> None:
        requested = os.environ.get("CONFIG_ARBITER_TEST_CRASH_AFTER_VIEW")
        if requested is None:
            return
        try:
            threshold = int(requested)
        except ValueError:
            return
        if committed_count == threshold:
            if hasattr(signal, "SIGKILL"):
                os.kill(os.getpid(), signal.SIGKILL)
            os._exit(137)

    def run_all(self, *, verify_digest: bool = True, expected_digest: str | None = None) -> dict[str, Any]:
        if verify_digest:
            self.ledger.verify_integrity(expected_digest=expected_digest)
        reports: list[dict[str, Any]] = []
        committed_count = 0
        for view_id in sorted(self.specs.views):
            plan = self.plan_view(self.specs.views[view_id])
            if plan["action"] == "unchanged":
                reports.append(
                    {
                        "view_id": view_id,
                        "action": "unchanged",
                        "version": plan["version"],
                        "fingerprint": plan["fingerprint"],
                    }
                )
                continue
            computed = self.compute_planned_view(plan)
            digest = self.ledger.commit_view(
                view_id=view_id,
                version=int(plan["version"]),
                fingerprint=plan["fingerprint"],
                result=computed["result"],
                key_results=computed["key_results"],
                dependencies=computed["dependencies"],
                stale_events=computed["stale_events"],
                environment_set_version=self.specs.environment_set_version,
                source_priority_version=self.specs.source_priority_version,
                scale_policy_version=self.specs.session.scale_policy_version,
                arbitration_policy_version=self.specs.session.arbitration_policy_version,
            )
            committed_count += 1
            reports.append(
                {
                    "view_id": view_id,
                    "action": "committed",
                    "version": plan["version"],
                    "fingerprint": plan["fingerprint"],
                    "affected_keys": computed["affected_keys"],
                    "stale_event_count": len(computed["stale_events"]),
                    "ledger_content_digest": digest,
                }
            )
            self._test_crash_hook(committed_count)
        status = self.status_report()
        return {
            "format_version": 1,
            "ledger_content_digest": self.ledger.meta()["content_digest"],
            "views": reports,
            "completed_views": status["completed_views"],
            "unknown_views": status["unknown_views"],
        }

    def status_report(self) -> dict[str, Any]:
        views: list[dict[str, Any]] = []
        completed: list[str] = []
        unknown: list[str] = []
        changed: list[str] = []
        for view_id in sorted(self.specs.views):
            view = self.specs.views[view_id]
            current = self.ledger.current_view(view_id)
            current_fingerprint = self._view_fingerprint(view)
            if current is not None and current["fingerprint"] == current_fingerprint:
                state = "complete"
                completed.append(view_id)
                version = int(current["current_version"])
                result_sha = current["result_sha256"]
            elif current is not None:
                state = "changed"
                changed.append(view_id)
                version = int(current["current_version"]) + 1
                result_sha = None
            else:
                state = "unknown"
                unknown.append(view_id)
                version = 1
                result_sha = None
            views.append(
                {
                    "view_id": view_id,
                    "status": state,
                    "version": version,
                    "fingerprint": current_fingerprint,
                    "result_sha256": result_sha,
                    "requires_rerun": state != "complete",
                    "reason": None
                    if state == "complete"
                    else ("fingerprint_changed" if state == "changed" else "no_complete_committed_record"),
                }
            )
        return {
            "views": views,
            "completed_views": completed,
            "unknown_views": unknown,
            "changed_views": changed,
            "ledger_format_version": int(self.ledger.meta()["format_version"]),
            "ledger_content_digest": self.ledger.meta()["content_digest"],
        }
