from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .canonical import sha256_json
from .diagnostics import (
    detect_parameter_flips,
    diagnostics_digest,
    flatten_flips_for_ledger,
    summarize_analyses,
)
from .generator import RecordStream, generate_records, summarize_records
from .sampling import SamplingResult, sample_records
from .spec import PoliciesDoc, SamplingPolicy, ServicesDoc, ServiceSpec
from .stats import analyze_policy


@dataclass(frozen=True)
class AnalysisBundle:
    services_doc: ServicesDoc
    policies_doc: PoliciesDoc
    service_analyses: tuple[dict[str, Any], ...]
    diagnostics: dict[str, Any]
    flip_events: tuple[dict[str, Any], ...]
    ledger_flips: tuple[dict[str, Any], ...]
    bundle_digest: str

    def to_report(self) -> dict[str, Any]:
        return {
            "report_format": 1,
            "service_version": self.services_doc.version,
            "strategy_version": self.policies_doc.version,
            "services_digest": self.services_doc.content_digest,
            "policies_digest": self.policies_doc.content_digest,
            "bundle_digest": self.bundle_digest,
            "services": list(self.service_analyses),
            "diagnostics": self.diagnostics,
            "flip_events": list(self.flip_events),
            "ledger_flips": list(self.ledger_flips),
        }


class Analyzer:
    def __init__(self, services_doc: ServicesDoc, policies_doc: PoliciesDoc) -> None:
        self.services_doc = services_doc
        self.policies_doc = policies_doc
        self.streams: dict[str, RecordStream] = {
            service.id: generate_records(service) for service in services_doc.services
        }
        self.stream_summaries: dict[str, dict[str, Any]] = {
            service.id: summarize_records(service, self.streams[service.id].records)
            for service in services_doc.services
        }
        self._sampling_cache: dict[tuple[str, str, str], SamplingResult] = {}
        self._policy_cache: dict[tuple[str, str, str], dict[str, Any]] = {}

    def _policy_content_key(self, policy: SamplingPolicy) -> str:
        return sha256_json(policy.content_dict())

    def sampling_for(self, service: ServiceSpec, policy: SamplingPolicy) -> SamplingResult:
        cache_key = (service.id, policy.id, self._policy_content_key(policy))
        if cache_key not in self._sampling_cache:
            self._sampling_cache[cache_key] = sample_records(
                self.streams[service.id].records,
                policy,
                self.policies_doc.version,
            )
        return self._sampling_cache[cache_key]

    def analyze_policy_result(
        self, service: ServiceSpec, policy: SamplingPolicy
    ) -> dict[str, Any]:
        cache_key = (service.id, policy.id, self._policy_content_key(policy))
        if cache_key not in self._policy_cache:
            sampling = self.sampling_for(service, policy)
            self._policy_cache[cache_key] = analyze_policy(
                service_id=service.id,
                records=self.streams[service.id].records,
                sampling=sampling,
                policy=policy,
                strategy_version=self.policies_doc.version,
            )
        return self._policy_cache[cache_key]

    def service_analysis(self, service: ServiceSpec) -> dict[str, Any]:
        policy_results = {
            policy.id: self.analyze_policy_result(service, policy)
            for policy in self.policies_doc.policies
        }
        stream = self.streams[service.id]
        result = {
            "service_id": service.id,
            "service_version": self.services_doc.version,
            "strategy_version": self.policies_doc.version,
            "description": service.description,
            "sample_stream_digest": stream.stream_digest,
            "stream_summary": self.stream_summaries[service.id],
            "policies": policy_results,
        }
        result["service_result_digest"] = sha256_json(result)
        return result

    def base_analyses(self) -> list[dict[str, Any]]:
        return [self.service_analysis(service) for service in self.services_doc.services]

    def scenario_analyses(
        self,
        base: list[dict[str, Any]],
        policy_id: str,
        **changes: Any,
    ) -> list[dict[str, Any]]:
        policy_map = self.policies_doc.policy_map()
        if policy_id not in policy_map:
            raise KeyError(policy_id)
        changed_policy = policy_map[policy_id].variant(**changes)
        result: list[dict[str, Any]] = []
        for service, original in zip(self.services_doc.services, base):
            replacement = self.analyze_policy_result(service, changed_policy)
            policies = dict(original["policies"])
            policies[policy_id] = replacement
            item = dict(original)
            item["policies"] = policies
            item.pop("service_result_digest", None)
            item["service_result_digest"] = sha256_json(item)
            result.append(item)
        return result

    def sensitivity_scenarios(self, base: list[dict[str, Any]]) -> list[dict[str, Any]]:
        scenarios: list[dict[str, Any]] = []
        for policy in self.policies_doc.policies:
            changes: list[tuple[str, int, int]] = []
            if policy.scale_power > 0:
                changes.append(("scale_power", policy.scale_power, policy.scale_power - 1))
            if policy.scale_power < 6:
                changes.append(("scale_power", policy.scale_power, policy.scale_power + 1))
            if policy.bucket_count > 2:
                changes.append(("bucket_count", policy.bucket_count, policy.bucket_count - 1))
            if policy.bucket_count < 256:
                changes.append(("bucket_count", policy.bucket_count, policy.bucket_count + 1))
            for parameter, old_value, new_value in changes:
                scenario_services = self.scenario_analyses(
                    base, policy.id, **{parameter: new_value}
                )
                scenarios.append(
                    {
                        "scenario_key": f"{policy.id}:{parameter}:{old_value}->{new_value}",
                        "policy_id": policy.id,
                        "parameter": parameter,
                        "old_value": old_value,
                        "new_value": new_value,
                        "service_analyses": scenario_services,
                    }
                )
        return sorted(scenarios, key=lambda item: item["scenario_key"])

    def build_bundle(self, *, include_sensitivity: bool = True) -> AnalysisBundle:
        base = self.base_analyses()
        diagnostics = summarize_analyses(base)
        scenarios = self.sensitivity_scenarios(base) if include_sensitivity else []
        flip_events = detect_parameter_flips(base, scenarios)
        diagnostics["flip_events"] = flip_events
        diagnostics["diagnostics_digest"] = diagnostics_digest(diagnostics, flip_events)
        ledger_flips = flatten_flips_for_ledger(
            diagnostics, flip_events, diagnostics["service_order"]
        )
        content = {
            "service_version": self.services_doc.version,
            "strategy_version": self.policies_doc.version,
            "services_digest": self.services_doc.content_digest,
            "policies_digest": self.policies_doc.content_digest,
            "services": base,
            "diagnostics": diagnostics,
            "ledger_flips": ledger_flips,
        }
        return AnalysisBundle(
            services_doc=self.services_doc,
            policies_doc=self.policies_doc,
            service_analyses=tuple(base),
            diagnostics=diagnostics,
            flip_events=tuple(flip_events),
            ledger_flips=tuple(ledger_flips),
            bundle_digest=sha256_json(content),
        )


def analyze_documents(
    services_doc: ServicesDoc,
    policies_doc: PoliciesDoc,
    *,
    include_sensitivity: bool = True,
) -> AnalysisBundle:
    return Analyzer(services_doc, policies_doc).build_bundle(
        include_sensitivity=include_sensitivity
    )
