"""Content-addressed fingerprints for each routing stage."""

from __future__ import annotations

from typing import Any

from .models import IntentSpec, RoutingConfig, Utterance
from .slots import slots_fingerprint
from .util import fingerprint

ALGORITHM_VERSION = "routing-workbench-v1"


def validation_fingerprint(config: RoutingConfig) -> str:
    return fingerprint(
        {
            "algorithm": ALGORITHM_VERSION,
            "vocabulary": sorted(config.vocabulary),
            "domains": {
                domain_name: sorted(intent_name for intent_name in domain.intents)
                for domain_name, domain in sorted(config.domains.items())
            },
            "slots": sorted(config.slot_declarations),
        }
    )


def domain_config_fingerprint(config: RoutingConfig) -> str:
    return fingerprint(
        {
            "algorithm": ALGORITHM_VERSION,
            "domains": {
                name: {
                    "triggers": [trigger.reference() for trigger in domain.triggers],
                }
                for name, domain in sorted(config.domains.items())
            },
            "threshold": f"{config.domain_threshold.numerator}/{config.domain_threshold.denominator}",
            "scale": config.quantization_scale,
            "decimals": config.fixed_decimals,
        }
    )


def intent_config_fingerprint(domain: str | None, config: RoutingConfig) -> str:
    if domain is None or domain not in config.domains:
        return fingerprint(
            {
                "algorithm": ALGORITHM_VERSION,
                "domain": None,
                "threshold": f"{config.intent_threshold.numerator}/{config.intent_threshold.denominator}",
                "scale": config.quantization_scale,
                "decimals": config.fixed_decimals,
            }
        )
    domain_spec = config.domains[domain]
    return fingerprint(
        {
            "algorithm": ALGORITHM_VERSION,
            "domain": domain,
            "intents": {
                name: {
                    "triggers": [trigger.reference() for trigger in intent.triggers],
                }
                for name, intent in sorted(domain_spec.intents.items())
            },
            "negation_rules": [
                {
                    "negators": list(rule.negators),
                    "scope": rule.scope,
                    "mode": rule.mode,
                    "penalty": f"{rule.penalty.numerator}/{rule.penalty.denominator}",
                    "direction": rule.direction,
                }
                for rule in config.negation_rules
            ],
            "threshold": f"{config.intent_threshold.numerator}/{config.intent_threshold.denominator}",
            "scale": config.quantization_scale,
            "decimals": config.fixed_decimals,
        }
    )


def recalibration_fingerprint(config: RoutingConfig) -> str:
    recal = config.recalibration
    return fingerprint(
        {
            "algorithm": ALGORITHM_VERSION,
            "completeness_weight": f"{recal.completeness_weight.numerator}/{recal.completeness_weight.denominator}",
            "suppression_penalty": f"{recal.suppression_penalty.numerator}/{recal.suppression_penalty.denominator}",
            "minimum": f"{recal.minimum.numerator}/{recal.minimum.denominator}",
            "maximum": f"{recal.maximum.numerator}/{recal.maximum.denominator}",
            "suppression_cap": recal.suppression_cap,
            "scale": config.quantization_scale,
            "decimals": config.fixed_decimals,
        }
    )


def utterance_fingerprint(utterance: Utterance) -> str:
    return fingerprint(utterance.canonical_content())


def stage_cache_key(
    stage: str,
    utterance: Utterance,
    config: RoutingConfig,
    *,
    selected_domain: str | None = None,
    selected_intent: str | None = None,
    slots_digest: str | None = None,
    base_confidence_int: int | None = None,
) -> str:
    payload: dict[str, Any] = {
        "algorithm": ALGORITHM_VERSION,
        "stage": stage,
        "utterance": utterance_fingerprint(utterance),
    }
    if stage == "validation":
        payload["config"] = validation_fingerprint(config)
    elif stage == "domain":
        payload["config"] = domain_config_fingerprint(config)
    elif stage == "intent":
        payload["selected_domain"] = selected_domain
        payload["config"] = intent_config_fingerprint(selected_domain, config)
    elif stage == "slots":
        payload["selected_intent"] = selected_intent
        intent = config.selected_intent_spec(selected_domain, selected_intent)
        payload["config"] = slots_fingerprint(intent, config)
    elif stage == "recalibration":
        payload["selected_domain"] = selected_domain
        payload["selected_intent"] = selected_intent
        payload["slots_digest"] = slots_digest
        payload["base_confidence_int"] = base_confidence_int
        payload["config"] = recalibration_fingerprint(config)
    else:
        raise ValueError(f"unknown stage {stage!r}")
    return fingerprint(payload)


def trigger_table_fingerprint(config: RoutingConfig) -> str:
    """Fingerprint the full trigger table, including negators, for snapshot manifests."""
    return fingerprint(
        {
            "domains": {
                name: [trigger.reference() for trigger in domain.triggers]
                for name, domain in sorted(config.domains.items())
            },
            "intents": {
                f"{domain_name}/{intent_name}": [trigger.reference() for trigger in intent.triggers]
                for domain_name, domain in sorted(config.domains.items())
                for intent_name, intent in sorted(domain.intents.items())
            },
            "negations": [
                {
                    "negators": list(rule.negators),
                    "scope": rule.scope,
                    "mode": rule.mode,
                    "penalty": f"{rule.penalty.numerator}/{rule.penalty.denominator}",
                    "direction": rule.direction,
                }
                for rule in config.negation_rules
            ],
        }
    )
