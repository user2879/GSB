"""Root-cause candidate derivation over the dependency DAG."""

from dataclasses import dataclass, field

from .model import Alert, Topology
from .stream import CLOSED


@dataclass
class RootCauseCandidate:
    service_id: str
    layer: int
    layer_weight: int
    active_alert_count: int
    score: int
    derivation_path: list[str]
    alert_ids: list[str]

    def to_dict(self) -> dict:
        return {
            "service_id": self.service_id,
            "layer": self.layer,
            "layer_weight": self.layer_weight,
            "active_alert_count": self.active_alert_count,
            "score": self.score,
            "derivation_path": list(self.derivation_path),
            "alert_ids": list(self.alert_ids),
        }


@dataclass
class RootCauseResult:
    candidates: list[RootCauseCandidate] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"candidates": [candidate.to_dict()
                               for candidate in self.candidates]}


def _weight_for_layer(layer: int, weights: tuple[int, ...]) -> int:
    if layer < len(weights):
        return weights[layer]
    return weights[-1]


def derive_root_causes(alerts: list[Alert], topology: Topology,
                       layer_weights: tuple[int, ...],
                       suppressed_alert_ids: list[str]) -> RootCauseResult:
    suppressed = set(suppressed_alert_ids)
    active = [alert for alert in alerts
              if alert.status != CLOSED or alert.alert_id not in suppressed]
    by_service: dict[str, list[Alert]] = {}
    for alert in active:
        by_service.setdefault(alert.service_id, []).append(alert)
    candidates: list[RootCauseCandidate] = []
    for service_id in sorted(by_service):
        service_alerts = sorted(by_service[service_id],
                                key=lambda alert: alert.alert_id)
        layer = topology.layer_of(service_id)
        weight = _weight_for_layer(layer, layer_weights)
        path = [service_id]
        path.extend(sorted(topology.reachable_from(service_id)))
        candidate = RootCauseCandidate(
            service_id=service_id,
            layer=layer,
            layer_weight=weight,
            active_alert_count=len(service_alerts),
            score=weight * len(service_alerts),
            derivation_path=path,
            alert_ids=[alert.alert_id for alert in service_alerts],
        )
        candidates.append(candidate)
    # Higher score first; stable service id tie break keeps this order
    # independent of alert arrival and batch scheduling order.
    candidates.sort(key=lambda candidate: (-candidate.score,
                                           candidate.service_id))
    return RootCauseResult(candidates=candidates)
