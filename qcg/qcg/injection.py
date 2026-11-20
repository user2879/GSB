"""Independent fault injection configuration and stage-level triggers."""
import os
from dataclasses import dataclass
from typing import List, Optional

import yaml

from .errors import QCGError, E_INJECTION_FAULT, E_GRAPH_INVALID, \
    E_QUANT_PACK_INVALID
from .snapshot import STAGES

FAULT_NONFINITE_SCALE = "nonfinite_scale"
FAULT_BITWIDTH_OVERFLOW = "bitwidth_overflow"
FAULT_PACK_TRUNCATION = "pack_truncation"
FAULT_SIDECAR_TAMPER = "sidecar_tamper"
KNOWN_FAULTS = (FAULT_NONFINITE_SCALE, FAULT_BITWIDTH_OVERFLOW,
                FAULT_PACK_TRUNCATION, FAULT_SIDECAR_TAMPER)


@dataclass
class Injection:
    injection_id: str
    node_id: str
    stage: str
    fault: str

    def trigger_error(self) -> QCGError:
        messages = {
            FAULT_NONFINITE_SCALE: ("injected non-finite scale factor",
                                    E_INJECTION_FAULT, "scale"),
            FAULT_BITWIDTH_OVERFLOW: ("injected bitwidth out of enum range",
                                      E_GRAPH_INVALID, "bitwidth"),
            FAULT_PACK_TRUNCATION: ("injected quant pack truncation",
                                    E_QUANT_PACK_INVALID, "quant_pack"),
            FAULT_SIDECAR_TAMPER: ("injected sidecar metadata tamper",
                                   E_INJECTION_FAULT, "sidecar"),
        }
        message, code, field = messages[self.fault]
        return QCGError(code, message, node_id=self.node_id, field=field,
                        stage=self.stage, injection_id=self.injection_id)


def load_injections(path: Optional[str]) -> List[Injection]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    raw = doc.get("injections", [])
    out = []
    for item in raw:
        injection = Injection(injection_id=item["id"],
                              node_id=item["node_id"],
                              stage=item["stage"],
                              fault=item["fault"])
        if injection.stage not in STAGES:
            raise ValueError("unknown stage %r" % injection.stage)
        if injection.fault not in KNOWN_FAULTS:
            raise ValueError("unknown fault %r" % injection.fault)
        out.append(injection)
    return out


def matching(injections: List[Injection], node_id: str,
             stage: str) -> List[Injection]:
    return [i for i in injections
            if i.node_id == node_id and i.stage == stage]
