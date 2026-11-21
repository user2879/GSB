"""Fault injection (requirement 8).

An independent injection config applies real input faults of four classes:

* nonfinite_acoustic        - inject a non-finite acoustic statistic
* tool_dimension_mismatch   - corrupt a tool's dimension tag
* precision_out_of_range    - set frac_bits outside the allowed set
* weights_all_zero          - zero all three fusion weights

Each injection carries a unique id which is written into the report to
prove the hit.  Faults are applied to in-memory copies of the inputs and
then flow through the normal validation / fusion path, so the resulting
business codes are produced by the real computation, not hard-coded.
"""

from __future__ import annotations

import copy

import yaml

from . import loader

INJECTION_TYPES = (
    "nonfinite_acoustic",
    "tool_dimension_mismatch",
    "precision_out_of_range",
    "weights_all_zero",
)


def load_injections(path):
    with open(path, "r", encoding="utf-8") as handle:
        doc = yaml.safe_load(handle)
    injections = doc.get("injections", []) if isinstance(doc, dict) else []
    for inj in injections:
        if inj.get("type") not in INJECTION_TYPES:
            raise ValueError(f"unknown injection type {inj.get('type')!r}")
        if not inj.get("id"):
            raise ValueError("injection missing unique id")
    return injections


def apply_injections(turns, acoustics, tools, config, injections):
    """Apply injections to deep copies of the inputs.

    Returns (turns, acoustics, tools, config, hits, config_failures).
    `hits` records each injection id, its target and the fault applied.
    """
    turns = copy.deepcopy(turns)
    acoustics = copy.deepcopy(acoustics)
    tools = copy.deepcopy(tools)
    config = copy.deepcopy(config)
    hits = []

    for inj in injections:
        inj_id = inj["id"]
        inj_type = inj["type"]
        if inj_type == "nonfinite_acoustic":
            target = inj["target_turn"]
            field = inj.get("field", "pitch_mean")
            if target in acoustics:
                acoustics[target][field] = float("nan")
                acoustics[target]["_injection_id"] = inj_id
                hits.append({"id": inj_id, "type": inj_type, "target": target,
                             "fault": f"{field}=NaN"})
        elif inj_type == "tool_dimension_mismatch":
            target = inj["target_tool"]
            if target in tools:
                tools[target]["dimension"] = inj.get("dimension", "corrupted_dimension")
                tools[target]["_injection_id"] = inj_id
                hits.append({"id": inj_id, "type": inj_type, "target": target,
                             "fault": f"dimension={tools[target]['dimension']}"})
        elif inj_type == "precision_out_of_range":
            config["precision"]["frac_bits"] = inj.get("frac_bits", 7)
            config["precision"]["_injection_id"] = inj_id
            hits.append({"id": inj_id, "type": inj_type, "target": "config.precision",
                         "fault": f"frac_bits={config['precision']['frac_bits']}"})
        elif inj_type == "weights_all_zero":
            for channel in config["weights"]:
                config["weights"][channel] = 0.0
            config["_injection_id"] = inj_id
            hits.append({"id": inj_id, "type": inj_type, "target": "config.weights",
                         "fault": "weights=0,0,0"})

    # Config-level faults flow through the real validation layer.
    config_failures = loader.validate_config(config)
    for failure in config_failures:
        if failure["code"] == loader.codes.E_LOAD_PRECISION:
            inj_id = config.get("precision", {}).get("_injection_id")
        else:
            inj_id = config.get("_injection_id")
        if inj_id:
            failure["injection_id"] = inj_id
    return turns, acoustics, tools, config, hits, config_failures
