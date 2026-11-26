"""Edit command stream application and impact manifests."""
from __future__ import annotations

import copy
from typing import Any

from .codes import E_SCHEMA_UNKNOWN, LabValidationError
from .hashing import fingerprint
from .validation import is_int, make_error, validate_sample, validate_scheme


KNOWN_OPS = {
    "add_scheme",
    "remove_scheme",
    "update_scheme",
    "set_train_length",
    "set_target_length",
    "replace_sample",
    "delete_sample",
    "set_fixed_point",
    "set_scheduling",
    "set_concurrency",
}


def validate_command_stream(commands: list[dict[str, Any]], config: dict[str, Any], samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    scheme_names = {scheme.get("name") for scheme in config.get("schemes", []) if isinstance(scheme, dict)}
    sample_ids = {sample.get("sample_id") for sample in samples}
    for index, command in enumerate(commands):
        where = f"commands[{index}]"
        command_id = command.get("command_id") if isinstance(command, dict) else None
        if not isinstance(command, dict):
            errors.append(make_error(E_SCHEMA_UNKNOWN, f"{where} must be an object"))
            continue
        op = command.get("op")
        if op not in KNOWN_OPS:
            errors.append(make_error(E_SCHEMA_UNKNOWN, f"{where} has unknown op {op!r}", command_id=command_id))
            continue

        if op == "add_scheme":
            scheme = command.get("scheme")
            errors.extend(validate_scheme(scheme, int(config.get("dimension", -1)), index))
            if isinstance(scheme, dict) and scheme.get("name") in scheme_names:
                errors.append(make_error(E_SCHEMA_UNKNOWN, "add_scheme target already exists", command_id=command_id, scheme=scheme.get("name")))
            elif isinstance(scheme, dict):
                scheme_names.add(scheme.get("name"))
        elif op in ("update_scheme", "remove_scheme"):
            name = command.get("name")
            if name not in scheme_names:
                errors.append(make_error(E_SCHEMA_UNKNOWN, f"{op} references undeclared scheme", command_id=command_id, scheme=name))
            elif op == "remove_scheme":
                scheme_names.discard(name)
        elif op in ("set_train_length", "set_target_length"):
            key = "train_length" if op == "set_train_length" else "target_length"
            if not is_int(command.get(key)) or command[key] < 0:
                errors.append(make_error(E_SCHEMA_UNKNOWN, f"{key} must be nonnegative integer", command_id=command_id, value=command.get(key)))
        elif op == "replace_sample":
            sample = command.get("sample")
            errors.extend(validate_sample(sample, int(config["dimension"]), scheme_names))
            if isinstance(sample, dict) and sample.get("sample_id"):
                sample_ids.add(sample["sample_id"])
        elif op == "delete_sample":
            sample_id = command.get("sample_id")
            if sample_id not in sample_ids:
                errors.append(make_error(E_SCHEMA_UNKNOWN, "delete_sample references unknown sample", command_id=command_id, sample_id=sample_id))
            else:
                sample_ids.discard(sample_id)
        elif op == "set_fixed_point":
            if not isinstance(command.get("fixed_point"), dict):
                errors.append(make_error(E_SCHEMA_UNKNOWN, "set_fixed_point requires fixed_point object", command_id=command_id))
        elif op == "set_scheduling":
            for key in ("point_budget", "sample_quota", "concurrency"):
                if key in command and (not is_int(command[key]) or command[key] <= 0):
                    errors.append(make_error(E_SCHEMA_UNKNOWN, f"{key} must be positive integer", command_id=command_id, value=command[key]))
        elif op == "set_concurrency":
            if not is_int(command.get("concurrency")) or command["concurrency"] <= 0:
                errors.append(make_error(E_SCHEMA_UNKNOWN, "concurrency must be positive integer", command_id=command_id))
    return errors


def dimension_dependency_fingerprint(sample: dict[str, Any], dimension: int) -> str:
    partner = dimension ^ 1
    dimensions = sorted({int(dimension), int(partner)})
    material = {
        "query_position": sample.get("query_position"),
        "key_positions": sample.get("key_positions"),
        "dimensions": dimensions,
        "query_pair": [sample.get("query_vector", [None] * (dimension + 1))[d] for d in dimensions],
        "key_pairs": [
            [vector[d] for d in dimensions]
            for vector in sample.get("key_vectors", [])
        ],
    }
    return fingerprint(material)


def sample_geometry_fingerprint(sample: dict[str, Any]) -> str:
    material = {
        "sample_id": sample.get("sample_id"),
        "query_position": sample.get("query_position"),
        "key_positions": sample.get("key_positions"),
        "query_vector": sample.get("query_vector"),
        "key_vectors": sample.get("key_vectors"),
    }
    return fingerprint(material)


def _empty_manifest(command: dict[str, Any]) -> dict[str, Any]:
    return {
        "command_id": command.get("command_id"),
        "op": command.get("op"),
        "added_schemes": [],
        "removed_schemes": [],
        "changed_schemes": {},
        "changed_samples": {},
        "replaced_samples": [],
        "deleted_samples": [],
        "fixed_point_changed": False,
        "classification_changed": False,
        "scheduling_changed": False,
        "geometry_changed": False,
    }


def apply_command(
    config: dict[str, Any],
    samples: list[dict[str, Any]],
    command: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    next_config = copy.deepcopy(config)
    next_samples = copy.deepcopy(samples)
    manifest = _empty_manifest(command)
    op = command.get("op")

    def scheme_index(name: Any) -> int:
        for index, scheme in enumerate(next_config.get("schemes", [])):
            if scheme.get("name") == name:
                return index
        raise LabValidationError(E_SCHEMA_UNKNOWN, f"unknown scheme name: {name}", {"command_id": command.get("command_id"), "scheme": name})

    if op == "add_scheme":
        scheme = copy.deepcopy(command["scheme"])
        if any(s.get("name") == scheme.get("name") for s in next_config["schemes"]):
            raise LabValidationError(E_SCHEMA_UNKNOWN, "scheme already exists", {"scheme": scheme.get("name")})
        next_config["schemes"].append(scheme)
        manifest["added_schemes"].append(scheme["name"])

    elif op == "remove_scheme":
        name = command["name"]
        index = scheme_index(name)
        next_config["schemes"].pop(index)
        manifest["removed_schemes"].append(name)

    elif op == "update_scheme":
        name = command["name"]
        index = scheme_index(name)
        params = command.get("params", {})
        changed_fields: list[str] = []
        for key, value in params.items():
            if next_config["schemes"][index].get(key) != value:
                next_config["schemes"][index][key] = value
                changed_fields.append(key)
        manifest["changed_schemes"][name] = changed_fields

    elif op == "set_train_length":
        value = int(command["train_length"])
        next_config["train_length"] = value
        for sample in next_samples:
            if sample.get("train_length") != value:
                sample["train_length"] = value
                manifest["classification_changed"] = True
        manifest["classification_changed"] = True

    elif op == "set_target_length":
        next_config["target_length"] = int(command["target_length"])
        manifest["classification_changed"] = True

    elif op == "replace_sample":
        replacement = copy.deepcopy(command["sample"])
        sample_id = replacement["sample_id"]
        found = False
        for index, sample in enumerate(next_samples):
            if sample.get("sample_id") == sample_id:
                old_dependency_fps = {
                    d: dimension_dependency_fingerprint(sample, d)
                    for d in range(int(next_config["dimension"]))
                }
                next_samples[index] = replacement
                changed_dimensions = [
                    d
                    for d in range(int(next_config["dimension"]))
                    if dimension_dependency_fingerprint(replacement, d) != old_dependency_fps[d]
                ]
                manifest["changed_samples"][sample_id] = {
                    "changed_dimensions": changed_dimensions,
                    "geometry_fingerprint_changed": sample_geometry_fingerprint(sample) != sample_geometry_fingerprint(replacement),
                }
                manifest["replaced_samples"].append(sample_id)
                manifest["geometry_changed"] = True
                found = True
                break
        if not found:
            next_samples.append(replacement)
            manifest["changed_samples"][sample_id] = {
                "changed_dimensions": list(range(int(next_config["dimension"]))),
                "geometry_fingerprint_changed": True,
                "inserted": True,
            }
            manifest["replaced_samples"].append(sample_id)
            manifest["geometry_changed"] = True

    elif op == "delete_sample":
        sample_id = command["sample_id"]
        before = len(next_samples)
        next_samples = [sample for sample in next_samples if sample.get("sample_id") != sample_id]
        if len(next_samples) == before:
            raise LabValidationError(E_SCHEMA_UNKNOWN, f"unknown sample id: {sample_id}", {"sample_id": sample_id})
        manifest["deleted_samples"].append(sample_id)

    elif op == "set_fixed_point":
        next_config["fixed_point"].update(copy.deepcopy(command.get("fixed_point", {})))
        manifest["fixed_point_changed"] = True

    elif op == "set_scheduling":
        for key in ("point_budget", "sample_quota", "concurrency"):
            if key in command:
                next_config[key] = int(command[key])
        manifest["scheduling_changed"] = True

    elif op == "set_concurrency":
        next_config["concurrency"] = int(command["concurrency"])
        manifest["scheduling_changed"] = True

    else:
        raise LabValidationError(E_SCHEMA_UNKNOWN, f"unknown command op: {op}", {"command_id": command.get("command_id")})

    return next_config, next_samples, manifest
