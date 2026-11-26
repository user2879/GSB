from __future__ import annotations

from pathlib import Path
from typing import Any

from .codes import E_INPUT_INVALID, OK
from .config import DEFAULT_CONFIG, merge_defaults, validate_config
from .samplegen import default_cameras, default_landmarks, render_view
from .utils import read_json, write_json
from .validation import validate_camera, validate_png


def initialize_workspace(root: str | Path, overwrite_config: bool = False) -> Path:
    root_path = Path(root)
    for directory in ("configs", "samples/views", "ledger/artifacts", "ledger/snapshots", "reports", "tests", "scripts"):
        (root_path / directory).mkdir(parents=True, exist_ok=True)
    config_path = root_path / "configs" / "reconstruction.json"
    if overwrite_config or not config_path.exists():
        write_json(config_path, DEFAULT_CONFIG)
    return config_path


def _views_by_id(dataset: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {view["view_id"]: view for view in dataset.get("views", []) if isinstance(view, dict) and "view_id" in view}


def apply_commands(
    root: str | Path,
    commands: list[dict[str, Any]],
) -> dict[str, Any]:
    root_path = Path(root)
    config_path = root_path / "configs" / "reconstruction.json"
    dataset_path = root_path / "samples" / "dataset.json"
    config = merge_defaults(read_json(config_path) if config_path.exists() else DEFAULT_CONFIG)
    dataset = read_json(dataset_path)
    camera_path = root_path / "samples" / "cameras.json"
    camera_file = dataset.get("camera_file", "cameras.json")
    camera_path = (root_path / "samples" / str(camera_file)).resolve()
    cameras_doc = read_json(camera_path) if camera_path.exists() else {"cameras": {}}
    cameras = cameras_doc.get("cameras", cameras_doc)

    truth_path = root_path / "samples" / "scene_truth.json"
    landmarks = read_json(truth_path).get("landmarks", default_landmarks()) if truth_path.exists() else default_landmarks()

    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    def record(index: int, op: str, code: str, detail: str, ref: str = "") -> None:
        item = {"index": index, "op": op, "code": code, "ref": ref, "detail": detail}
        results.append(item)
        if code == E_INPUT_INVALID:
            errors.append({"code": E_INPUT_INVALID, "ref": ref or f"command[{index}]", "reason": detail})

    for index, command in enumerate(commands):
        if not isinstance(command, dict):
            record(index, "invalid", E_INPUT_INVALID, "command must be an object")
            continue
        op = str(command.get("op", ""))
        view_id = command.get("view_id")
        views = _views_by_id(dataset)
        try:
            if op == "replace_view":
                if not isinstance(view_id, str) or view_id not in views:
                    record(index, op, E_INPUT_INVALID, "unknown view reference", str(view_id))
                    continue
                view = views[view_id]
                if command.get("image"):
                    image_ref = str(command["image"])
                    image_path = (root_path / "samples" / image_ref).resolve()
                    info = validate_png(image_path, int(config["width"]), int(config["height"]), str(config["mode"]))
                    if not info["ok"]:
                        err = info["error"]
                        record(index, op, E_INPUT_INVALID, err["reason"], err["ref"])
                        continue
                    view["image"] = image_ref
                else:
                    variant = int(command.get("variant", view.get("replacement_variant", 0) + 1))
                    image_rel = view.get("image", f"views/{view_id}.png")
                    image_path = root_path / "samples" / str(image_rel)
                    camera = cameras[view_id]
                    sequence = int(view.get("sequence", 0))
                    render_view(image_path, view_id, sequence, camera, landmarks, config, int(config["seed"]), variant)
                    view["replacement_variant"] = variant
                record(index, op, OK, "view replaced", str(view_id))

            elif op == "add_view":
                if not isinstance(view_id, str) or not view_id:
                    record(index, op, E_INPUT_INVALID, "add_view requires view_id", str(view_id))
                    continue
                if view_id in views:
                    record(index, op, E_INPUT_INVALID, "view already exists", view_id)
                    continue
                sequence = int(command.get("sequence", max([int(v.get("sequence", -1)) for v in dataset["views"]]) + 1))
                if any(int(v.get("sequence")) == sequence for v in dataset["views"]):
                    record(index, op, E_INPUT_INVALID, f"sequence {sequence} already exists", view_id)
                    continue
                base_cameras = default_cameras(config)
                camera = command.get("camera") or base_cameras.get(f"v{sequence}") or {
                    "K": base_cameras["v0"]["K"],
                    "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "t": [-6 * sequence, 0, 0],
                    "center": [6 * sequence, 0, 0],
                }
                camera_errors = validate_camera(camera, f"camera:{view_id}")
                if camera_errors:
                    errors.extend(camera_errors)
                    record(index, op, E_INPUT_INVALID, "invalid camera for new view", view_id)
                    continue
                image_rel = f"views/{view_id}.png"
                render_view(root_path / "samples" / image_rel, view_id, sequence, camera, landmarks, config, int(config["seed"]), int(command.get("variant", 0)))
                dataset["views"].append({"view_id": view_id, "sequence": sequence, "image": image_rel})
                cameras[view_id] = camera
                for other in command.get("pair_with", []):
                    if other in _views_by_id(dataset) and other != view_id:
                        pair = [other, view_id]
                        if pair not in dataset.setdefault("pairs", []):
                            dataset["pairs"].append(pair)
                record(index, op, OK, "view added", view_id)

            elif op == "remove_view":
                if not isinstance(view_id, str) or view_id not in views:
                    record(index, op, E_INPUT_INVALID, "unknown view reference", str(view_id))
                    continue
                dataset["views"] = [v for v in dataset["views"] if v.get("view_id") != view_id]
                dataset["pairs"] = [
                    pair for pair in dataset.get("pairs", [])
                    if not (isinstance(pair, list) and view_id in pair)
                ]
                record(index, op, OK, "view removed", view_id)

            elif op == "set_extrinsic":
                if not isinstance(view_id, str) or view_id not in views:
                    record(index, op, E_INPUT_INVALID, "unknown view reference", str(view_id))
                    continue
                candidate = dict(cameras[view_id])
                if isinstance(command.get("t"), list):
                    candidate["t"] = command["t"]
                if isinstance(command.get("R"), list):
                    candidate["R"] = command["R"]
                if isinstance(command.get("center"), list):
                    center = command["center"]
                    candidate["center"] = center
                    r = candidate.get("R", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                    candidate["t"] = [
                        -sum(r[i][j] * center[j] for j in range(3)) for i in range(3)
                    ] if False else [
                        -(r[0][0] * center[0] + r[0][1] * center[1] + r[0][2] * center[2]),
                        -(r[1][0] * center[0] + r[1][1] * center[1] + r[1][2] * center[2]),
                        -(r[2][0] * center[0] + r[2][1] * center[1] + r[2][2] * center[2]),
                    ]
                camera_errors = validate_camera(candidate, f"camera:{view_id}")
                if camera_errors:
                    for err in camera_errors:
                        record(index, op, E_INPUT_INVALID, err["reason"], err["ref"])
                    continue
                cameras[view_id] = candidate
                record(index, op, OK, "extrinsic updated", view_id)

            elif op == "set_param":
                key = str(command.get("key", ""))
                candidate = merge_defaults(config)
                if key in {"global_point_budget", "view_point_quota", "max_subdivisions_per_batch", "max_workers"}:
                    candidate["budgets"][key] = command.get("value")
                elif key in candidate:
                    candidate[key] = command.get("value")
                else:
                    record(index, op, E_INPUT_INVALID, f"unknown parameter {key}", key)
                    continue
                param_errors = validate_config(candidate)
                if param_errors:
                    for err in param_errors:
                        record(index, op, E_INPUT_INVALID, err["reason"], err["ref"])
                    continue
                config = candidate
                record(index, op, OK, f"parameter {key} updated", key)

            else:
                record(index, op or "unknown", E_INPUT_INVALID, "unknown command operation")
        except Exception as exc:
            record(index, op, E_INPUT_INVALID, f"command failed: {type(exc).__name__}: {exc}", str(view_id))

    write_json(config_path, config)
    write_json(dataset_path, dataset)
    camera_doc = cameras_doc if "cameras" in cameras_doc else {}
    if "cameras" in cameras_doc:
        camera_doc["cameras"] = cameras
    else:
        camera_doc = cameras
    write_json(camera_path, camera_doc)
    command_result = {
        "results": results,
        "errors": errors,
        "business_code_counts": {OK: sum(1 for r in results if r["code"] == OK), E_INPUT_INVALID: len(errors)},
    }
    write_json(root_path / "ledger" / "command_result.json", command_result)
    return command_result


def load_command_stream(path: str | Path) -> list[dict[str, Any]]:
    data = read_json(path)
    if isinstance(data, dict) and isinstance(data.get("commands"), list):
        return data["commands"]
    if isinstance(data, list):
        return data
    raise ValueError("command stream must be a list or an object with commands list")
