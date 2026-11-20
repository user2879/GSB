"""增量编辑命令流：对工作副本中的视图与配置应用确定性编辑。"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field

from .config import ReconstructionConfig
from .errors import E_VIEW_INVALID, StageError
from .util import read_json, write_json_atomic

SUPPORTED_COMMANDS = {
    "replace_image",
    "replace_camera",
    "add_view",
    "remove_view",
    "update_config",
}


@dataclass
class EditResult:
    applied: list[dict] = field(default_factory=list)


def _view_paths(work_dir: str, view_id: int) -> tuple[str, str]:
    views_dir = os.path.join(work_dir, "views")
    return (
        os.path.join(views_dir, f"view_{view_id:04d}.png"),
        os.path.join(views_dir, f"view_{view_id:04d}.cam.json"),
    )




def _resolve_source(work_input_dir: str, source_input_dir: str, reference: str) -> str:
    base = os.path.basename(reference)
    candidates = [
        os.path.join(source_input_dir, base),
        reference,
        os.path.join(work_input_dir, base),
        os.path.join(source_input_dir, reference),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"编辑引用的源文件不存在: {reference}")


def apply_edit_stream(
    source_input_dir: str,
    work_input_dir: str,
    edit_stream_path: str | None,
) -> EditResult:
    """把源样本目录复制为工作副本后，顺序应用编辑命令，返回应用清单。"""
    work_exists = os.path.isdir(os.path.join(work_input_dir, "views"))
    if os.path.abspath(source_input_dir) != os.path.abspath(work_input_dir) and not work_exists:
        if os.path.exists(work_input_dir):
            shutil.rmtree(work_input_dir)
        shutil.copytree(source_input_dir, work_input_dir)

    result = EditResult()
    if not edit_stream_path:
        return result
    stream = read_json(edit_stream_path)
    commands = stream.get("commands", [])
    for index, command in enumerate(commands):
        cmd_type = command.get("command")
        if cmd_type not in SUPPORTED_COMMANDS:
            raise StageError(
                E_VIEW_INVALID, f"第 {index} 条编辑命令类型非法: {cmd_type}", stage="editor"
            )
        if cmd_type == "replace_image":
            view_id = int(command["view_id"])
            image_path, _ = _view_paths(work_input_dir, view_id)
            source = _resolve_source(work_input_dir, source_input_dir, command["source_image"])
            shutil.copyfile(source, image_path)
        elif cmd_type == "replace_camera":
            view_id = int(command["view_id"])
            _, camera_path = _view_paths(work_input_dir, view_id)
            source = _resolve_source(work_input_dir, source_input_dir, command["source_camera"])
            shutil.copyfile(source, camera_path)
        elif cmd_type == "add_view":
            view_id = int(command["view_id"])
            image_path, camera_path = _view_paths(work_input_dir, view_id)
            source_image = _resolve_source(
                work_input_dir, source_input_dir, command["source_image"]
            )
            source_camera = _resolve_source(
                work_input_dir, source_input_dir, command["source_camera"]
            )
            shutil.copyfile(source_image, image_path)
            shutil.copyfile(source_camera, camera_path)
        elif cmd_type == "remove_view":
            view_id = int(command["view_id"])
            for path in _view_paths(work_input_dir, view_id):
                if os.path.exists(path):
                    os.remove(path)
        elif cmd_type == "update_config":
            config_path = os.path.join(work_input_dir, "reconstruction_config.json")
            raw = read_json(config_path)
            for key, value in command["parameters"].items():
                raw[key] = value
            write_json_atomic(config_path, raw)
        result.applied.append({"seq": index, "command": command})
    return result


def load_config(input_dir: str) -> ReconstructionConfig:
    config_path = os.path.join(input_dir, "reconstruction_config.json")
    if not os.path.exists(config_path):
        raise StageError(E_VIEW_INVALID, f"重建配置缺失: {config_path}", stage="config")
    return ReconstructionConfig.from_dict(read_json(config_path))
