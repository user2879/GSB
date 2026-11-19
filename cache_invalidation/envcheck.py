"""本地环境检查：Python 版本、标准库模块与回环端口可用性。"""

from __future__ import annotations

import importlib
import json
import platform
import sys

from .netloop import find_free_loopback_port


REQUIRED_MODULES = ("sqlite3", "json", "threading", "subprocess", "tkinter")


def check_environment() -> dict:
    result: dict = {
        "python_version": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "modules": {},
        "loopback": {},
    }
    for module_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
            result["modules"][module_name] = "available"
        except ImportError as exc:
            result["modules"][module_name] = f"missing: {exc}"
    port = find_free_loopback_port()
    result["loopback"] = {
        "host": "127.0.0.1",
        "port": port,
        "bindable": True,
    }
    result["python_minor_ok"] = sys.version_info >= (3, 11)
    return result


def main() -> int:
    result = check_environment()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    tk_state = result["modules"]["tkinter"]
    if not result["python_minor_ok"] or not result["loopback"]["bindable"]:
        return 1
    if tk_state != "available":
        return 2
    return 0
