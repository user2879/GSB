from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mdr.utils import write_json


def main() -> int:
    report: dict[str, object] = {
        "linux": sys.platform.startswith("linux"),
        "python": sys.version,
        "executable": sys.executable,
    }
    try:
        import numpy

        report["numpy"] = numpy.__version__
    except Exception as exc:
        report["numpy_error"] = repr(exc)
    try:
        import PIL

        report["pillow"] = PIL.__version__
    except Exception as exc:
        report["pillow_error"] = repr(exc)
    try:
        from mdr.tk_support import import_tkinter

        tk = import_tkinter()
        report["tkinter"] = f"available (Tk {tk.TkVersion})"
    except Exception as exc:
        report["tkinter"] = f"unavailable: {type(exc).__name__}: {exc}"
        report["system_install_hint"] = "sudo apt-get install -y python3-tk"
        report["user_local_hint"] = "bash scripts/install_local_tk.sh"

    report["passed"] = all(key in report for key in ("numpy", "pillow", "tkinter"))
    write_json(ROOT / "reports" / "environment-report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
