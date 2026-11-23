"""Convenience entry point: regenerate samples and run the full demo."""

from pathlib import Path

from diagconsole.demo import run_demo
from scripts.generate_samples import main as generate

if __name__ == "__main__":
    generate(["--out", "samples"])
    report = run_demo(Path("samples"), Path("data"), local=False, crash=True)
    import json
    print(json.dumps(report["acceptance"], indent=2, sort_keys=True))
