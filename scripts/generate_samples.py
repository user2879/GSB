"""Generate deterministic local samples (config / JSONL / edits / injections / policies)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rope_diag.samples import generate_all

if __name__ == "__main__":
    out = generate_all(Path(__file__).resolve().parent.parent / "samples")
    print(f"generated {len(out['samples'])} samples into samples/")
