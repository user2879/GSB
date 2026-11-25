#!/usr/bin/env python3
"""Generate deterministic local samples into a data directory."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rankacc.samplegen import write_samples

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data"
    write_samples(out)
    print("samples written to", out)
