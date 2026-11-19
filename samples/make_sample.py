"""Generate the deterministic Windows demo sample without external dependencies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


DECLARATION_PATH = Path(__file__).with_name("sample_declaration.json")
SAMPLE_PATH = Path(__file__).with_name("windows-sample.bin")


def sample_bytes() -> bytes:
    header = b"CHUNKDEPS-WINDOWS-DETERMINISTIC-SAMPLE/v1\n"
    body = bytes((index * 37 + 11) % 256 for index in range(8192))
    footer = b"\nEND-OF-SAMPLE\n"
    return header + body + footer


def main() -> int:
    payload = sample_bytes()
    SAMPLE_PATH.write_bytes(payload)
    declaration = {
        "schema": "chunkdeps-sample-declaration/v1",
        "path": SAMPLE_PATH.name,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "construction": {
            "header": "CHUNKDEPS-WINDOWS-DETERMINISTIC-SAMPLE/v1\\n",
            "body": "8192 bytes: (index * 37 + 11) mod 256",
            "footer": "\\nEND-OF-SAMPLE\\n",
        },
        "chunk_sizes": [1024, 2048, 4096],
        "expected_chunk_counts": {
            "1024": (len(payload) + 1023) // 1024,
            "2048": (len(payload) + 2047) // 2048,
            "4096": (len(payload) + 4095) // 4096,
        },
    }
    DECLARATION_PATH.write_text(
        json.dumps(declaration, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(declaration, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

