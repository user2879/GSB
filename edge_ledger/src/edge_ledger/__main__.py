from __future__ import annotations

import sys


def main() -> int:
    print("edge-ledger package. Available commands:")
    print("  edge-ledger-samples --out samples")
    print("  edge-ledger-demo --out artifacts/acceptance")
    print("  edge-ledger-verify --spec-dir SPEC --ledger DB")
    print("  edge-ledger-ui --spec-dir SPEC --ledger DB")
    print("  python -m pytest -q")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
