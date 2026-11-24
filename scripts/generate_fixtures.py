"""Generate all deterministic sample fixtures under samples/."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fcdiag.generator import (  # noqa: E402
    CONFIG_DOC,
    build_baseline_receipts,
    write_fixtures,
)
from fcdiag.loader import load_config, load_pipeline, load_samples  # noqa: E402
from fcdiag.receipts import (  # noqa: E402
    downgrade_v2_to_v1,
    downgrade_v3_to_v2,
    render_receipts,
)
from fcdiag.state import WorkState  # noqa: E402


def main() -> None:
    sample_dir = ROOT / "samples"
    write_fixtures(str(sample_dir))
    samples = load_samples((sample_dir / "samples.jsonl").read_text(encoding="utf-8"))
    pipeline = load_pipeline((sample_dir / "pipeline.yaml").read_text(encoding="utf-8"))
    config = load_config(CONFIG_DOC)
    state = WorkState.initial(samples, pipeline, config)
    receipts_v3 = build_baseline_receipts(state)
    receipt_dir = sample_dir / "receipts"
    receipt_dir.mkdir(exist_ok=True)
    v3_text = render_receipts(3, receipts_v3)
    v2 = downgrade_v3_to_v2(receipts_v3)
    v1 = downgrade_v2_to_v1(v2.records)
    (receipt_dir / "v3.txt").write_text(v3_text, encoding="utf-8")
    (receipt_dir / "v2.txt").write_text(v2.text, encoding="utf-8")
    (receipt_dir / "v1.txt").write_text(v1.text, encoding="utf-8")
    (receipt_dir / "conversion-sidecar.json").write_text(
        json.dumps(
            {
                "v3_to_v2": v2.sidecar,
                "v2_to_v1": v1.sidecar,
                "v3_to_v2_diffs": list(v2.diffs),
                "v2_to_v1_diffs": list(v1.diffs),
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"generated fixtures in {sample_dir}")


if __name__ == "__main__":
    main()
