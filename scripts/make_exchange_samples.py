"""Prepare standalone v1 fixed-width and v2 JSON exchange samples."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from detbox.config import DetectionParams
from detbox.detect import detect_frame, load_frame_image
from detbox.exchange import write_exchange_file, v2_to_v1, v1_to_v2
from detbox.models import FrameCandidates
from detbox.samplegen import HEIGHT, WIDTH, generate_dataset
from detbox.util import atomic_write_json


def main() -> int:
    dataset = ROOT / "data" / "sample"
    if not (dataset / "ground_truth.json").exists():
        generate_dataset(dataset)
    out = ROOT / "data" / "exchange_samples"
    out.mkdir(parents=True, exist_ok=True)

    from detbox.config import DEFAULT_CLASS_TABLE

    classes = DEFAULT_CLASS_TABLE
    params = DetectionParams.load(dataset / "detection_params.json")
    frames = []
    for sequence in ("seqA", "seqB"):
        for frame_no in range(3):
            image = load_frame_image(
                dataset / "images" / sequence / f"frame_{frame_no:04d}.png"
            )
            candidates, _ = detect_frame(image, params, sequence, frame_no)
            frames.append(
                FrameCandidates(
                    sequence=sequence,
                    frame=frame_no,
                    width=WIDTH,
                    height=HEIGHT,
                    candidates=candidates,
                    source_format="v2",
                )
            )

    v1_frames, sidecar = v2_to_v1(frames, classes)
    write_exchange_file(out / "candidates_v1.txt", "v1", v1_frames, classes)
    write_exchange_file(out / "candidates_v2.json", "v2", frames, classes)
    atomic_write_json(out / "v2_to_v1_sidecar.json", sidecar)

    upgraded = v1_to_v2(v1_frames)
    write_exchange_file(out / "candidates_v1_upgraded_v2.json", "v2", upgraded, classes)

    line_count = sum(1 for _ in (out / "candidates_v1.txt").open())
    print(f"v1 lines: {line_count}")
    print(f"sidecar records: {len(sidecar['records'])}")
    print(f"outputs in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
