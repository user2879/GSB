"""Independent fault injection harness with per-audio isolation."""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Sequence

from .audioio import read_wav
from .config import DiarConfig, parse_config
from .errors import AudioFormatError, ConfigInvalidError
from .pipeline import process_audio, write_fault_injection_cases


def _decode_with_nonfinite(path: str) -> None:
    wave = read_wav(path)
    wave.samples[len(wave.samples) // 2] = math.nan
    # Reuse the same validation branch used by the reader.
    for value in wave.samples:
        if not math.isfinite(value):
            raise AudioFormatError("non-finite sample value encountered",
                                   {"injection": "INJ-01"})


def run_fault_injections(sample_path: str, base_samples: Sequence[float],
                         cfg: DiarConfig, work_root: str) -> List[Dict[str, Any]]:
    fault_dir = os.path.join(work_root, "faults")
    cases = write_fault_injection_cases(fault_dir, base_samples, sample_path)
    report: List[Dict[str, Any]] = []

    for case in cases:
        entry = {"id": case["id"], "type": case["type"], "hit": False,
                 "error_code": "", "message": ""}
        try:
            if case["type"] == "non_finite_sample":
                _decode_with_nonfinite(case["path"])
            elif case["type"] == "all_silence":
                # Silence is a legal format; pipeline must still complete.
                result = process_audio(case["path"], cfg, work_root)
                entry["hit"] = True
                entry["detail"] = {
                    "window_count": result["window_count"],
                    "segment_count": len(result["segmentation"].segments),
                }
            elif case["type"] == "threshold_out_of_range":
                bad = cfg.to_dict()
                bad["threshold"] = case["threshold"]
                parse_config(bad)
            elif case["type"] == "truncated_data":
                read_wav(case["path"])
        except (AudioFormatError, ConfigInvalidError) as exc:
            entry["hit"] = True
            entry["error_code"] = exc.code
            entry["message"] = exc.message
        except Exception as exc:  # isolation: record, never abort the batch
            entry["hit"] = True
            entry["error_code"] = "E_INTERNAL"
            entry["message"] = str(exc)
        report.append(entry)
    return report
