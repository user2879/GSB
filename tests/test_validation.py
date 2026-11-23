import json
from pathlib import Path

from voicecalib import load_inputs
from voicecalib.errors import ErrorCode


def test_good_samples_pass(samples_dir):
    inputs = load_inputs(samples_dir)
    assert inputs.errors.records == []
    assert inputs.valid_rounds == [0, 1, 2, 3, 4, 5]


def test_bad_riff_maps_to_audio_format(samples_dir, tmp_path):
    (tmp_path / "round_001.wav").write_bytes(b"BAD" * 100)
    # 复制其余样本，只破坏一个 WAV。
    import shutil
    for f in samples_dir.iterdir():
        if f.name != "round_001.wav":
            shutil.copy(f, tmp_path / f.name)
    inputs = load_inputs(tmp_path)
    assert ErrorCode.E_AUDIO_FORMAT in inputs.errors.codes()
    # 其余合规轮次继续处理。
    assert 1 not in inputs.valid_rounds
    assert 0 in inputs.valid_rounds


def test_missing_audio_and_nonfinite_evidence(samples_dir, tmp_path):
    import shutil
    for f in samples_dir.iterdir():
        if f.name != "round_002.wav":
            shutil.copy(f, tmp_path / f.name)
    # 证据里注入 NaN。
    ev = tmp_path / "acoustic_evidence.jsonl"
    entries = [json.loads(line) for line in ev.read_text().splitlines() if line]
    for e in entries:
        if e["round"] == 0:
            e["features"]["energy_rms"] = float("inf")
    ev.write_text("\n".join(json.dumps(e) for e in entries))
    inputs = load_inputs(tmp_path)
    codes = inputs.errors.codes()
    assert ErrorCode.E_AUDIO_MISSING in codes
    assert ErrorCode.E_NONFINITE_SAMPLE in codes
    # 违规项不影响其余合规项。
    assert 3 in inputs.valid_rounds


def test_config_enum_and_positive_tolerance(tmp_path):
    from voicecalib.config import load_config
    from voicecalib.errors import ErrorCollector

    cfg = json.loads((Path(__file__).resolve().parents[1]
                      / "samples" / "calibration_config.json").read_text())
    cfg["fixed_point"]["frac_bits"] = 10
    target = tmp_path / "cfg.json"
    target.write_text(json.dumps(cfg))
    errors = ErrorCollector()
    assert load_config(target, errors) is None
    assert ErrorCode.E_CONFIG_VALUE in errors.codes()


def test_injection_out_of_range(samples_dir):
    inputs = load_inputs(samples_dir, injection_file="injections.json")
    # 6 轮样本下，注入清单中的 round 9 越界。
    assert ErrorCode.E_ROUND_OUT_OF_RANGE in inputs.errors.codes()
