from voicecalib import load_inputs, run_pipeline, build_report
from voicecalib.evidence import EMOTIONS
from voicecalib.hashing import fingerprint


def _run(samples_dir, output_dir, workers, budget=None):
    inputs = load_inputs(samples_dir)
    inputs.config.max_workers = workers
    if budget is not None:
        inputs.config.batch_segment_budget = budget
    result = run_pipeline(inputs, output_dir, resume=False)
    return inputs, result


def test_emotion_and_link_weights_recorded(full_samples_dir, output_dir):
    inputs, result = _run(full_samples_dir, output_dir, 1)
    rows = [result.calibrations[r] for r in sorted(result.calibrations)]
    assert len(rows) == 12
    for row in rows:
        assert row.link in ("init", "same_speaker", "speaker_switch")
        assert len(row.calib_q) == len(EMOTIONS)
        assert row.emotion in EMOTIONS
        assert all(isinstance(v, int) for v in row.calib_q)
    # 切换轮与同说话人轮的衔接权重不同且显式记录。
    switch = next(r for r in rows if r.link == "speaker_switch")
    same = next(r for r in rows if r.link == "same_speaker")
    assert switch.link_weight_q == inputs.config.transition_weight_q
    assert same.link_weight_q == inputs.config.same_speaker_weight_q
    assert switch.link_weight_q != same.link_weight_q


def test_single_vs_multi_thread_field_identical(samples_dir, tmp_path):
    _, single = _run(samples_dir, tmp_path / "single", 1)
    _, multi = _run(samples_dir, tmp_path / "multi", 4)
    _, multi_again = _run(samples_dir, tmp_path / "multi2", 4)
    for getter in (
        lambda r: [r.evidence[k].to_dict() for k in sorted(r.evidence)],
        lambda r: [r.calibrations[k].to_dict() for k in sorted(r.calibrations)],
        lambda r: [r.attributions[k].to_dict() for k in sorted(r.attributions)],
        lambda r: [t.to_dict() for t in r.trajectories],
        lambda r: r.errors.sorted_dict_list(),
    ):
        assert fingerprint(getter(single)) == fingerprint(getter(multi))
        assert fingerprint(getter(multi)) == fingerprint(getter(multi_again))


def test_float64_reference_and_quant_metadata(samples_dir, output_dir):
    _, result = _run(samples_dir, output_dir, 1)
    artifact = result.evidence[0]
    assert artifact.fp_metadata["rounding_mode"] == "ROUND_HALF_EVEN"
    assert artifact.fp_metadata["frac_bits"] == 12
    assert len(artifact.quant_residuals) == len(EMOTIONS)
    for q, f64, residual in zip(artifact.strength_q, artifact.softmax_float,
                                artifact.quant_residuals):
        assert isinstance(q, int)
        assert abs(f64 - q / artifact.fp_metadata["scale"] - residual) < 1e-12


def test_speaker_trace_backtrace(samples_dir, output_dir):
    _, result = _run(samples_dir, output_dir, 1)
    for rnd, attr in result.attributions.items():
        assert attr.calib_round == rnd
        assert attr.evidence_component == "features.f0_mean_hz"
        assert attr.segment_id is not None
    speakers = {rnd: attr.speaker for rnd, attr in result.attributions.items()}
    # 合成脚本 S1/S2 交替，归属应能区分。
    assert len(set(speakers.values())) == 2
