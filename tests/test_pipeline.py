from dataclasses import replace

import numpy as np

from voicecmd.config import load_config
from voicecmd.errors import E_WINDOW_BUDGET_PENDING
from voicecmd.injection import InjectionConfig
from voicecmd.pipeline import Pipeline


def _full_run(cfg, workspace, *, workers=None, budget=1000):
    config = load_config(workspace / "config.json")
    if workers is not None:
        config.runtime.max_workers = workers
    pipeline = Pipeline(config, str(workspace), snapshot=False)
    pipeline.load_all()
    result = pipeline.run_batch(budget=budget)
    return pipeline, result


def _serial_signature(pipeline):
    return {
        index: (
            pipeline.features[index].fixed_point,
            pipeline.posteriors[index].scores,
            pipeline.posteriors[index].posteriors,
            pipeline.posteriors[index].boundary,
        )
        for index in sorted(pipeline.features)
    }


def test_full_run_produces_windows_and_conclusions(workspace, cfg):
    pipeline, result = _full_run(cfg, workspace)
    assert len(pipeline.windows) > 0
    assert result.pending == []
    assert pipeline.conclusions
    assert all(c.window_indices for c in pipeline.conclusions)


def test_fragment_replacement_precise_invalidation(workspace, cfg):
    pipeline, _ = _full_run(cfg, workspace)
    before = {
        index: (pipeline.features[index].cache_key,
                pipeline.posteriors[index].cache_key)
        for index in pipeline.windows
    }
    fragment = pipeline.fragments[2]
    changed = replace(
        fragment,
        samples=np.clip(fragment.samples.astype(np.int32) + 11, -32768, 32767).astype(np.int16),
    )
    affected = pipeline.replace_fragment(changed)
    expected = sorted(
        index for index, window in pipeline.windows.items()
        if 2 in window.fragment_seqs
    )
    assert affected == expected
    result = pipeline.run_batch(budget=1000)
    assert sorted(result.recomputed) == expected
    assert sorted(result.reused) == sorted(set(pipeline.windows) - set(expected))
    assert all(
        before[index] == (pipeline.features[index].cache_key,
                          pipeline.posteriors[index].cache_key)
        for index in pipeline.windows if index not in expected
    )


def test_budget_pending_priority_and_no_false_completion(workspace, cfg):
    config = load_config(workspace / "config.json")
    config.runtime.default_budget = 5
    pipeline = Pipeline(config, str(workspace), snapshot=False)
    pipeline.load_all()
    first = pipeline.run_batch(budget=5)
    assert len(first.recomputed) == 5
    assert len(first.pending) == len(pipeline.windows) - 5
    assert all(record["code"] == E_WINDOW_BUDGET_PENDING
               for record in first.budget_pending_records)
    pending_head = first.pending[0]
    second = pipeline.run_batch(budget=5)
    assert second.recomputed[0] == pending_head
    assert set(second.reused) == set(first.recomputed)
    while pipeline.pending or pipeline.invalid_windows:
        pipeline.run_batch(budget=5)
    assert len(pipeline.features) == len(pipeline.windows)


def test_concurrency_matches_serial_signature(workspace, cfg):
    serial, _ = _full_run(cfg, workspace, workers=1)
    serial_signature = _serial_signature(serial)
    for workers in (2, 4, 8):
        parallel, _ = _full_run(cfg, workspace, workers=workers)
        assert _serial_signature(parallel) == serial_signature
        assert [c.cache_key for c in parallel.conclusions] == \
               [c.cache_key for c in serial.conclusions]


def test_rebuild_from_blank_matches_incremental(workspace, cfg):
    full, _ = _full_run(cfg, workspace)
    config = load_config(workspace / "config.json")
    incremental = Pipeline(config, str(workspace), snapshot=False)
    incremental.load_all()
    for seq in sorted(incremental.fragments):
        incremental._rebuild_front_stage()
    incremental.run_batch(budget=1000)
    assert _serial_signature(incremental) == _serial_signature(full)


def test_quant_level_update_invalidates_all_windows(workspace, cfg):
    pipeline, _ = _full_run(cfg, workspace)
    before = {i: p.scores for i, p in pipeline.posteriors.items()}
    invalidated = pipeline.update_quant_level(24)
    assert sorted(invalidated) == sorted(pipeline.windows)
    pipeline.run_batch(budget=1000)
    after = {i: p.scores for i, p in pipeline.posteriors.items()}
    assert before != after


def test_snapshot_resume_and_rebuild_decision(workspace, cfg):
    config = load_config(workspace / "config.json")
    pipeline = Pipeline(config, str(workspace), snapshot=True)
    pipeline.load_all()
    pipeline.run_batch(budget=5)
    assert (workspace / config.paths.snapshot_dir / "snapshot_manifest.json").exists()
    restored = Pipeline(config, str(workspace), snapshot=True)
    assert restored.resume_if_possible() == "resume"
    assert restored.pending == pipeline.pending
    assert restored.global_version == pipeline.global_version
    config2 = load_config(workspace / "config.json")
    config2.quant.quant_level = 32
    rebuilt = Pipeline(config2, str(workspace), snapshot=True)
    assert rebuilt.resume_if_possible() == "rebuild"


def test_snapshot_atomic_replace_keeps_last_good_state(workspace, cfg):
    config = load_config(workspace / "config.json")
    pipeline = Pipeline(config, str(workspace), snapshot=True)
    pipeline.load_all()
    pipeline.run_batch(budget=5)
    manifest_path = workspace / config.paths.snapshot_dir / "snapshot_manifest.json"
    first_manifest = manifest_path.read_text("utf-8")
    pipeline.run_batch(budget=5)
    second_manifest = manifest_path.read_text("utf-8")
    assert first_manifest != second_manifest
    assert len(pipeline.features) == 10


def test_injection_isolation_and_report(workspace, cfg):
    injection = InjectionConfig.model_validate({
        "entries": [
            {"id": "I1", "type": "span_out_of_range", "window_index": 2,
             "stage": "window_feature", "enabled": True},
            {"id": "I2", "type": "posterior_dimension_mismatch", "window_index": 5,
             "stage": "classify", "enabled": True},
        ]
    })
    config = load_config(workspace / "config.json")
    pipeline = Pipeline(config, str(workspace), injection_config=injection, snapshot=False)
    pipeline.load_all()
    pipeline.run_batch(budget=1000)
    hit_ids = {hit["injection_id"] for hit in pipeline.engine.report.hits}
    assert hit_ids == {"I1", "I2"}
    failed_windows = {
        record["window_index"] for record in pipeline.engine.report.failures
    }
    assert failed_windows == {2, 5}
    assert 2 not in pipeline.features or 5 not in pipeline.features
    healthy = [index for index in pipeline.windows if index not in (2, 5)]
    assert all(index in pipeline.posteriors for index in healthy)


def test_load_stage_injection_failures_are_listed(workspace, cfg):
    injection = InjectionConfig.model_validate({
        "entries": [
            {"id": "T1", "type": "truncate_wav", "fragment_seq": 1,
             "stage": "audio_load", "enabled": True},
            {"id": "N1", "type": "non_finite_sample", "fragment_seq": 3,
             "stage": "audio_load", "enabled": True},
        ]
    })
    config = load_config(workspace / "config.json")
    pipeline = Pipeline(config, str(workspace), injection_config=injection, snapshot=False)
    failures = pipeline.load_all()
    assert {record["fragment_seq"] for record in failures} == {1, 3}
    assert {record["code"] for record in failures} == {
        "E_AUDIO_FORMAT", "E_SAMPLE_NON_FINITE"
    }
    assert 1 in pipeline.failed_fragments and 3 in pipeline.failed_fragments


def test_bad_quant_injection_isolates_single_window(workspace, cfg):
    injection = InjectionConfig.model_validate({
        "entries": [
            {"id": "Q1", "type": "bad_quant_level", "window_index": 3,
             "stage": "quantize", "enabled": True},
        ]
    })
    config = load_config(workspace / "config.json")
    pipeline = Pipeline(config, str(workspace), injection_config=injection, snapshot=False)
    pipeline.load_all()
    pipeline.run_batch(budget=1000)
    codes = {record["code"] for record in pipeline.engine.report.failures}
    assert "E_QUANT_LEVEL_INVALID" in codes
    assert 3 in pipeline.failed_windows
    healthy = [i for i in pipeline.windows if i != 3]
    assert all(i in pipeline.posteriors for i in healthy)


def test_worker_count_never_exceeds_configured_max(workspace, cfg, monkeypatch):
    import voicecmd.pipeline as pipeline_mod
    observed = []
    real_init = pipeline_mod.ThreadPoolExecutor.__init__

    def tracking_init(self, max_workers=None, *args, **kwargs):
        observed.append(max_workers)
        return real_init(self, max_workers=max_workers, *args, **kwargs)

    monkeypatch.setattr(pipeline_mod.ThreadPoolExecutor, "__init__", tracking_init)
    config = load_config(workspace / "config.json")
    config.runtime.max_workers = 3
    pipeline = Pipeline(config, str(workspace), snapshot=False)
    pipeline.load_all()
    pipeline.run_batch(budget=2)
    assert observed and all(workers <= 3 for workers in observed)


def test_arbitrary_delivery_order_matches_sorted_order(workspace, cfg):
    import json
    from voicecmd.audio import Fragment, read_wav

    config = load_config(workspace / "config.json")
    manifest = json.loads((workspace / cfg.paths.manifest_file).read_text("utf-8"))
    entries = manifest["fragments"]
    ordered_pipeline = Pipeline(config, str(workspace), snapshot=False)
    ordered_pipeline.load_all()
    ordered_pipeline.run_batch(budget=1000)

    shuffled = Pipeline(config, str(workspace), snapshot=False)
    for entry in reversed(entries):
        samples = read_wav(workspace / cfg.paths.data_dir / entry["wav_file"])
        shuffled.ingest_fragment(Fragment(
            seq=entry["seq"], wav_file=entry["wav_file"],
            arrival_ms=entry["arrival_ms"], capture_ms=entry["capture_ms"],
            command_word=entry.get("command_word"),
            samples=samples, sample_count=len(samples),
        ))
    shuffled.run_batch(budget=1000)
    assert _serial_signature(shuffled) == _serial_signature(ordered_pipeline)
    assert [w.fragment_seqs for w in shuffled.windows.values()] == \
           [w.fragment_seqs for w in ordered_pipeline.windows.values()]
