import copy

from depedit.cache import IncrementalCache
from depedit.codecs import load_json, v1_parse_document, load_text
from depedit.config import AppConfig
from depedit.errors import ESnapshotStale
from depedit.fingerprints import model_fingerprint, mapping_fingerprint
from depedit.pipeline import Pipeline, fingerprints_of_inputs
from depedit.relations import load_mapping
from depedit.snapshot import SnapshotStore
from depedit.injection import InjectionEngine


def _corpus(sample_dir):
    cfg = AppConfig()
    mapping = load_mapping(sample_dir / "relation_map.json")
    sentences = v1_parse_document(
        load_text(sample_dir / "parse_v1.txt"), cfg, mapping["v1_to_v2"]
    )
    from depedit.models import Corpus

    return Corpus(sentences={s.sent_id: s for s in sentences}), cfg, mapping


def test_concurrency_independent_ordering(sample_dir):
    corpus, cfg, mapping = _corpus(sample_dir)
    model_fp, mapping_fp = fingerprints_of_inputs(
        sample_dir / "model_params.json", sample_dir / "relation_map.json"
    )
    serialized = {}
    for workers in (1, 2, 4, 8, 16):
        cfg_work = copy.deepcopy(cfg)
        cfg_work.max_workers = workers
        pipe = Pipeline(cfg_work, mapping, InjectionEngine([]))
        result = pipe.fingerprint_and_serialize(corpus, model_fp, mapping_fp, "v1")
        ordered = {sid: result.artifacts[sid] for sid in sorted(result.artifacts)}
        serialized[workers] = ordered
    for workers in (2, 4, 8, 16):
        assert serialized[workers] == serialized[1]


def test_reusable_sentences_not_reparsed(sample_dir):
    corpus, cfg, mapping = _corpus(sample_dir)
    model_fp, mapping_fp = fingerprints_of_inputs(
        sample_dir / "model_params.json", sample_dir / "relation_map.json"
    )
    cache = IncrementalCache(model_fp, mapping_fp, "v1")
    pipe = Pipeline(cfg, mapping, InjectionEngine([]))
    result = pipe.fingerprint_and_serialize(corpus, model_fp, mapping_fp, "v1", cache)
    # 无变更时全部句子可复用。
    assert set(result.reusable) == set(corpus.sentences)
    # 样本替换仅失效对应句子。
    cache.invalidate_sample_replaced(corpus, {"S002"})
    reusable = set(cache.reusable_sentences())
    assert "S002" not in reusable
    assert "S001" in reusable


def test_mapping_change_scoped_invalidation(sample_dir):
    corpus, cfg, mapping = _corpus(sample_dir)
    model_fp, mapping_fp = fingerprints_of_inputs(
        sample_dir / "model_params.json", sample_dir / "relation_map.json"
    )
    cache = IncrementalCache(model_fp, mapping_fp, "v1")
    pipe = Pipeline(cfg, mapping, InjectionEngine([]))
    pipe.fingerprint_and_serialize(corpus, model_fp, mapping_fp, "v1", cache)
    target_rel = next(iter(mapping["named_relations"]))
    report = cache.invalidate_mapping_change(corpus, {target_rel})
    affected = {item["sent_id"] for item in report}
    # 只有包含该关系的句子失效。
    for sid in corpus.sentences:
        has_rel = any(a.relation == target_rel for a in corpus.sentences[sid].arcs)
        assert (sid in affected) == has_rel


def test_model_change_invalidates_all(sample_dir):
    corpus, cfg, mapping = _corpus(sample_dir)
    model_fp, mapping_fp = fingerprints_of_inputs(
        sample_dir / "model_params.json", sample_dir / "relation_map.json"
    )
    cache = IncrementalCache(model_fp, mapping_fp, "v1")
    pipe = Pipeline(cfg, mapping, InjectionEngine([]))
    pipe.fingerprint_and_serialize(corpus, model_fp, mapping_fp, "v1", cache)
    cache.invalidate_model_change(corpus, "changed-model-fp")
    assert cache.reusable_sentences() == []


def test_snapshot_commit_and_stale_recovery(sample_dir, tmp_path):
    corpus, cfg, mapping = _corpus(sample_dir)
    model_fp, mapping_fp = fingerprints_of_inputs(
        sample_dir / "model_params.json", sample_dir / "relation_map.json"
    )
    pipe = Pipeline(cfg, mapping, InjectionEngine([]))
    result = pipe.fingerprint_and_serialize(corpus, model_fp, mapping_fp, "v1")
    store = SnapshotStore(tmp_path / "snap")
    store.commit(
        result.corpus,
        global_version=3,
        model_fp=model_fp,
        mapping_fp=mapping_fp,
        artifacts=result.artifacts,
        degradation=result.degradation,
    )
    # 一致时恢复通过。
    manifest = store.verify_recovery(
        corpus, expected_version=3, model_fp=model_fp, mapping_fp=mapping_fp
    )
    assert manifest["global_version"] == 3
    # 版本不一致时拒绝并保留旧快照。
    with __import__("pytest").raises(ESnapshotStale):
        store.verify_recovery(
            corpus, expected_version=99, model_fp=model_fp, mapping_fp=mapping_fp
        )
    assert store.load_manifest()["global_version"] == 3
