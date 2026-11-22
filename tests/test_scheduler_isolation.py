from __future__ import annotations

from corpus_workbench.cache import SignatureCache
from corpus_workbench.codes import E_MINHASH_FAILED
from corpus_workbench.config import DedupConfig, SchedulerConfig
from corpus_workbench.models import Document, content_fingerprint
from corpus_workbench import scheduler as scheduler_module
from corpus_workbench.scheduler import BudgetScheduler


def _document(doc_id: str, text: str) -> Document:
    return Document(
        source="alpha",
        doc_id=doc_id,
        text=text,
        metadata={},
        fingerprint=content_fingerprint(text),
        file="memory.jsonl",
        line_no=1,
    )


def test_single_document_exception_is_isolated(monkeypatch, tmp_path):
    docs = [
        _document("d-01", "red green blue yellow black white orange"),
        _document("d-02", "silver copper iron nickel zinc tin lead"),
        _document("d-03", "planet orbit vector signal garden machine archive"),
    ]
    original = scheduler_module.shingle_hashes

    def broken(text: str, shingle_length: int):
        if "silver" in text:
            raise RuntimeError("injected minhash-stage failure")
        return original(text, shingle_length)

    monkeypatch.setattr(scheduler_module, "shingle_hashes", broken)
    scheduler = BudgetScheduler(
        SchedulerConfig(
            max_workers=3,
            global_shingle_budget=1000,
            source_quotas={"alpha": 1000},
            batch_size=3,
            snapshot_interval_batches=1,
        ),
        DedupConfig(shingle_length=2, signature_dim=32, jaccard_threshold=0.8),
        SignatureCache(tmp_path / "cache"),
    )
    result = scheduler.execute(docs)

    assert len(result.signatures) == 2
    assert len(result.failures) == 1
    assert result.failures[0].doc_id == "d-02"
    assert result.failures[0].stage == "minhash"
    assert result.failures[0].code == E_MINHASH_FAILED
    assert result.consumed_points > 0

