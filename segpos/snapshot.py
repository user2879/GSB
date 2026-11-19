"""版本化快照：临时目录暂存 + 文件级原子替换 + manifest 提交。"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .errors import ConfigMismatchError
from .fingerprints import sha256_hex, stable_sort_key
from .models import SentenceRecord

MANIFEST_NAME = "manifest.json"
RESULTS_DIR = "results"
TMP_PREFIX = ".tmp-"


@dataclass
class Manifest:
    global_version: int
    dict_hash: str
    prob_table_hash: str
    prob_table_version: int
    completed_sentences: List[str]
    artifact_versions: Dict[str, int]
    failures: List[dict]

    def to_dict(self) -> dict:
        return {
            "global_version": self.global_version,
            "dict_hash": self.dict_hash,
            "prob_table_hash": self.prob_table_hash,
            "prob_table_version": self.prob_table_version,
            "completed_sentences": list(self.completed_sentences),
            "artifact_versions": dict(self.artifact_versions),
            "failures": list(self.failures),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Manifest":
        return cls(
            global_version=int(data["global_version"]),
            dict_hash=data["dict_hash"],
            prob_table_hash=data["prob_table_hash"],
            prob_table_version=int(data.get("prob_table_version", 1)),
            completed_sentences=list(data["completed_sentences"]),
            artifact_versions=dict(data["artifact_versions"]),
            failures=list(data.get("failures", [])),
        )


def atomic_write_text(path: Path, content: str) -> None:
    """同目录临时文件 + os.replace 原子替换（Windows/Linux 均支持）。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=TMP_PREFIX, dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, payload: dict) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
    )


class SnapshotStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.results_dir = self.root / RESULTS_DIR
        self.manifest_path = self.root / MANIFEST_NAME
        self.global_version = 0
        self.artifact_versions: Dict[str, int] = {
            "segmentation": 0,
            "oov": 0,
            "tagging": 0,
            "summary": 0,
        }

    def exists(self) -> bool:
        return self.manifest_path.exists()

    def load_manifest(self) -> Manifest:
        return Manifest.from_dict(
            json.loads(self.manifest_path.read_text(encoding="utf-8"))
        )

    def verify_resume(
        self, dict_hash: str, prob_table_hash: str, prob_table_version: int
    ) -> Manifest:
        """启动时哈希/版本判定；不一致抛 E_CONFIG_MISMATCH 拒绝续跑。"""

        manifest = self.load_manifest()
        mismatches = []
        if manifest.dict_hash != dict_hash:
            mismatches.append("dict_hash")
        if manifest.prob_table_hash != prob_table_hash:
            mismatches.append("prob_table_hash")
        if manifest.prob_table_version != prob_table_version:
            mismatches.append("prob_table_version")
        if mismatches:
            raise ConfigMismatchError(
                "快照与当前输入不一致，拒绝续跑: " + ", ".join(mismatches),
                code="E_CONFIG_MISMATCH",
            )
        self.global_version = manifest.global_version
        self.artifact_versions = dict(manifest.artifact_versions)
        return manifest

    def load_results(self, manifest: Manifest) -> List[SentenceRecord]:
        records: List[SentenceRecord] = []
        for sentence_id in manifest.completed_sentences:
            payload = json.loads(
                self._result_path(sentence_id).read_text(encoding="utf-8")
            )
            records.append(_record_from_payload(payload))
        return records

    @staticmethod
    def _result_path_for(root: Path, sentence_id: str) -> Path:
        return root / RESULTS_DIR / f"{sha256_hex(sentence_id)}.json"

    def _result_path(self, sentence_id: str) -> Path:
        return self._result_path_for(self.root, sentence_id)

    def commit(
        self,
        records: List[SentenceRecord],
        failures: List[dict],
        dict_hash: str,
        prob_table_hash: str,
        prob_table_version: int,
    ) -> Manifest:
        """分词/OOV/标注/汇总产物写临时文件后整体替换，再提交 manifest。"""

        staged_root = self.root / f"{TMP_PREFIX}commit-{os.getpid()}-{self.global_version}"
        staged_results = staged_root / RESULTS_DIR
        staged_results.mkdir(parents=True, exist_ok=True)

        ordered = sorted(records, key=lambda item: item.sentence_id)
        completed: List[str] = []
        for record in ordered:
            target = self._result_path_for(staged_root, record.sentence_id)
            atomic_write_json(target, record.to_dict())
            completed.append(record.sentence_id)

        staged_manifest = Manifest(
            global_version=self.global_version + 1,
            dict_hash=dict_hash,
            prob_table_hash=prob_table_hash,
            prob_table_version=prob_table_version,
            completed_sentences=completed,
            artifact_versions={
                stage: version + 1
                for stage, version in self.artifact_versions.items()
            },
            failures=sorted(
                failures,
                key=lambda item: stable_sort_key(item["sentence_id"], item["stage"]),
            ),
        )
        staged_manifest_path = staged_root / MANIFEST_NAME
        atomic_write_json(staged_manifest_path, staged_manifest.to_dict())

        self.root.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        for staged_file in staged_results.iterdir():
            os.replace(staged_file, self.results_dir / staged_file.name)

        live_hashes = {sha256_hex(sentence_id) for sentence_id in completed}
        for stale in self.results_dir.glob("*.json"):
            if stale.stem not in live_hashes:
                stale.unlink()

        os.replace(staged_manifest_path, self.manifest_path)
        try:
            staged_root.rmdir()
        except OSError:
            pass
        self.global_version = staged_manifest.global_version
        self.artifact_versions = staged_manifest.artifact_versions
        return staged_manifest

def _record_from_payload(payload: dict) -> SentenceRecord:
    from .models import Token

    tokens = [
        Token(
            start=item["start"],
            end=item["end"],
            text=item["text"],
            match_basis=item["match_basis"],
            hit=item["hit"],
            boundary_confidence=item["boundary_confidence"],
            pos=item.get("pos"),
            local_path_score=item.get("local_path_score"),
        )
        for item in payload["tokens"]
    ]
    return SentenceRecord(
        sentence_id=payload["sentence_id"],
        text=payload["text"],
        tokens=tokens,
        cache_key=payload["cache_key"],
        reused=payload.get("reused", False),
        sentence_logprob=payload["sentence_logprob"],
        smoothing_hits=payload.get("smoothing_hits", 0),
        low_confidence_words=payload.get("low_confidence_words", []),
    )
