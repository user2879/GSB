"""页级恢复: 先扫缓冲选可用镜像, 再逐页比对修复主文件, 最后开放键空间。

判定真值来源(要求 4): 双写缓冲中摘要匹配且序号最新的镜像,
与主文件槽位摘要逐页比对。整个过程不使用零填充冒充恢复。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .buffer import DoubleWriteBuffer, Mirror, ScanResult
from .errors import LostPageError
from .keyspace import Keyspace
from .paging import MainDataFile, Page
from .util import StructuredLogger, digest_of_json
from .versions import VersionMap


@dataclass
class RepairEntry:
    page_no: int
    seq: int
    reason: str          # torn_main / digest_mismatch / missing_slot
    before_digest: str | None
    after_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_no": self.page_no,
            "seq": self.seq,
            "reason": self.reason,
            "before_digest": self.before_digest,
            "after_digest": self.after_digest,
        }


@dataclass
class RecoveryReport:
    repaired: list[RepairEntry]
    intact_pages: list[int]
    discarded_mirrors: int
    max_seq: int
    final_digests: dict[int, str]
    lost_pages: list[dict[str, Any]] = field(default_factory=list)
    truth_source: str = (
        "per-page comparison of latest digest-matched buffer mirrors "
        "against main-file slot digests"
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "repaired_count": len(self.repaired),
            "intact_count": len(self.intact_pages),
            "discarded_mirrors": self.discarded_mirrors,
            "max_seq": self.max_seq,
            "repaired": [e.to_dict() for e in self.repaired],
            "intact_pages": list(self.intact_pages),
            "final_page_digests": {
                str(k): self.final_digests[k] for k in sorted(self.final_digests)
            },
            "lost_pages": list(self.lost_pages),
            "truth_source": self.truth_source,
        }

    def digest(self) -> str:
        return digest_of_json(self.to_dict())


class RecoveryEngine:
    def __init__(
        self,
        main: MainDataFile,
        dwb: DoubleWriteBuffer,
        versions: VersionMap,
        keyspace: Keyspace,
        logger: StructuredLogger,
    ) -> None:
        self.main = main
        self.dwb = dwb
        self.versions = versions
        self.keyspace = keyspace
        self.logger = logger

    def run(self, *, raise_on_lost: bool = True) -> RecoveryReport:
        # 1) 容错扫描缓冲: 结构错误仍 E_PAGE, 摘要坏镜像计数丢弃
        scan: ScanResult = self.dwb.scan(strict_digest=False)
        latest = scan.latest_per_page

        repaired: list[RepairEntry] = []
        intact: list[int] = []
        lost: list[dict[str, Any]] = []

        # 期望页集合 = 版本映射已知页 ∪ 缓冲出现过的页(并集, 稳定升序)
        expected = set(self.versions.pages) | {m.page_no for m in scan.mirrors}
        for page_no in sorted(expected):
            mirror: Mirror | None = latest.get(page_no)
            current = self.main.read_slot(page_no)
            current_digest = Page.digest_hex(current) if current is not None else None
            version = self.versions.get(page_no)
            expected_digest = None
            if mirror is not None:
                expected_digest = mirror.digest
            elif version is not None:
                expected_digest = bytes.fromhex(version.digest)

            if mirror is None:
                # 缓冲无可用镜像
                if current is not None and Page.structural_valid(current, self.main.page_size):
                    if expected_digest is None or Page.digest(current) == expected_digest:
                        intact.append(page_no)
                        continue
                # 主文件也不可用 => E_LOST, 禁止零填充
                lost.append(
                    {"page_no": page_no, "max_seq": scan.max_seq}
                )
                self.logger.log(
                    "lost_page",
                    code="E_LOST",
                    detail={"page_no": page_no, "max_seq": scan.max_seq},
                )
                if raise_on_lost:
                    raise LostPageError(
                        f"page {page_no} has no usable mirror in main or buffer",
                        page_no=page_no,
                        max_seq=scan.max_seq,
                    )
                continue

            # 2) 逐页比对主文件摘要与最新可用缓冲镜像
            if current is not None and Page.structural_valid(
                current, self.main.page_size
            ) and Page.digest(current) == mirror.digest:
                intact.append(page_no)
                # 版本映射补齐(保证映射与缓冲真值一致)
                self.versions.upsert(page_no, mirror.seq, mirror.digest.hex())
                continue

            reason = self._classify_reason(current)
            before = current_digest
            # 3) 用缓冲完整镜像修复主文件
            self.main.write_slot(page_no, mirror.frame)
            self.versions.upsert(page_no, mirror.seq, mirror.digest.hex())
            entry = RepairEntry(
                page_no=page_no,
                seq=mirror.seq,
                reason=reason,
                before_digest=before,
                after_digest=mirror.digest.hex(),
            )
            repaired.append(entry)
            self.logger.log(
                "page_repaired",
                detail=entry.to_dict(),
            )

        self.versions.save()
        final_digests = self.main.iter_digests()
        report = RecoveryReport(
            repaired=repaired,
            intact_pages=sorted(intact),
            discarded_mirrors=scan.discarded_count,
            max_seq=scan.max_seq,
            final_digests=final_digests,
            lost_pages=lost,
        )
        self.logger.log("recovery_report", detail=report.to_dict())
        # 4) 恢复完成后才开放键空间作答
        self.keyspace.mark_ready()
        return report

    @staticmethod
    def _classify_reason(current: bytes | None) -> str:
        if current is None:
            return "missing_slot"
        if not Page.structural_valid(current, len(current)):
            return "torn_main"
        return "digest_mismatch"
