"""会话运行器：顺序推进轨迹、驱动并发重建、记录结论并产出快照。"""

from __future__ import annotations

import os
import threading
from typing import Any, Optional

from . import codec
from .engine import Engine
from .rebuild import RebuildWorker
from .snapshot import save_snapshot
from .storage import Storage
from .structlog import StructuredLogger
from .trace import operations_digest


class Session:
    def __init__(
        self,
        db_path: str,
        logger: Optional[StructuredLogger] = None,
        initial_records: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        self.storage = Storage(db_path)
        self.engine = Engine(self.storage)
        self.logger = logger or StructuredLogger()
        if initial_records is not None:
            self.engine.load_initial_records(initial_records)
        self.rebuild_worker = RebuildWorker(self.storage, self.logger)
        self.engine.delta_listener = self.rebuild_worker.on_delta
        self.conclusions: list[dict[str, Any]] = []
        self.trace_digest: dict[str, Any] = {}

    def run(
        self,
        operations: list[dict[str, Any]],
        *,
        rebuild_barrier: Optional[threading.Event] = None,
        during_rebuild_queries: Optional[list[dict[str, Any]]] = None,
        query_clock: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        self.trace_digest = operations_digest(operations)
        self.logger.log("trace_loaded", **self.trace_digest)
        for operation in operations:
            conclusion = self.step(operation, rebuild_barrier)
            self.conclusions.append(conclusion)
            self.engine.set_trace_cursor(operation["op_seq"])
            # 重建挂起期间执行前台查询，验证并发一致视图
            if (
                rebuild_barrier is not None
                and self.rebuild_worker.is_running
                and during_rebuild_queries is not None
            ):
                clock = query_clock if query_clock is not None else operation["clock"]
                for prefix in during_rebuild_queries:
                    snapshot = self.engine.query_prefix(prefix, clock)
                    self.logger.log("during_rebuild_query", **snapshot)
        # 确保所有重建线程收尾
        if self.rebuild_worker.thread is not None:
            self.rebuild_worker.join()
        return self.conclusions

    def step(
        self,
        operation: dict[str, Any],
        rebuild_barrier: Optional[threading.Event] = None,
    ) -> dict[str, Any]:
        seq = operation["op_seq"]
        name = operation["op"]
        clock = operation["clock"]
        if name == "PUT":
            conclusion = self.engine.put(
                operation["key"], operation["value"],
                operation["expire_ms"], seq, clock,
            )
        elif name == "GET":
            conclusion = self.engine.get(operation["key"], clock)
            self.logger.log("visibility_judgment",
                            **self.engine.visibility_basis(operation["key"], clock))
        elif name == "DEL":
            conclusion = self.engine.delete(operation["key"], clock)
        elif name == "PURGE":
            conclusion = self.engine.purge(clock)
            for basis in conclusion["basis"]:
                self.logger.log("purge_judgment", **basis)
        elif name == "REBUILD_INDEX":
            self.rebuild_worker.start(clock, barrier=rebuild_barrier)
            if rebuild_barrier is None:
                rebuild_result = self.rebuild_worker.join()
            else:
                rebuild_result = {"barred": True}
            conclusion = {"op": "REBUILD_INDEX", "op_seq": seq, "clock": clock,
                          "rebuild": rebuild_result,
                          "keyspace_version": self.engine.keyspace_version}
            self.logger.log("rebuild_op", op_seq=seq, **rebuild_result)
        else:  # pragma: no cover
            raise ValueError(name)
        conclusion = dict(conclusion)
        conclusion["op_seq"] = seq
        return conclusion

    def finish_rebuild(self) -> dict[str, Any]:
        return self.rebuild_worker.join()

    def snapshot(self, path: str) -> dict[str, Any]:
        rebuild_progress = {"phase": self.rebuild_worker.phase,
                            "result": self.rebuild_worker.result}
        return save_snapshot(
            path, self.engine, self.conclusions,
            rebuild_progress=rebuild_progress,
            trace_digest=self.trace_digest,
        )

    def close(self) -> None:
        if self.rebuild_worker.thread is not None and self.rebuild_worker.is_running:
            self.rebuild_worker.join(timeout=5)
        self.storage.close()
