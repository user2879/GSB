"""专用账本写入线程。

SQLite 连接具有线程亲和性，而下发真实发生在线程池工作线程中。账本连接在
本线程内创建，所有写操作通过唯一队列在该线程串行执行，工作线程同步等待
提交结果，保证连接亲和性与事务原子性。读操作使用独立只读连接。
"""

import queue
import threading


class LedgerThread:
    def __init__(self, ledger_factory):
        self._q = queue.Queue()
        self._ready = threading.Event()
        self.ledger = None
        self._init_error = None
        self._worker = threading.Thread(target=self._loop,
                                        args=(ledger_factory,),
                                        name="ledger-writer", daemon=True)
        self._worker.start()
        self._ready.wait()
        if self._init_error is not None:
            raise self._init_error

    def _loop(self, ledger_factory):
        try:
            self.ledger = ledger_factory()
        except Exception as exc:  # noqa: BLE001
            self._init_error = exc
            self._ready.set()
            return
        self._ready.set()
        while True:
            job = self._q.get()
            if job is None:
                self._q.task_done()
                return
            fn, result = job
            try:
                result["value"] = fn()
            except Exception as exc:  # noqa: BLE001
                result["error"] = exc
            result["event"].set()
            self._q.task_done()

    def submit(self, fn):
        result = {"event": threading.Event()}
        self._q.put((fn, result))
        result["event"].wait()
        if "error" in result:
            raise result["error"]
        return result["value"]

    def shutdown(self):
        self._q.put(None)
        self._worker.join(timeout=10)
