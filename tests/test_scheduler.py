import json
import socket
import threading
import time
import unittest

from cfgpipe.scheduler import Dispatcher, DispatchTask


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class SlowServer(threading.Thread):
    """记录收到的每个字节并延迟应答的桩服务器。"""

    def __init__(self, port, delay_s=0.0, payloads=None, drop_first=0):
        super().__init__(daemon=True)
        self.port = port
        self.delay_s = delay_s
        self.received = []
        self._payloads = payloads or []
        self._drop_first = drop_first
        self._halt = False
        # 构造时同步完成 bind+listen，避免客户端竞态
        self._srv = socket.socket()
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", self.port))
        self._srv.listen(8)
        self._srv.settimeout(0.2)

    def run(self):
        srv = self._srv
        while not self._halt:
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            with conn:
                buf = b""
                while not buf.endswith(b"\n"):
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
                self.received.append(buf.strip())
                if self._drop_first > 0:
                    self._drop_first -= 1
                    continue  # 直接关闭连接，不回复
                if self.delay_s:
                    time.sleep(self.delay_s)
                resp = self._payloads.pop(0) if self._payloads else {
                    "type": "receipt", "node_id": "stub", "batch_id": "b",
                    "digest": "d", "config_version": 1, "elapsed_ms": 1,
                    "scale_exp": 3}
                conn.sendall(json.dumps(resp).encode() + b"\n")
        srv.close()

    def stop(self):
        self._halt = True
        self.join(timeout=3)


class TestScheduler(unittest.TestCase):
    def test_concurrency_limits_and_wait_recorded(self):
        port = free_port()
        server = SlowServer(port, delay_s=0.15)
        server.start()
        try:
            disp = Dispatcher(global_limit=2, per_node_limit=1,
                              timeout_ms=2000, backoff_ms=(10,))
            tasks = [DispatchTask("n%d" % i, "127.0.0.1", port,
                                  b'{"type":"dispatch","i":%d}' % i, "b")
                     for i in range(5)]
            start = time.monotonic()
            results = disp.run(tasks)
            elapsed = time.monotonic() - start
            self.assertTrue(all(r.ok for r in results))
            # 全局上限 2、每个 0.15s：5 个任务至少需 3 波次
            self.assertGreaterEqual(elapsed, 0.40)
            # 排队等待起止时间已记录
            for res in results:
                self.assertGreaterEqual(res.wait_end_ms, res.wait_start_ms)
        finally:
            server.stop()

    def test_retry_keeps_payload_bytes(self):
        port = free_port()
        # 第一次连接直接丢弃（不回复）触发超时重试
        server = SlowServer(port, drop_first=2)
        server.start()
        try:
            disp = Dispatcher(global_limit=1, per_node_limit=1,
                              timeout_ms=500, backoff_ms=(20, 20, 20))
            payload = b'{"type":"dispatch","batch_id":"b1","v":9}'
            results = disp.run([DispatchTask("n1", "127.0.0.1", port,
                                             payload, "b1")])
            self.assertTrue(results[0].ok)
            self.assertGreaterEqual(results[0].attempts, 2)
            # 重试期间线路字节逐字节不变
            self.assertTrue(all(rec == payload for rec in server.received))
        finally:
            server.stop()

    def test_retry_exhaustion_reports_error(self):
        port = free_port()  # 无服务器监听
        disp = Dispatcher(global_limit=1, per_node_limit=1,
                          timeout_ms=100, backoff_ms=(10, 10))
        results = disp.run([DispatchTask("n1", "127.0.0.1", port,
                                         b"{}", "b")])
        self.assertFalse(results[0].ok)
        self.assertEqual(results[0].attempts, 3)
        self.assertTrue(results[0].error)


if __name__ == "__main__":
    unittest.main()
