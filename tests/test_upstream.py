import socketserver
import threading
import unittest

from dns_workbench.errors import ChainError, TxnError, UpstreamError
from dns_workbench.upstream import TcpUpstreamClient
from dns_workbench.wire import (
    CLASS_IN,
    TYPE_A,
    TYPE_CNAME,
    DnsMessage,
    Question,
    ResourceRecord,
    build_response_flags,
    encode_tcp_frame,
    parse_message,
    read_tcp_frame,
)


class _Handler(socketserver.BaseRequestHandler):
    mode = "ok"

    def handle(self) -> None:
        request = parse_message(read_tcp_frame(self.request))
        question = request.first_question()
        txn_id = request.txn_id ^ 0xFFFF if self.mode == "txn" else request.txn_id
        answers = [
            ResourceRecord(question.qname, TYPE_CNAME, CLASS_IN, 300, b"\x04edge\x07example\x04test\x00", "edge.example.test"),
            ResourceRecord("edge.example.test", TYPE_A, CLASS_IN, 300, bytes([10, 0, 0, 5]), "10.0.0.5"),
        ]
        if self.mode == "chain":
            answers[1] = ResourceRecord("broken.example.test", TYPE_A, CLASS_IN, 300, bytes([1, 2, 3, 4]), "1.2.3.4")
        response = DnsMessage(txn_id, build_response_flags(rcode=0, aa=False), [question], answers)
        self.request.sendall(encode_tcp_frame(response))


class _Server(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


class UpstreamTest(unittest.TestCase):
    def _start(self, mode: str) -> tuple[_Server, int]:
        handler = type("H", (_Handler,), {"mode": mode})
        server = _Server(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        return server, server.server_address[1]

    def _request(self) -> DnsMessage:
        return DnsMessage(0x2222, 0x0100, [Question("www.example.test", TYPE_A)])

    def test_forward_ok_and_chain_verified(self) -> None:
        _server, port = self._start("ok")
        result = TcpUpstreamClient(port=port, retries=1).forward(self._request())
        self.assertTrue(result.consistent)
        self.assertEqual(result.attempts, 1)
        self.assertEqual(len(result.response.answers), 2)

    def test_bad_txn_is_e_txn(self) -> None:
        _server, port = self._start("txn")
        with self.assertRaises(TxnError) as ctx:
            TcpUpstreamClient(port=port, retries=0).forward(self._request())
        self.assertEqual(ctx.exception.fields, ["txn_id"])

    def test_bad_chain_is_e_chain(self) -> None:
        _server, port = self._start("chain")
        with self.assertRaises(ChainError) as ctx:
            TcpUpstreamClient(port=port, retries=0).forward(self._request())
        self.assertIn("answers[1].name", ctx.exception.fields)

    def test_unreachable_is_e_upstream_with_retry(self) -> None:
        _server, port = self._start("ok")
        _server.shutdown()
        _server.server_close()
        with self.assertRaises(UpstreamError) as ctx:
            TcpUpstreamClient(port=port, timeout=0.2, retries=2).forward(self._request())
        self.assertEqual(ctx.exception.context["attempts"], 3)


if __name__ == "__main__":
    unittest.main()
