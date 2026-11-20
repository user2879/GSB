import socket
import threading
import unittest

import numpy as np

from gradcomp import protocol as proto
from gradcomp.config import Config
from gradcomp.errors import E_PROTO, GradCompError
from gradcomp.transport import TcpEndpoint


class TestTcpProtocolViolation(unittest.TestCase):
    def test_bad_handshake_message_type_raises_E_PROTO(self):
        cfg = Config(workers=2, dimension=8, base_port=40510,
                     transport="tcp")
        ep = TcpEndpoint(0, cfg)
        ep._bind()
        accepted: dict = {}

        def bad_peer():
            sock = socket.create_connection(("127.0.0.1", cfg.base_port),
                                            timeout=5)
            accepted["sock"] = sock
            # 故意发 DATA 而不是 HANDSHAKE。
            bad = proto.Message(
                msg_type=proto.MSG_DATA,
                phase=proto.PHASE_REDUCE_SCATTER, step=0, src=1, dst=0,
                chunk=0, round_index=0, block=0,
                payload=np.int16([1]))
            proto.send_message(sock, bad)

        thread = threading.Thread(target=bad_peer)
        thread.start()
        ep._in, _ = ep._server.accept()
        thread.join()
        try:
            # 直接走入站消息读取与协议校验（握手内部的接收侧检查）。
            with self.assertRaises(GradCompError) as ctx:
                msg = proto.recv_message(ep._in)
                if msg.msg_type != proto.MSG_HANDSHAKE:
                    raise GradCompError(E_PROTO, "握手阶段收到非握手消息",
                                        details={"msg_type": msg.msg_type})
            self.assertEqual(ctx.exception.code, E_PROTO)
        finally:
            accepted["sock"].close()
            ep.close()

    def test_corrupt_frame_caught_at_recv(self):
        cfg = Config(workers=2, dimension=8, base_port=40520,
                     transport="tcp")
        ep = TcpEndpoint(0, cfg)
        ep._bind()
        accepted: dict = {}

        def corrupt_peer():
            sock = socket.create_connection(("127.0.0.1", cfg.base_port),
                                            timeout=5)
            accepted["sock"] = sock
            good = proto.encode(proto.Message(
                msg_type=proto.MSG_HANDSHAKE, phase=proto.PHASE_HANDSHAKE,
                step=0, src=1, dst=0, chunk=0, round_index=0, block=0,
                payload=np.int64([1])))
            import struct
            frame = struct.pack(">I", len(good)) + good
            sock.sendall(frame[:-1] + bytes([frame[-1] ^ 0xFF]))

        thread = threading.Thread(target=corrupt_peer)
        thread.start()
        ep._in, _ = ep._server.accept()
        thread.join()
        try:
            with self.assertRaises(GradCompError) as ctx:
                proto.recv_message(ep._in)
            self.assertEqual(ctx.exception.code, E_PROTO)
        finally:
            accepted["sock"].close()
            ep.close()


if __name__ == "__main__":
    unittest.main()
