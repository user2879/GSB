import asyncio
import socket
import unittest

from configlease.errors import (
    HandshakeTimeoutError,
    NodeStartError,
    PortInUseError,
)
from configlease.platform import NodeProcess, NodeSpec


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class ProcessOrchestrationTests(unittest.TestCase):
    def test_port_in_use_reports_E_PORT(self):
        port = free_port()
        holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        holder.bind(("127.0.0.1", port))
        holder.listen(1)
        try:
            node = NodeProcess(NodeSpec("n-port", port))

            async def run():
                with self.assertRaises(PortInUseError) as ctx:
                    await node.start()
                self.assertEqual(ctx.exception.code, "E_PORT")

            asyncio.run(run())
        finally:
            holder.close()

    def test_bad_python_reports_E_START(self):
        node = NodeProcess(NodeSpec("n-start", free_port()),
                           python_exe="definitely-not-a-real-python-xyz")

        async def run():
            with self.assertRaises(NodeStartError) as ctx:
                await node.start()
            self.assertEqual(ctx.exception.code, "E_START")

        asyncio.run(run())

    def test_handshake_timeout_is_a_real_socket_timeout(self):
        port = free_port()
        # A real listening peer that accepts the connection but never sends the
        # ready frame, so the timeout comes from the socket read deadline.
        async def scenario():
            server = await asyncio.start_server(
                lambda r, w: asyncio.sleep(10), "127.0.0.1", port,
            )
            node = NodeProcess(NodeSpec("n-hand", port))
            # Bypass child spawn: point the client straight at the silent peer.
            node.bound_port = port
            node.proc = None
            with self.assertRaises(HandshakeTimeoutError) as ctx:
                await node._handshake(timeout=0.4)
            self.assertEqual(ctx.exception.code, "E_HANDSHAKE")
            server.close()
            await server.wait_closed()

        asyncio.run(scenario())

    def test_successful_node_roundtrip(self):
        port = free_port()
        node = NodeProcess(NodeSpec("n-ok", port))

        async def scenario():
            info = await node.start(handshake_timeout=5)
            self.assertEqual(info["port"], port)
            pong = await node.request({"type": "ping"})
            self.assertEqual(pong["type"], "pong")
            await node.kill()

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
