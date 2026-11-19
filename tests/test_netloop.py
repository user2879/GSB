import unittest

from cache_invalidation.engine import Engine
from cache_invalidation.model import EntryID
from cache_invalidation.netloop import (
    LoopbackClient,
    LoopbackServer,
    find_free_loopback_port,
)
from cache_invalidation.store import Store
from tests.support import sample_graph


class LoopbackTest(unittest.TestCase):
    def test_server_and_client_on_loopback_only(self):
        port = find_free_loopback_port()
        engine = Engine(sample_graph(), store=Store())
        server = LoopbackServer(engine, port=port)
        bound_port = server.start()
        self.assertEqual(bound_port, port)
        try:
            client = LoopbackClient(bound_port)
            self.assertTrue(client.call({"command": "ping"})["pong"])
            response = client.call({
                "command": "invalidate",
                "namespace": "catalog",
                "key": 100,
                "now_ms": 1000,
            })
            self.assertTrue(response["ok"])
            self.assertEqual(
                response["affected"],
                ["catalog/100", "inventory/300", "pricing/200",
                 "storefront/400", "search/500"])
            state = client.call({"command": "state"})
            self.assertEqual(state["format_version"], 3)
            self.assertEqual(
                state["versions"]["catalog/100"], 1)
            self.assertNotIn("catalog/100", state["alive"])
        finally:
            server.stop()
            engine.close()

    def test_business_error_is_returned_as_structured_payload(self):
        engine = Engine(sample_graph(), store=Store())
        server = LoopbackServer(engine)
        port = server.start()
        try:
            client = LoopbackClient(port)
            client.call({
                "command": "invalidate",
                "namespace": "catalog",
                "key": 100,
                "now_ms": 100,
            })
            response = client.call({
                "command": "invalidate",
                "namespace": "catalog",
                "key": 100,
                "now_ms": 99,
            })
            self.assertFalse(response["ok"])
            self.assertEqual(
                response["error"]["code"], "E_CLOCK")
        finally:
            server.stop()
            engine.close()


if __name__ == "__main__":
    unittest.main()
