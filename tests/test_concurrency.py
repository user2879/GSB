import asyncio
import unittest

from configlease.engine import Budget, KeyConflictSerializer


class BudgetTests(unittest.TestCase):
    def test_fifo_queue_and_release(self):
        async def scenario():
            budget = Budget("g", "global", 1)
            order = []

            async def worker(name):
                await budget.acquire()
                order.append(f"in-{name}")
                await asyncio.sleep(0.02)
                order.append(f"out-{name}")
                await budget.refund()

            await asyncio.gather(worker("a"), worker("b"), worker("c"))
            self.assertEqual(order, ["in-a", "out-a", "in-b", "out-b",
                                     "in-c", "out-c"])
            self.assertEqual(budget.consumed, 0)

        asyncio.run(scenario())

    def test_set_limit_releases_waiter(self):
        async def scenario():
            budget = Budget("n", "node", 0)

            async def worker():
                await asyncio.wait_for(budget.acquire(), timeout=2)
                return "ok"

            task = asyncio.create_task(worker())
            await asyncio.sleep(0.05)
            self.assertFalse(task.done())
            await budget.set_limit(1)
            self.assertEqual(await task, "ok")

        asyncio.run(scenario())


class SerializerTests(unittest.TestCase):
    def test_stable_order_on_same_key(self):
        async def scenario():
            ser = KeyConflictSerializer()
            # Deterministically populate the waiter set for one key, then assert
            # the admission predicate picks the minimum (node_id, rev_seq).
            tokens = [("node-3", 3), ("node-1", 1), ("node-2", 2)]
            for token in tokens:
                ser._ensure("k")["waiters"].add(token)
            winner = min(tokens, key=lambda item: (item[0], item[1]))
            self.assertEqual(winner, ("node-1", 1))
            self.assertTrue(ser._can_enter(winner, ["k"]))
            for token in tokens:
                if token != winner:
                    self.assertFalse(ser._can_enter(token, ["k"]))
            # A current holder blocks everyone until it releases.
            ser._ensure("k")["holder"] = ("node-9", 9)
            for token in tokens:
                self.assertFalse(ser._can_enter(token, ["k"]))
            ser._ensure("k")["holder"] = None
            # End-to-end: a release notifies the next minimum waiter.
            order = []

            async def worker(token):
                await ser.claim(token[0], token[1], {"k"})
                order.append(token)
                await asyncio.sleep(0.005)
                await ser.release(token[0], token[1], {"k"})

            ser._keys.clear()
            await asyncio.gather(*(worker(t) for t in
                                   sorted(tokens, key=lambda x: (x[0], x[1]))))
            self.assertEqual(order, [("node-1", 1), ("node-2", 2),
                                     ("node-3", 3)])

        asyncio.run(scenario())

    def test_different_keys_do_not_block(self):
        async def scenario():
            ser = KeyConflictSerializer()
            held = asyncio.Event()

            async def holder():
                await ser.claim("n1", 1, {"a"})
                held.set()
                await asyncio.sleep(0.1)
                await ser.release("n1", 1, {"a"})

            async def other():
                await held.wait()
                await asyncio.wait_for(ser.claim("n2", 1, {"b"}), timeout=0.2)
                await ser.release("n2", 1, {"b"})
                return "parallel"

            result = await asyncio.gather(holder(), other())
            self.assertIn("parallel", result)

        asyncio.run(scenario())

    def test_multi_key_atomic_claim_no_deadlock(self):
        async def scenario():
            ser = KeyConflictSerializer()

            async def worker(node, rev, keys):
                await asyncio.wait_for(ser.claim(node, rev, set(keys)), timeout=2)
                await asyncio.sleep(0.005)
                await ser.release(node, rev, set(keys))

            tasks = [
                worker("n1", 1, ["a", "shared"]),
                worker("n2", 2, ["shared", "b"]),
                worker("n3", 3, ["b", "c"]),
                worker("n1", 4, ["c", "a"]),
            ]
            await asyncio.gather(*tasks)  # must complete, never deadlock

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
