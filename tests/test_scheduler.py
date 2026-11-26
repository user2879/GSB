import time

from cacheflow.scheduler import Job, Scheduler


def test_scheduler_enforces_global_and_node_limits():
    scheduler = Scheduler(global_limit=2, node_limits={"node-a": 1, "node-b": 2})
    state = {"active": 0, "max_global": 0, "node_active": {"node-a": 0, "node-b": 0}, "max_a": 0}
    lock = __import__("threading").Lock()

    def work(context):
        global_inflight, nodes = context.inflight()
        assert global_inflight <= 2
        assert nodes.get("node-a", 0) <= 1
        assert nodes.get("node-b", 0) <= 2
        with lock:
            state["active"] += 1
            state["max_global"] = max(state["max_global"], state["active"])
            node = context.node_id
            if node:
                state["node_active"][node] += 1
                if node == "node-a":
                    state["max_a"] = max(state["max_a"], state["node_active"][node])
        time.sleep(0.04)
        with lock:
            state["active"] -= 1
            if context.node_id:
                state["node_active"][context.node_id] -= 1
        return context.seq

    jobs = []
    for seq in range(8):
        node = "node-a" if seq % 2 == 0 else "node-b"
        jobs.append(Job(seq=seq, node_id=node, fn=work))
    results = scheduler.run_all(jobs)
    assert len(results) == 8
    assert all(result.ok for result in results.values())
    assert state["max_global"] == 2
    assert state["max_a"] == 1
    assert scheduler.observed_violations == []
    records = scheduler.ordered_wait_records()
    assert [row["seq"] for row in records] == list(range(8))
    assert all(row["wait_ended_ms"] >= row["wait_started_ms"] for row in records)


def test_exclusive_barrier_waits_for_inflight_to_drain():
    scheduler = Scheduler(global_limit=3, node_limits={"node-a": 3})
    order = []
    lock = __import__("threading").Lock()

    def record(name):
        def fn(_ctx):
            with lock:
                order.append(("start", name))
            time.sleep(0.05)
            with lock:
                order.append(("end", name))
            return name

        return fn

    jobs = [
        Job(seq=1, node_id="node-a", fn=record("early")),
        Job(seq=2, exclusive=True, fn=record("barrier")),
        Job(seq=3, node_id="node-a", fn=record("after")),
    ]
    scheduler.run_all(jobs)
    starts = [name for kind, name in order if kind == "start"]
    assert starts.index("barrier") > starts.index("early")
    assert starts.index("after") > starts.index("barrier")
