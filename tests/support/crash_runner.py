"""Helper used by the SIGKILL recovery test.

The first batch runs normally so one snapshot is fully committed; every later
batch sleeps before its work, letting the parent kill the process while a
subsequent snapshot is only partially staged.
"""
import sys
import time

from edge_cache.runner import RunOrchestrator

config_dir = sys.argv[1]
work_dir = sys.argv[2]
sleep_s = float(sys.argv[3])

orchestrator = RunOrchestrator(config_dir, work_dir, "c3")
original = orchestrator._run_one_batch
calls = {"n": 0}


def slow(pool):
    calls["n"] += 1
    if calls["n"] >= 2:
        time.sleep(sleep_s)
    return original(pool)


orchestrator._run_one_batch = slow
orchestrator.run()
