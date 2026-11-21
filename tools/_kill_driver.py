#!/usr/bin/env python3
"""Driver used by recovery tests to terminate a run at a chosen batch index."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from render_workbench.runner import run_workflow

MODE = sys.argv[1]
KILL_BATCH = int(sys.argv[2])
WORKDIR = sys.argv[3]
CONFIG = sys.argv[4]
PROMPTS = sys.argv[5]
PLAN = sys.argv[6]


def hard_gate(batch_index, pipeline):
    if batch_index == KILL_BATCH:
        os._exit(137)


def graceful_gate(batch_index, pipeline):
    if batch_index == KILL_BATCH:
        import signal
        os.kill(os.getpid(), signal.SIGTERM)


gate = hard_gate if MODE == "hard" else graceful_gate
report, code = run_workflow(CONFIG, PROMPTS, PLAN, WORKDIR, stop_gate=gate)
print("finished", report["status"], code)
