"""First recovery phase: commit some batches, then force-kill the process.

The kill happens after a batch intent row is written but before its hits are
committed, simulating an interrupted transaction. No exit report is produced.
"""

import os
import signal
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hotgray.gateway import Gateway


def main():
    declaration_path, db_path = sys.argv[1], sys.argv[2]
    gateway = Gateway(declaration_path, db_path)
    gateway.start()
    stream = gateway.declaration["request_stream"]
    for request in stream[:8]:
        gateway.admit_request(request)
    gateway.flush_all()

    request = stream[8]
    rule = gateway._rule_for(request)
    gateway.admit_request(request)
    key = (rule["rule_id"],
           __import__("hotgray.hashing", fromlist=["route_decision"])
           .route_decision(rule["rule_id"], request["client_key"],
                           rule["gray_weight_bp"]))
    gateway.store.mark_batch_intent("batch-ORPHAN", rule["rule_id"], key[1])
    print("forcing process termination before committing batch-ORPHAN",
          flush=True)
    os.kill(os.getpid(), signal.SIGKILL)


if __name__ == "__main__":
    main()
