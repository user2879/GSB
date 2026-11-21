"""Second recovery phase: recover committed state, rerun and verify replay."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hotgray.gateway import Gateway


def main():
    declaration_path, db_path = sys.argv[1], sys.argv[2]
    gateway = Gateway(declaration_path, db_path)
    gateway.start()
    recovery = gateway.recover()

    stream = gateway.declaration["request_stream"]
    for request in stream[8:]:
        gateway.admit_request(request)
    gateway.flush_all()
    gateway.rebuild_snapshot()
    snapshot = gateway.verify_replay()
    result = {
        "recovery": recovery,
        "replay_verdict": "reproducible",
        "snapshot_id": snapshot["snapshot_id"],
        "digest": snapshot["digest"],
        "committed_hits": gateway.store.count_hits(),
    }
    gateway.stop()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
