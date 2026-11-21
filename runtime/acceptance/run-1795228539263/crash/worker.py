
import asyncio, json, sys
from pathlib import Path
sys.path.insert(0, '/home/zsl/feedback/uidfc5909d3-default-30')
from scripts.acceptance import make_platform

async def main():
    workdir = Path(sys.argv[1]); data = Path(sys.argv[2])
    platform = make_platform(workdir, data / "manifest.json", data / "nodes.json",
                             session="crash-victim")
    await platform.start_nodes()
    await platform.dispatch({"node_id": "node-1", "rev_seq": 1})
    await platform.dispatch({"node_id": "node-2", "rev_seq": 2})
    # Lease left active and unsettled when the process group is killed.
    platform.issue_lease("node-3", rev_seq=3, low_seq=3, high_seq=3,
                         keys={"x"}, ttl_ms=120000)
    platform.save_snapshot("session")
    print("READY_TO_CRASH", flush=True)
    await asyncio.sleep(30)

asyncio.run(main())
