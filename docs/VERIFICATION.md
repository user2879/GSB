# Verification Record

Date: 2026-11-20 (Asia/Shanghai)  
Workspace: `/home/zsl/feedback/uidfc5909d3-default-282`

## Environment

```text
python=3.11.0rc1
dnslib=0.9.26
pytest=9.2.0
tkinter=unavailable ModuleNotFoundError: No module named 'tkinter'
```

The project virtual environment is `.venv`. Runtime code has no third-party dependency; `dnslib` is used
only for test cross-checking. The build container does not contain Tk, so the GUI module was compiled and
imported headlessly, but an actual Tk window could not be launched here. The Windows documentation and
`scripts/start_gui.bat` target the official Python installer, which includes Tcl/Tk.

## 1. Compilation

Command:

```bash
PYTHONPATH=. .venv/bin/python -m compileall -q dns_local run_demo.py tools/make_fixtures.py
```

Result: `COMPILE_OK`. `python -m dns_local.cli --help`, `python -m dns_local.upstream --help` and
`python -m dns_local.gui --help` also exit successfully.

## 2. Full automated test suite

Command:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q
```

Result:

```text
................................................                         [100%]
50 passed in 14.06s
```

Coverage groups:

| Area | Representative tests |
|---|---|
| Wire + dnslib cross-check | query/response round trips, compressed response, pointer loop/forward/OOB, TCP prefix, TC |
| Zone | content-digest versioning, mtime ignored, invalid record index/field, all-or-nothing load |
| Names | case folding, wildcard synthesis/priority, CNAME attribution, loop/hop limit, NODATA SOA |
| Cache | monotonic decrement/expiry, LRU tie-break, single-flight, stale-fill discard, atomic replace |
| Upstream | valid chain, bad transaction ID/question/chain, unreachable, SOA serial v1→v2 |
| Network | real loopback UDP/TCP queries, UDP TC, TCP full answer, E_PORT preflight |
| Concurrency | 8 concurrent same-key queries produce one upstream fetch/source serial |
| Snapshot | read-only revisions, mismatch detection, full replacement, old revision preserved |
| Reproducibility | demo subprocess runs twice and compares stdout bytes |

## 3. Deterministic demo

Commands:

```bash
PYTHONPATH=. .venv/bin/python -m dns_local.demo > /tmp/verify-demo-a.txt 2>/tmp/verify-demo-a.err
PYTHONPATH=. .venv/bin/python -m dns_local.demo > /tmp/verify-demo-b.txt 2>/tmp/verify-demo-b.err
diff -u /tmp/verify-demo-a.txt /tmp/verify-demo-b.txt
```

Results:

```text
demo stdout lines=34
diff_bytes=0
observed run A=3172.469 ms (stderr)
observed run B=3171.605 ms (stderr)
FINAL scenarios=3 queries=18 codes=OK:18 cache=HIT:5,MISS:13,ERROR:0 snapshots_saved=2 snapshots_restored=1
```

The full stdout is stored verbatim in `docs/demo-output.txt`. Important observed lines:

```text
QUERY scenario=cold-start-local ... cache=HIT source=ZONE serial=1 ttl=120 ...
QUERY scenario=upstream-version-change ... cache=MISS source=UPSTREAM serial=1 ttl=45 ...
UPSTREAM_SWITCH previous=1 current=2 invalidated=1
QUERY scenario=upstream-version-change ... name=conflict.upstream.test. ... basis=local-authority-wins ...
SNAPSHOT_COMPARE id=0001-demo status=MISMATCH snapshot_zone=1 current_zone=2 ... differences=12
SNAPSHOT_RESTORE id=0001-demo status=MISMATCH mode=whole entries=8 zone_version=1
CACHE_AFTER_RESTORE entries=8 contains_v2_v6=false ...
QUERY scenario=snapshot-save-restore ... name=www.lab.test. ... cache=HIT source=ZONE serial=1 ttl=117 ...
```

This verifies:

- local cache hit with source/sequence/remaining TTL;
- upstream v1 entry, then whole invalidation when SOA serial becomes v2;
- local authority overriding conflicting upstream RDATA;
- v1 and v2 named snapshots coexist as immutable revisions;
- restoring v1 into a v2 service replaces zone/cache wholesale (`mode=whole`), removes v2-only `v6` entry,
  and restores the frozen TTL/source serial.

## 4. End-to-end CLI smoke test

An upstream simulator was started on loopback port 16354, then the CLI started both DNS listeners on
127.0.0.1:16353 and sent a real UDP query:

```text
QUERY seq=1 id=0x6000 transport=udp name=api.upstream.test. type=A rcode=NOERROR code=OK cache=MISS source=UPSTREAM serial=1 ttl=45
NAME                  TYPE   TTL  RDATA
api.upstream.test.    CNAME  45   node1.upstream.test.
node1.upstream.test.  A      45   198.51.100.11
AUTHORITY       TYPE  TTL  RDATA
upstream.test.  SOA   60   ns1.upstream.test. hostmaster.upstream.test. 1 30 60 300 30
ATTRIBUTION cname=api.upstream.test. target=node1.upstream.test. hop=1; terminal=node1.upstream.test. type=A; upstream_version=1
UPSTREAM_READY host=127.0.0.1 port=16354 version=1 mode=valid
```

The response was TCP-forwarded to the loopback upstream because the name was outside the `lab.test.`
authority. Transaction ID, question echo, CNAME chain and SOA serial were validated before caching.

## 5. Wire fixture integrity

`data/wire/manifest.json` records byte length and SHA-256 for all eight samples. Tests verify both digest and
the expected `E_WIRE` offset for:

```text
pointer-loop.dns            offset=14
pointer-forward.dns         offset=12
pointer-out-of-bounds.dns   offset=14
bad-label-length.dns        offset=12
truncated-header.dns        offset=0
```

## 6. Known environment limitation

The current Linux build image lacks the optional Tk runtime (`_tkinter` and a display). The Tk application
therefore has compile/import coverage only here. Actual rendering must be performed on Windows using the
official Python 3.11+ installer or on Linux after installing `python3-tk` and an X/Wayland display. No GUI
claim in this record relies on an unexecuted window.
