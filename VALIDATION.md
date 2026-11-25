# Validation Record

Environment: WSL Ubuntu 22.04, Python `3.11.0rc1`, numpy `1.26.4`, pytest
`8.2.0`, loopback ports in the high range, no containers and no network services.
Recorded on 2026-11-26 (Asia/Shanghai).

## Automated Tests

Command:

Command:

```bash
.venv/bin/python -m pytest -q
```

Result:

```text
..........................................                               [100%]
45 passed in 12.45s
```

The test suite includes real child processes and real TCP sockets. Temporary
ports and temporary SQLite databases are isolated per test.

## End-to-End Demo

Command:

```bash
.venv/bin/python -m shard_diag demo --root .
```

Recorded result: `ok: true`; every entry in `checks` is `true`.

Key observations from `ledger/demo_report.json`:

- 14 deterministic events across 3 physical database children;
- layout A (batch size 3, reverse startup) and layout B (batch size 5, different
  startup order) have the same routing digest
  `sha256:d2b5d18dab743b3bc6587d466ec53e2ed433b201da8ab8e21b7151b64e051f9d`;
- the generated expected-drift sample matches the runtime records exactly;
- parallel in-flight high-water reached 3 while all three quota limits held;
- queue wait records include per-database lease, per-database concurrency, shard
  ownership, and global lease reasons;
- an occupied `127.0.0.1:49311` produced child error `E_PORT`;
- optimistic renewal moved lease version 1 to 2 and stale version 1 returned
  `E_LEASE` without extending expiry;
- real monotonic expiry returned the quota to 0 active leases;
- two reclaim trigger orders produced byte-identical final ownership/quota
  snapshots;
- a `SIGKILL`ed child left an unknown batch intent; recovery resumed after the
  last committed sequence, reclaimed quota, and reported
  `exit_report_available: false`;
- all committed layouts replayed successfully;
- evolving `orders` from route version 2 to 3 invalidated 8 result keys; only
  those 8 event sequences were re-probed, with unaffected drift records and
  unaffected hit-shard responses both byte-identical (record digest and response
  digest matched before/after);
- replacing the event stream performed a whole snapshot replacement: the old 6
  batches disappeared, generation advanced to 2, and 15 edited batches ran and
  replayed successfully.

The detailed JSON evidence is in `ledger/demo_report.json`. SQLite files are
runtime artifacts and can be removed with `./scripts/clean.sh`.
