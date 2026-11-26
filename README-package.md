# shardcast

Deterministic, multi-node shard dispatcher for local WSL Ubuntu demos.

- Node workers run as real subprocesses listening on `127.0.0.1` TCP ports.
- JSON-lines protocol with handshake/read timeouts and strict receipt validation.
- NumPy int64 fixed-point group aggregation from explicit integer seeds.
- Monotonic-clock leases with optimistic version checks and real expiry.
- Whole-replacement capacity views, stale marking, backups and diffs.
- Budget, per-node lease-limit and concurrency constrained scheduling.
- Dependency-aware readiness, freeze/cancel semantics and wait measurement.
- SQLite ledger with atomic per-shard commits and forced-kill replay recovery.
- Incremental invalidation propagation with full-rebuild field parity checks.

See `TECHNICAL_NOTES.md` for the deterministic rules and verification record.
