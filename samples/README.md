# Deterministic local samples

The declared services listen only on `127.0.0.1` ports `39171` and `39172`.
Close any local process that already owns those ports before running the
samples.  The event timestamps are logical milliseconds, not wall-clock sleeps;
lease duration is 5000 ms, so an ordinary local run does not trigger the hybrid
real-clock guard.  Replay mode always uses the logical clock.

Covered paths:

- `alpha` capacity 1: grant, renewal, higher-priority takeover, rejected lower
  priority request, logical expiry, new grant, release, new grant.
- `beta` capacity 2: concurrent grants, higher-priority takeover, expiry of two
  unrenewed leases, reclamation and another takeover.
- `clusters_evolved.json` / `holders_evolved.json`: capacity, lease duration and
  priority changes that force a whole-view replacement and re-adjudication.

All files are local, require no network or account, and are suitable for
byte-for-byte deterministic reruns when `--clock-mode logical` is used.

`expected_deterministic.json` fixes the expected 14-step decision sequence,
zero-based watermark facts, two fresh-run digests, and final active holders.
