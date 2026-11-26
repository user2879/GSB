# Design notes

## 1. Process and protocol boundary

`ClusterRuntime` launches one `gsb.node_worker` subprocess per declared node.
The child binds the declared literal loopback address and prints one startup
control line to stdout:

- `GSB_READY {...}` after a successful bind/listen;
- `GSB_ERROR {"code": ...}` if binding fails.

All request/reply traffic is then real TCP on `127.0.0.1` using UTF-8 JSON
lines. The parent uses `socket.create_connection(..., timeout=...)` and a
deadline loop around `recv()`; the child uses the same line channel. This gives
real connect, handshake and read deadlines rather than mocked sockets.

The handshake reply contains node id, protocol version and node fixed scale.
Each aggregate reply contains node id, shard id, capacity version, request/batch
identity, integer group sums/counts, batch aggregate digest and integer
monotonic elapsed milliseconds. Missing reply fields become `E_PROTO`; bind
failure becomes `E_PORT`; elapsed socket deadlines become `E_TIMEOUT`.

## 2. Deterministic sample and aggregation rules

Sample generation uses `numpy.random.PCG64` seeded from:

```text
[node.seed, sha256(shard.id)[0:8], sha256(shard.id)[8:16],
 sample_count, group_count, value_width]
```

Rows are generated once from that sequence. A row has group
`row_index % group_count` and three signed integer values in
`[-10000, 10000]`. The canonical JSON bytes of all rows are the sample bytes.

Each request batch is aggregated by `gsb.node_worker` with NumPy `int64`
arrays: raw values are multiplied by the node's declared scale, `np.add.at`
groups sums, and counts are accumulated with integer arrays. The parent merges
batch group sums/counts. Batch size, network concurrency and independent shard
dispatch order therefore cannot affect the final bytes.

The final aggregate digest binds:

```text
node_id, shard_id, sample_row_count, capacity_version, participating_rows,
node fixed_scale, group sums/counts, sample_digest, algorithm/schema names
```

It is SHA-256 over canonical JSON. A stage digest in each node reply covers the
same node/shard/version plus that batch's groups.

## 3. Watermark arithmetic and error decomposition

All amounts are non-negative integer resource units. For session scale `S`:

```text
quota_fp = min(I64, Q * S)
used_fp  = min(I64, U * S)
lease_fp = min(I64, L * S)
util_fp  = floor(U * S / Q)
head_fp  = floor((Q - U) * S / Q)
```

The intermediate multiplication is done with Python unlimited integers. Only a
final quotient greater than `MAX_I64` saturates. The remainder of each ratio is
retained in the watermark object for audit.

For `n` nodes, one node contributes at most two truncated ratios (utilization
and headroom), so its quantization contribution is below `2` fixed units. A
comparison of two candidate utilization values adds at most two truncation
errors, hence the comparison bound is `2` fixed units. Summing `n` node
watermarks contributes at most `n` fixed units. The implementation reports the
three contributions separately and chooses the maximum with a documented tie
priority. This lets operators see whether truncation, comparison or aggregation
dominates.

Resource budget residuals are exact integer resource units. The default budget
bound is therefore `0`: an exact residual tie is flagged and resolved by stable
topological rank then shard id. A caller can supply a positive unit error bound
only when its conversion to fixed units does not exceed the declared total
fixed error bound.

## 4. Stable arbitration

Water arbitration compares only `utilization_fp` integers. If the absolute
difference is at most the comparison bound, the outcome is marked
`error_affected` and chosen by:

1. stable topological rank;
2. shard id.

Outside the bound, lower utilization wins. The output includes both full water
vectors, the utilization/headroom differences, the topological keys, error
model, basis list and decision source. Budget arbitration similarly reports
each residual, residual difference, bound, candidate vectors and basis.

## 5. Invalidation graph

`plan_invalidation(old, new)` identifies direct causes:

- capacity version, global fixed scale, budget or scheduler changes: all shards;
- node seed or node fixed scale: shards on the node plus descendants;
- node quota/used/lease/concurrency changes: shards on the node plus descendants;
- shard demand, node reference, sample/group/batch declaration or dependency
  changes: that shard plus descendants;
- new shards: direct; removed shards are reported separately.

Direct causes are retained in each step. A stable topological pass adds every
shard having an invalid upstream and records `upstream_invalid:<id>` evidence.
The preserved set is the new topological order minus the invalid closure.

Incremental rerun imports preserved records into the new session ledger, runs
only the invalid closure, merges by topological order, and compares canonical
record bytes for every preserved shard. A separate blank full rebuild is
compared field by field. Timing-only fields are not in the semantic comparison
set; aggregate digest, sample digest, lease attribution, units, water, error
attribution, participation and progress cursor are compared. The first mismatch
raises `E_DIVERGE` with shard and field.

## 6. Scheduler and leases

The scheduler keeps one `LeaseManager` per run. Before dispatch it checks all
gates atomically:

- sum of active resource units plus new demand <= global budget;
- node active resource units plus demand <= node quota;
- node active lease count < node lease limit;
- node active shard count < node concurrency;
- free worker-pool slot.

A lease id is deterministic from capacity version, node id and shard id. The
worker invokes `renew()` before the TCP handshake and before each batch. Renewal
uses `time.monotonic_ns()` and `lease_ttl_ms`. Expiry transitions a lease to
`expired`; if its worker is still active, accounting stays quarantined until
that worker exits, then capacity is released exactly once. Quota-change
reclamation changes the same accounting flag, so an expiry/release/quota
transition cannot assign the same capacity twice.

Ready-but-blocked shards get a wait record with monotonic start/end milliseconds
and sorted reason codes (`global_budget`, `node_water_quota`,
`node_lease_limit`, `node_concurrency`, `worker_pool`). A shard can only be
dispatched after its lease and SQLite reservation are both committed.

## 7. SQLite state machine

Tables:

- `meta`: integer format/session versions and current state digest;
- `records`: one row per complete shard outcome;
- `leases`: issue/deadline/status/close reason;
- `uncommitted`: active dispatch reservations;
- `unknown_shards`: crash-recovered uncommitted shard markers;
- `recovery_events`: durable recovery reports.

`reserve_lease` commits a lease and uncommitted marker together.
`complete_shard` verifies the record content digest, writes the record, marks
the lease released, removes the uncommitted marker and updates state digest in a
single `BEGIN IMMEDIATE` transaction. A forced kill therefore leaves either a
complete outcome or an uncommitted reservation, never a half-written record.

On reopen, committed rows are loaded unchanged. Each uncommitted row is marked
unknown, requires rerun, has its units reclaimed, and has its lease marked
abandoned. The recovery report explicitly states that no forced-termination exit
report is available.

Snapshot creation copies a consistent SQLite database and stores a state digest.
Inspection verifies SQLite integrity, integer format version and every record
digest. Restoration validates the snapshot, copies it to a temporary file and
atomically replaces the target database. Existing target rows are never merged;
version/digest mismatches are reported and the whole snapshot wins.

