# Design Notes

## 1. Boundaries and trust model

The process accepts only IPv4 loopback host literals. Socket helpers reject non-loopback addresses before
name resolution or connection, so an accidental configuration cannot make the resolver contact the public
Internet. Authoritative data, upstream responses, snapshots and wire fixtures are all local artifacts.

The upstream is a separate loopback TCP process only to make transaction IDs, framing and failure modes
observable. It is not a public recursive resolver.

## 2. Wire codec

`dns_local/wire.py` implements the codec without runtime third-party packages.

- All header counts and the question section are parsed before resource records.
- Expanded names are limited to 255 bytes and labels to 63 bytes.
- A compression pointer is accepted only when `target < pointer_position`, never equal to or greater than
  the pointer location. This removes forward references and, together with a visited-target set and a jump
  cap, rejects loops.
- Pointer targets and label/name bounds are checked against the packet length. Failures raise `E_WIRE` with
  the offending absolute offset.
- Known RDATA types are A, AAAA, CNAME, NS and SOA. Unknown types remain raw bytes but cannot participate in
  an authoritative/upstream chain.
- The encoder records only suffixes already emitted at lower buffer offsets. It does not emit a forward
  pointer.
- UDP truncation re-encodes a header/question-only response with TC when the complete response exceeds the
  configured limit. TCP callers add and consume the two-byte length prefix.

Tests cross-check valid packets with `dnslib` but do not use it in runtime code.

## 3. Zone image and versions

JSON schema: `{"origin": "...", "records": [{"name", "type", "ttl", "rdata"}]}`.

Validation is all-or-nothing and reports the zero-based record index plus field. Supported TTLs are integers
from 0 through `2^31-1`; A/AAAA values are canonicalized through `ipaddress`; name RDATA is canonicalized;
SOA has exactly seven text fields. Exactly one apex SOA is required. CNAME owners cannot carry other types.

Each zone load hashes the exact file bytes. The active version increases only when this SHA-256 changes.
Loading identical bytes again—even at a new path or mtime—returns the same image/version. Snapshot restore
also installs an immutable image and advances the manager's next-version counter to avoid reuse.

## 4. Resolution and attribution

- Query names and zone owners are ASCII-lower-cased absolute names.
- Exact owner lookup wins. A wildcard is considered only when the exact owner is absent. The closest existing
  encloser determines the sole wildcard candidate `*.<closest-encloser>`.
- Wildcard answers synthesize the owner as the current query name. They never expose the wildcard owner as an
  answer owner.
- CNAME resolution records each hop, detects visited targets and enforces a configurable maximum (default 8).
  `E_CNAME` includes limit/hops/chain evidence.
- NODATA and NXDOMAIN include the apex SOA. The negative TTL is `min(SOA TTL, SOA MINIMUM)`, capped by the
  active policy.

Every result carries ordered attribution strings for match type, synthesized name, CNAME hops, negative basis
or upstream arbitration.

## 5. Cache invariants

`CacheKey(name, qtype, qclass)` contains the normalized query identity. A `CacheEntry` is immutable and holds
the complete answer/authority sets, original minimum TTL, monotonic insertion time, source
(`ZONE`, `NEGATIVE`, `UPSTREAM`), source serial, attribution and metadata.

One condition lock guards the dictionary. Readers observe either the pre-replacement or post-replacement
dictionary, never a partially built entry. `replace_all` swaps the complete dictionary while holding that
lock. A reader that already obtained an immutable entry may finish serving it; it represents a coherent old
state, not a mixed new/old table.

Same-key fills use a single-flight set. One leader executes the loader outside the cache lock; waiters block
until the leader inserts or fails. A commit guard is evaluated under the lock immediately before insertion.
It rejects a fill when the service state epoch changed or an upstream source serial is older than the current
observed upstream version. The service then recomputes with fresh state or returns `E_STATE`.

TTL uses integer seconds of elapsed time from `Clock.monotonic_ns()`. Expired entries are removed on access,
capacity eviction and enumeration. Eviction order is: expired first, then least recent access, then canonical
key string. The clock source is printed in the startup/status report.

## 6. Upstream validation and arbitration

Forwarding reuses the downstream transaction ID and sends one TCP-framed query to the loopback simulator.
Validation requires:

- response QR bit, standard opcode and exactly one echoed question;
- identical transaction ID and case-insensitive identical question name/type/class;
- supported result code, IN records and bounded TTL;
- an answer sequence beginning at the query name, following only CNAMEs, ending in the requested type (or a
  valid negative answer), without loops, extra owners or unrelated types;
- consistent SOA serials when provenance records are present.

Malformed/unreachable upstreams produce `E_UPSTREAM` or `E_CHAIN` before any cache insertion, so the
downstream cache remains unchanged.

The simulator includes its apex SOA in responses. A higher SOA serial atomically invalidates every
upstream-origin cache entry before the new result is committed. `upstream refresh` probes the apex SOA on
demand; this makes version control explicit rather than issuing a probe on every cache hit.

If a validated upstream chain enters the local authoritative zone, the fixed rule is
`local-authority-wins`: covered upstream records are discarded, the remaining local chain is resolved from
the same owner, and the decision is recorded in attribution. A local CNAME pointing out of the zone is
completed by forwarding the target and merging the two validated chains.

## 7. Snapshots

A snapshot stores:

- format marker, name, revision, creation metadata and clock-source label;
- full zone origin/version/digests/records;
- negative policy and observed upstream version;
- complete cache entries with TTLs frozen at save time and each source/source serial;
- query history and service/cache generations;
- SHA-256 checksum over canonical JSON.

Files are atomically written, then marked read-only. Saving the same name creates a new revision instead of
overwriting the prior file.

Consistency comparison excludes naturally elapsed TTL seconds but includes zone version/digest, upstream
version, negative TTL cap, and each entry's key/source/source serial. Opening a snapshot always performs a
whole-state restore, even when the comparison says `CONSISTENT`; `MISMATCH` is reported but is not a merge
mode. Restore installs the zone, policy, upstream version, cache table and history as one state transition
under the state lock, while the cache table itself is swapped under its own lock. Old snapshots remain
available for listing and diffing.

## 8. Listener and error behavior

UDP and TCP preflight checks require loopback literals. TCP preflight enables address reuse to avoid false
`E_PORT` reports from `TIME_WAIT`; UDP does not enable reuse so another live UDP socket is detected. Binding
failures map to `E_PORT` with host/port/transport details.

Expected query failures still produce a DNS response (FORMERR/NOTIMP/SERVFAIL) and a history row containing
the business code. Wire failures include the absolute offset. Unexpected exceptions are contained at the
network boundary and mapped to `E_STATE`/SERVFAIL.

