# On-disk Formats and Determinism Rules

All integers are little-endian. CRC values use `zlib.crc32(...) & 0xFFFFFFFF`. Page IDs and overflow references are UTF-8 base filenames; path separators are rejected.

## 1. Page file

Magic `PGDATA01`, trailer magic `PGEND001`.

```text
header  = <8s H H I H H>   magic, version=1, column_count, row_count,
                           page_id_length, reserved=0
page_id = bytes[page_id_length]
columns = repeated column descriptor:
          <B H> kind (1=int64, 2=float64), name_length
          name bytes
bitmaps = for each column, ceil(row_count/8) bytes; bit i = row i is null
values  = for each column, row_count fixed 8-byte values
trailer = <8s I> PGEND001, CRC32(all preceding bytes)
```

Null slots store eight zero bytes. Non-finite floats are rejected. A page's manifest `payload_checksum` is CRC32 over the complete page file bytes.

Column summaries used by v2/v3 contain name, kind, null count, and the first 16 bytes of SHA-256 over `kind || name || null_bitmap || raw_value_bytes`.

## 2. Manifest file

Magic `PCMANIF1`; trailer magic `PCMEND01`.

```text
file header = <8s H H I> magic, header_version, flags=0, record_count
record      = <I record_length> record_bytes
file trailer= <8s I> PCMEND01, CRC32(all preceding bytes)
```

`record_length` includes the record CRC but excludes its own four-byte length prefix. `record_bytes`:

```text
common (v1+):
  record_version: u8
  page_id_length: u16
  page_id: UTF-8 bytes
  row_count: u32

v2+:
  summary_count: u16
  repeated:
    name_length: u16
    kind: u8
    null_count: u64
    digest: 16 bytes
    name: UTF-8 bytes
  payload_checksum: u32

v3+:
  compressed: u8 (0/1)
  overflow_ref_length: u16
  overflow_ref: UTF-8 bytes (empty = null)

record_crc: u32 over every preceding byte of this record
```

The v1 domain payload only contains page ID and row count. The record CRC and file trailer CRC are physical-integrity envelope fields, not additional v1 business fields.

Decoder validation is all-or-nothing:

- supported file, header, and record versions;
- record length within the body and exactly consumed by parsed fields;
- UTF-8 and base-filename constraints;
- duplicate page IDs;
- per-record CRC and file trailer CRC;
- footer magic and no trailing bytes.

Failures use `E_FORMAT` with an error path such as
`manifest.records[7].record_length` and `record_index=7` when a record is involved.

## 3. Overflow segment

One file holds one page segment. Magic `PGSEG001`; trailer magic `PGSEND01`.

```text
header = <8s H B B H I I>
         magic, version, flags(bit0=compressed), reserved=0,
         page_id_length, row_count, stored_payload_length
page_id = UTF-8 bytes
v2+: payload_checksum: u32 = CRC32(uncompressed page bytes)
stored_payload = raw page bytes, or zlib.compress(page_bytes, level=9)
trailer = <8s I> PGSEND01, CRC32(all preceding bytes)
```

v1 stores no semantic payload checksum and cannot be marked compressed. v2 adds the semantic payload checksum. v3 adds the compression flag.

Compression mismatch detection covers both directions:

- flag set but stored bytes are not a zlib stream;
- flag clear but stored bytes are a zlib stream whose result starts with `PGDATA01`;
- manifest `compressed` disagrees with the segment header;
- decompressed/plain bytes are not a valid page for the segment page ID.

All such cases raise `E_PAYLOAD` with the first mismatching `page_id`. Physical trailer or payload checksum disagreement raises `E_INTEGRITY`.

## 4. Upgrade algorithm

Paths derived from `manifest.pcm`:

```text
manifest.pcm.pre-upgrade.backup             original bytes, mode read-only
manifest.pcm.upgrade.work                   current-version rewrite target
manifest.pcm.upgrade-checkpoint.json        atomically replaced per record
manifest.pcm.upgrade-report.json            explicit filled-field report
```

1. Decode and validate the complete source manifest before changing anything.
2. Atomically write the read-only backup and a v3 work header.
3. For each record, upgrade in memory, append and `fsync` current-version bytes, then atomically replace the checkpoint with `done_records`, work byte size, SHA-256, CRC32, report tail, and processed indices.
4. A simulated or real interruption may truncate an append that was not acknowledged by a checkpoint. Resume hashes only the acknowledged prefix, truncates any unacknowledged bytes, and continues at `done_records`; completed records are never re-upgraded.
5. Finish by appending the file trailer, validating the complete work file, atomically replacing the manifest, writing the report, and marking the checkpoint `complete`.

Explicit v1/v2 completion rules:

- v1: read the page file, verify row count, derive column summaries and payload checksum; derive v3 fields from an existing `page_id.seg`, otherwise `compressed=false`, `overflow_ref=null`.
- v2: verify checksum and summaries against the page file; if `page_id.seg` exists, infer compression from its header and set the reference, otherwise use the same uncompressed/null default.
- Every filled field is listed in the upgrade report with its value and the overflow inference rule.

## 5. Deterministic cache and spill

```text
capacity_frames = floor(budget_bytes / page_size)
victim = min(resident pages, key=(last_access_clock, page_id))
```

The comparison key is total because page IDs are unique. No set/dict iteration order participates in a decision. A new or swapped-in page gets the next monotonically increasing access clock. Eviction writes a v3 segment using the manifest compression flag and the original page bytes.

The cache state records stable sorted resident/overflow indices, `last_access`, overflow sequence, swap-in sequence/count, load count, eviction count, and full event stream.

## 6. Fixed-point aggregation

```text
version = fixed-v1-half-even-trunc-toward-zero
S       = 1_000_000
quant(x)= round_half_even(Decimal(str(x)) * S)
sum     = integer sum of quantized non-null values in scan order
mean    = truncate_toward_zero(sum, count)
```

Rows with a null group key or null value are skipped. Groups are emitted in strictly increasing integer key order.

Error bounds:

- each value: `|quant(x)/S - x| <= 1/(2S)`;
- group sum: `<= count/(2S)`;
- group mean: `<= 1/(2S) + 1/S = 3/(2S) = 0.0000015`.

The sum bound includes half-even quantization; the extra `1/S` covers truncation of the integer fixed-point mean. The mean bound is independent of group count.

## 7. Snapshot and replay

Snapshot JSON is canonicalized and atomically written. Required top-level state:

- `manifest_format_versions`, `manifest_header_version`;
- `disk_state.manifest_sha256` and full segment inventory with file hashes;
- `cache` resident/overflow indices, LRU clock, counters, event stream;
- `aggregation` intermediate integer groups;
- `fixed_point` version/scale/columns;
- `position`, `scan_order`, full `reference_trace`, and final reference output/hash.

Restore first compares the snapshot manifest versions, header, content fingerprint, and segment hashes/sets with current disk. Regardless of compatibility, restoration replaces both cache and aggregation containers wholesale; it never merges. Lost/new/changed segment names are sorted.

Exact replay rejects an incompatible manifest version/state, fixed-point version change, or missing/changed required segment with `E_REPLAY` and a machine-readable `basis`. Otherwise it continues from `position`, compares each trace field in fixed order, and reports the first `page_id`, `field`, and `access_index` mismatch.
