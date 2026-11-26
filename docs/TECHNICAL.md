# IntentLab Technical Notes

## 1. Scope and invariants

IntentLab is a single-host diagnostic pipeline. It uses no network service, container, GPU, database, or unrestricted target. All timestamps only affect generation names; conclusions, scores, cache keys, segment IDs, pending order, and failure order are content-derived and deterministic.

Core invariants:

1. Valid input fragments remain available after another fragment is rejected.
2. Every emitted intent conclusion references its selected and considered trigger entries plus its source query.
3. Any reused artifact has an explicit dependency ledger and source generation.
4. A query's conclusion does not depend on the order in which worker threads complete.
5. A hard-killed run cannot make staging artifacts loadable; only a complete generation referenced by `current.json` is recoverable.

## 2. Module map

| module | responsibility |
|---|---|
| `intentlab/validation.py` | JSON loading, schemas, enums, graph validation, edit application |
| `intentlab/rules.py` | revision lineage/branch composition and provenance |
| `intentlab/matching.py` | character-span longest matching and suppression ledger |
| `intentlab/scoring.py` | fixed-point scoring, float64 and high-width references |
| `intentlab/merging.py` | adjacent-query merge and offset-space slot alignment |
| `intentlab/diagnostics.py` | four-stage error attribution and flip reasons |
| `intentlab/fingerprints.py` | canonical SHA-256 fingerprints and cache keys |
| `intentlab/cache.py` | dependency-aware incremental reuse decisions |
| `intentlab/scheduler.py` | budget/quota/batch scheduler and bounded concurrency |
| `intentlab/snapshot.py` | staging generations, atomic commit, integrity and recovery |
| `intentlab/pipeline.py` | orchestration, report assembly, rollback-safe commit |
| `intentlab/report.py` | JSON and Markdown report rendering |
| `intentlab/cli.py`, `intentlab/gui.py` | local command line and Tkinter viewer |

## 3. Input contracts

### 3.1 Vocabulary

`vocabulary.json` is either an array of terms or `{"terms": [...]}`. Every query term and trigger term must be a member. Duplicate vocabulary values are collapsed while preserving a sorted canonical order.

### 3.2 Queries

Required query fields:

- `query_id`: unique non-empty string.
- `terms`: non-empty list of controlled-vocabulary strings.
- `session_id`: non-empty string.
- `turn`: non-negative integer.
- `source`: non-empty channel name; controls per-source quota.
- `ground_truth_intent`: label used only for diagnostics.
- `candidates`: non-empty candidate intent list.
- `slots`: optional list of `{name, start, end}`.

Offsets are zero-based term offsets with half-open intervals. Every accepted slot satisfies `0 <= start < end <= len(terms)`. `text` is derived as terms joined by one space. A conflicting supplied `text` produces `E_SCHEMA_INVALID` and the derived value wins.

### 3.3 Rule revisions

Each revision has `revision_id`, `parent_id`, `seq`, `label`, and `ops`. Roots use `parent_id: null`; every other parent must exist. Duplicate IDs, missing parents, and cycles are rejected. The active revision list may contain multiple tips, which is how rollback can retain an independent sibling branch.

Operation forms:

```text
add:    trigger_id + terms + intent_id + weight + optional slot_pattern
update: trigger_id + at least one of terms/intent_id/weight/slot_pattern
delete: trigger_id only
```

Weights must be members of `config.allowed.weights`. Terms must be controlled-vocabulary members.

### 3.4 Configuration

Important enums and controls:

```json
{
  "allowed": {
    "fraction_bits": [8, 12, 16, 20],
    "quantums": [1, 4, 16, 32, 64],
    "weights": [600, 700, 800, 900, 1000, 1100, 1200]
  },
  "fixed_point": {"fraction_bits": 16, "quantum": 32},
  "thresholds": {"default": 120, "intents": {}, "tie_rule": "accept_on_tie"},
  "scheduler": {"workers": 4, "match_budget_points": 100000, "source_quotas": {}, "batch_query_limit": 1000}
}
```

Invalid enum values, thresholds, or scheduler bounds are rejected by business code and replaced by conservative defaults. A report always includes the retained/defaulted values and issue list.

### 3.5 Edit commands

Supported commands are applied in order:

- `replace_query`: target must exist; the replacement keeps the same ID.
- `add_revision`: parent must exist and no cycle may be introduced.
- `set_active_revisions`: all listed revision IDs must exist.
- `set_fixed_point`: fraction bits and quantum must be in the configured enums.
- `set_threshold`: intent ID plus non-negative integer value; `default` targets the global threshold.

An invalid command is rejected without undoing earlier valid commands.

## 4. Revision composition

For every active tip, the engine walks to its root. The union of lineages is sorted by depth, active-tip rank, `seq`, and revision ID. Operations are applied in that order. This has these consequences:

- Selecting only an ancestor excludes every descendant from its old branch.
- Adding an independent sibling tip includes non-conflicting sibling triggers.
- A conflict on the same trigger ID is resolved deterministically by active-tip order.
- Each effective entry records `source_revision`, stable `entry_order`, and provenance containing the previous content fingerprint.
- Deletes remove the entry from the effective view; re-adding later creates a new entry order and provenance chain.

## 5. Fixed-point formulas

Let `B = fraction_bits`, `S = 2^B`, `Q = quantum`, `L = max(1, len(query.terms))`. For each candidate intent in ascending ID order and each matching trigger entry in ascending `entry_order`:

```text
contribution = floor(weight * hit_count * S / L)
raw_score    = sum(contribution)
score        = floor((raw_score + floor(Q/2)) / Q) * Q
threshold    = configured_threshold * S
margin       = score - threshold
boundary     = abs(margin) <= floor(Q/2)
decision     = score >= threshold          # accept_on_tie
decision     = score >  threshold          # reject_on_tie
```

The winning candidate is selected by highest score, then lexicographically smallest intent ID. A winner below threshold yields `fallback_intent`. Exact threshold hits and half-quantum neighborhoods are marked `boundary=true`.

The float64 path uses the same ordered aggregation with `numpy.float64` and no quantization. The high-width path uses `B + reference_extra_bits` with the same integer quantum, giving a finer normalized reference.

## 6. Matching order and suppression

Candidate matches are generated for every exact contiguous trigger-term occurrence. The fixed priority is:

1. Longer character span first (terms joined with one space for character positions).
2. Lexicographically smaller trigger-term tuple.
3. Smaller term start.
4. Trigger ID and entry order as deterministic final tie-breakers.

Greedy acceptance suppresses every later candidate that overlaps an accepted term interval. Suppression entries include the accepted trigger ID and accepted span under `swallowed_by`/`swallowed_by_start`/`swallowed_by_end`.

## 7. Merge and slot offset space

Queries merge only when they share `session_id`, their turn gap is between `1` and `merge.max_turn_gap`, and both have the same non-fallback intent. A merged segment concatenates each query's terms and inserts `merge.term_gap` literal `<gap>` entries, so all aligned offsets remain valid indexes into the segment term space.

Each local slot maps to:

```text
global_start = query_base + local_start
global_end   = query_base + local_end
```

Overlaps are conflicts even when slot names differ. Accepted priority is: earlier query turn, longer local span, query ID lexical order, slot name lexical order. Both accepted and conflicted mappings remain in the ledger; conflicted entries point to their accepted counterpart.

## 8. Diagnostics and flip attribution

Each processed query reports four stage deltas:

1. `trigger_match`: symmetric difference of selected trigger/span signatures.
2. `intent_score`: maximum normalized candidate-score difference from float64.
3. `boundary_decision`: threshold-margin difference, normalized by the threshold scale.
4. `slot_alignment`: symmetric difference of aligned slot mappings around the fixed and reference segments.

`first_exceeding_stage` is the first stage whose normalized delta exceeds `diagnostics.tolerance`. If fixed and reference intents differ, the report contains independent cause flags:

- `quantization_rounding`: raw fixed decision differs from the quantized decision.
- `threshold_comparison`: the flip is not explained by quantization.
- `merge_boundary_movement`: fixed and reference segment membership differ.

The high-width intent, high-width fraction bits, normalized margins, and fixed/float margins are reported together.

## 9. Cache derivation and exact invalidation

All hashes are SHA-256 over canonical sorted JSON.

- Query fingerprint: query identity, terms, context, source, candidates, slots, priority.
- Trigger full fingerprint: ID, terms, intent, weight, slot pattern, entry order, source revision.
- Trigger match fingerprint: ID, terms, entry order.
- Effective rule-view key: active tips, lineage, chain version, all effective trigger full fingerprints.
- Match key: query fingerprint + considered trigger match fingerprints + matching-rule version.
- Score key: query fingerprint + match key + referenced trigger full fingerprints + fraction bits + quantum + tie rule + relevant thresholds.
- Segment key: member score keys + query fingerprints + intent + merge configuration.

On a new snapshot the engine compares old and new effective entries. A weight-only change leaves match keys stable because weights do not affect matching. A query is match-invalidated only if:

- its query content changed;
- one of its considered trigger entries changed matching fields; or
- a changed/new trigger's terms can occur in that query.

A score cache is reusable only when its key recomputes identically from the current match key, referenced entries, quantum, tie rule, and relevant thresholds. Segment caches are reusable only when their complete member score keys and query fingerprints are identical.

The reuse ledger records stage, query/segment ID, cache key, dependency list, source generation, and a human-readable basis. Invalidation counts are grouped by the source revision of the changed trigger; config-only changes use an explicit config bucket such as `score-config-key`, and pure query edits use `query-content`.

## 10. Scheduler constraints

Before dispatch, a query that needs matching reserves:

```text
match_points = len(query.terms) * max(1, number_of_effective_triggers)
```

Cached matches reserve zero. Selection is deterministic and pending-first:

```text
(pending flag, query priority, session ID, turn, query ID)
```

Three limits are checked independently:

- global match-point budget;
- per-source selected-task quota;
- per-batch recompute count.

Deferred tasks are written to `pending.json` with `E_PENDING_MATCH_BUDGET`, `E_PENDING_SOURCE_QUOTA`, or `E_PENDING_BATCH_LIMIT`. The next run marks them pending and schedules them before ordinary work. Thread outcomes are collected into query-ID-ordered dictionaries; failures are sorted by business code and query ID.

## 11. Snapshot protocol

One run writes all stage artifacts into:

```text
snapshots/staging/<generation>/
  state/
  rules/
  matches/
  scores/
  segments/
  results/
  pending.json
  failures.json
  ledger/
  stages.json
```

Stages complete in fixed order: `rules`, `match`, `score`, `merge`. The stager fsyncs file contents and stage indexes. On commit it:

1. Rechecks the current pointer against the generation, global version, state token, and config hash loaded at run start.
2. Rehashes all input files to detect mid-run edits.
3. Writes a complete manifest containing every artifact SHA-256.
4. Moves the whole staging directory to `snapshots/generations/<generation>`.
5. Atomically replaces `snapshots/current.json`.

No committed file is updated in place. Loading verifies the manifest hash and every artifact hash. If the process is killed at any stage, only the old `current.json` generation loads. Staging entries and complete-but-unreferenced generations are surfaced by `intentlab recover` and the report's `recovery` block.

## 12. Business codes

Representative codes:

| code | meaning |
|---|---|
| `E_QUERY_ID_DUPLICATE` | unique query ID violation |
| `E_QUERY_TERM_UNKNOWN` | term outside controlled vocabulary |
| `E_SLOT_RANGE` / `E_SLOT_TYPE` | invalid slot bounds or non-integer offsets |
| `E_PARENT_MISSING` / `E_REVISION_CYCLE` | invalid revision graph |
| `E_TRIGGER_WEIGHT_ENUM` | trigger weight outside configured enum |
| `E_FRACTION_BITS_ENUM` / `E_QUANTUM_ENUM` | fixed-point setting outside enum |
| `E_EDIT_TARGET_MISSING` / `E_EDIT_OP_UNKNOWN` | invalid edit command |
| `E_PENDING_MATCH_BUDGET` / `E_PENDING_SOURCE_QUOTA` / `E_PENDING_BATCH_LIMIT` | persisted deferred work |
| `E_SNAPSHOT_INTEGRITY` | committed artifact or manifest checksum failure |
| `E_SNAPSHOT_CONCURRENT_COMMIT` | version/config/input changed before commit |

All accepted and rejected issues are listed in report order `(code, scope, entity_id, message)`.

## 13. Verification map

| acceptance concern | test |
|---|---|
| invalid queries/slots/terms and retained valid part | `test_validation.py` |
| invalid revision parent/cycle/weight enum | `test_validation.py` |
| rule add/update/delete/rollback/sibling | `test_rules_matching.py` |
| longest/lexical/start matching and suppression | `test_rules_matching.py` |
| fixed aggregation and boundary tie | `test_rules_matching.py` |
| quantization flip and high-width reference | `test_rules_matching.py` |
| slot alignment and conflict priority | `test_rules_matching.py` |
| single query replacement exact reuse | `test_incremental.py` |
| single trigger weight change exact reuse | `test_incremental.py` |
| fixed-point width/quantum change reuse boundary | `test_incremental.py` |
| serial versus concurrent field equality | `test_scheduler_snapshot.py` |
| triple budget and next-batch pending priority | `test_scheduler_snapshot.py` |
| SIGKILL recovery and integrity detection | `test_scheduler_snapshot.py` |
| sample end-to-end and CLI | `test_end_to_end.py` |

## 14. Known boundaries

- Slot offsets are term-index offsets, not UTF-8 byte or display-cell offsets. Character spans in matching are Python Unicode code-point counts after joining terms with spaces.
- The scheduler runs CPU work in local threads as requested by the local config; no subprocess pool is created.
- Snapshot history is retained intentionally for audit. There is no automatic garbage collection in this delivery.
- The Tkinter viewer is optional; all evidence is available in JSON/Markdown without a display.
