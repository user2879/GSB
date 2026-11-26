# Verification Record

Execution date: 2026-11-27 (Asia/Shanghai). Worktree: `/home/zsl/feedback/uidfc5909d3-default-404`.

## Environment

```text
Python 3.11.0rc1
NumPy 1.26.4
Tkinter 8.6
Linux WSL2 x86_64
```

No network service, container, GPU, or external dataset is used.

## Full test suite

Command:

```bash
python3 -m unittest discover -s tests -v
```

Result:

```text
Ran 24 tests in 11.036s

OK
```

The suite covers validation rejection/partial retention, rule branches and rollback, longest matching and suppression, fixed-point aggregation and boundary tie, quantization flip attribution, slot conflict resolution, three incremental edit classes, serial/concurrent equality, all three scheduler constraints, pending priority, SIGKILL recovery, snapshot integrity, sample end-to-end behavior, and CLI execution.

## Deterministic sample generation

Command:

```bash
python3 scripts/generate_samples.py
```

Actual output:

```text
generated seed=20261127 queries=24 into /home/zsl/feedback/uidfc5909d3-default-404/input
```

## Validation

Command:

```bash
python3 -m intentlab validate
```

Actual output:

```json
{
  "active_revision_ids": ["r07"],
  "issues": [],
  "queries": 24,
  "revisions": 8
}
```

## First pipeline run

Command:

```bash
python3 -m intentlab run
```

Actual output:

```text
processed=24/24 pending=0 segments=14 boundary=1 reuse=0 codes={}
snapshot=gen-20261126T222058Z-298014-a57171f965 committed=True
```

Observed report facts:

- All 24 fixed-point intents equal the generated ground-truth labels.
- `q018` has `score == threshold == 9830400`, `margin == 0`, `boundary == true`.
- Segment `q016,q017,q018` records one offset-mapping conflict: accepted `q018:poi [13,16)` wins over rejected `q018:wide_fragment [13,16)` by the fixed priority.
- Effective rule view has active tip `r07`, nine trigger entries, a stable rules cache key, and complete provenance.

## Second pipeline run (reuse)

Command:

```bash
python3 -m intentlab run
```

Actual output:

```text
processed=24/24 pending=0 segments=14 boundary=1 reuse=62 codes={}
snapshot=gen-20261126T222058Z-298034-acf6e01f3b committed=True
```

Reuse split:

```text
match=24 score=24 merge=14
```

The reuse ledger records query fingerprint equality, unchanged dependencies, source generation, and artifact cache keys. There are no invalidation events on a no-op rerun.

## Recovery inspection

Command:

```bash
python3 -m intentlab recover
```

Actual output:

```json
{
  "committed_generation": "gen-20261126T222058Z-298034-acf6e01f3b",
  "ignored_staging": [],
  "policy": "only the generation referenced by current.json is loadable",
  "resumed_from_committed_snapshot": true,
  "unreferenced_generations": [
    {
      "complete_manifest": true,
      "completed_stages": ["rules", "match", "score", "merge"],
      "generation": "gen-20261126T222058Z-298014-a57171f965",
      "loadable": false
    }
  ]
}
```

## Incremental edit evidence

Automated tests use temporary copies and compare against clean full runs.

| edit | observed invalidation |
|---|---|
| replace `q010` query content | only `q010` match+score recomputed; other 23 match/score artifacts reused; all other intents unchanged |
| update `t_nearby_coffee` weight via new revision `r08` | all 24 matches reused; scores reused 21; only `q004/q016/q018` recomputed; `r08` records 3 score invalidations |
| change fraction bits/quantum from `12/16` to `16/32` | all 24 matches reused; all 24 scores invalidated under `score-config-key`; results equal a clean full run |

## Concurrency and pending evidence

- One-worker and eight-worker runs produce identical query conclusions, scores, cache keys, segment IDs, and segment content.
- With zero match budget, all 24 tasks are persisted with `E_PENDING_MATCH_BUDGET`.
- After increasing budget but retaining a five-query batch cap, five persisted pending items are selected while a newly added ordinary query is not selected, proving pending-first scheduling.

## Forced termination evidence

The test launches separate processes with:

```text
INTENTLAB_CRASH_AFTER_STAGE=rules
INTENTLAB_CRASH_AFTER_STAGE=score
```

and sends `SIGKILL`. After both kills, `snapshots/current.json` is byte-for-byte unchanged and still points to the committed generation. The next run reports both abandoned staging directories, loads only the committed generation, and reuses all 24 matches and all 24 scores. A separate tampering test changes a committed artifact and confirms `E_SNAPSHOT_INTEGRITY` is raised.
