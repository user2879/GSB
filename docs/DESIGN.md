# Technical Design

## 1. Stage Graph

```text
config + data description + overrides + injections
        |
        v
  deterministic table + per-column fingerprints
        |
        v
  per-candidate preprocessing cache entry
        |
        v
  per-candidate cluster cache entry (allocation-specific)
        |
        v
  per-candidate evaluation cache entry
        |
        v
  diagnostics summary + ranking + budget ledger
```

Every arrow is a content-addressed edge. A stage can only be reused when every upstream fingerprint and all relevant parameters match.

## 2. Input Isolation

Configuration normalization returns a safe runtime object plus a list of issues. Invalid scheduler values use non-crashing fallbacks while retaining the original invalid values in the command-state raw configuration, so a later diagnostic run reports the same business codes. Candidate errors are attached to that candidate only. Data-generation errors that make a referenced column unavailable become `E_COLUMN_DESCRIPTION_MISSING` on the dependent candidate.

Injection processing has two phases:

1. Data-level transformations (remove a description, write a non-finite value).
2. Candidate-level transformations (invalid hyperparameters, deliberate delay).

Both are included in fingerprints or candidate execution keys, so they cannot accidentally reuse a healthy artifact.

## 3. Deterministic Data and Preprocessing

Column RNG entropy is derived from a stable hash of `(seed, column name, distribution, categories)`. Each column has separate value and missing-mask streams. Cross-column redundancy reads only previously generated base values.

The preprocessing plan chooses a training partition from a stable permutation. Fit operations use only these indices. Numeric and encoded-categorical missing values are filled with the median of observed training values. Standardization uses explicitly ordered scalar accumulation:

```text
sum(x_i) -> mean
sum((x_i - mean)^2) -> variance
```

Quantile knots are unique training values. Their rank is the configured tie policy (`average_rank` by default). Transformation is piecewise linear between knots and clamps outside the training range. The metadata records the policy both globally and per feature.

The matrix fingerprint is SHA-256 over dtype/shape and C-order `<f8` bytes. The content fingerprint includes selected column description hashes, selected content hashes, preprocessing parameters, training indices, matrix hash and statistical metadata.

## 4. Algorithms

### KMeans

Initialization:

1. first centroid is the lexicographically smallest row;
2. each next centroid is the unselected row with maximum minimum squared distance;
3. equal distances choose the lower row index.

Assignment ties choose the lower cluster index. Empty clusters retain their previous centroid and are reported at evaluation. Each round appends inertia before centroid updates.

### DBSCAN

The pairwise distance matrix is computed once. Neighbor lists are ascending. Outer traversal is ascending. A FIFO queue expands core neighborhoods, and border/noise assignment is deterministic because neighbors and traversal order are fixed.

### Gaussian Mixture

Means start at deterministic farthest points. The first covariance is the regularized global diagonal covariance. EM performs:

```text
log probabilities -> log-sum-exp normalization -> responsibilities
-> ordered weighted means/covariances -> configured regularization
```

The convergence threshold is `abs(delta_loglik) <= tolerance * max(1, abs(previous_loglik))`. Full/tied covariance precision is the actual inverse covariance; a numerically singular covariance is regularized and produces `E_COVARIANCE_SINGULAR` as a warning.

## 5. Metrics and Ranking

- Silhouette and CH use non-noise clustered samples and report coverage.
- DBSCAN noise remains an explicit component in the density-likelihood metric, so every sample is covered.
- Empty KMeans/GMM clusters and all-noise DBSCAN results remain in the report, with warning codes and all metric entries marked non-comparable.
- Ranking tuple: comparability, silhouette desc, CH desc, density log-likelihood desc, method name, canonical parameter JSON, candidate ID.

## 6. Scheduler and Ledger

Candidates are sorted by stable ID. The scheduler computes a full requested iteration count (DBSCAN requests one pass). It reserves `min(request, quota)` if the global budget can cover it, otherwise marks a global-budget pending entry. Actual consumption can be lower on early convergence.

The ledger is a logical, deterministic accounting artifact, not a wall-clock trace. Cache hits reproduce the same consumed iteration count from cached metadata. This is required for incremental/full-rebuild field equality.

## 7. Interactive Execution

`Workbench` owns one pipeline executor (serialized state builds) and uses a per-stage thread pool sized by configuration. Commands immediately update state and schedule a build. The caller waits only for the remaining interaction budget. A newer state never allows an older build to overwrite the latest completed report; every build is tied to a generation number.

For a view or configuration-only command that does not change the compute fingerprint, the workbench may wait on an already-running build for the same fingerprint. This still respects the latency deadline and avoids returning stale data needlessly.

## 8. Snapshot Protocol

```text
write temp/state.json, temp/report.json, temp/manifest.json
fsync files
rename temp -> versions/vNNNNNN-<snapshot-id>
atomically replace snapshots/CURRENT -> directory + manifest hash
```

Readers only trust a snapshot after verifying the pointer schema, manifest hash and all listed file hashes. A corrupt current pointer falls back to the newest complete older version. An incompatible schema is reported explicitly.

## 9. Equivalence Definition

The CLI canonicalization excludes only non-semantic provenance:

- interaction latency and response timestamps;
- cache-hit/recompute provenance;
- cache directory statistics.

It includes data fingerprints, preprocessing metadata/statistics, labels, cluster traces, metrics, ranking, budget entries, failure ordering, warnings, injection report and diagnostics content fingerprint.
