# Technical Design

## 1. Deterministic input

`SampleConfig` fixes the seed, point counts, room bounds, ground plane, jitter amplitudes, patch layout and reconstruction defaults. NumPy `PCG64` receives only the configured seed, and the sample is written in a fixed order. JSON is canonicalized with sorted keys and compact separators. No timestamps or process identifiers enter sample files. The manifest contains file hashes, the 120,000-point count, bounding box, confidence histogram/statistics, class counts and a SHA-256 summary of generation parameters.

Binary formats:

```text
points.bin : <8sIIQ> magic="PCSTPTS1", version=1, flags=0, count
             count * float32_le(x,y,z)
normals.bin: <8sIIQ> magic="PCSTNRM1", version=1, flags=1, count
             count * float32_le(nx,ny,nz,confidence)
```

Loader validation checks magic, version, exact payload length, finiteness, unit normals and confidence range. Any violation is `E_POINT_CLOUD_FORMAT` with path and offending index/value context.

## 2. Voxel neighbourhood and normal re-estimation

`VoxelGrid` sorts points by integer cell coordinates and stores contiguous cell ranges. A radius query visits the 3x3x3 surrounding cells and filters by exact distance. The default legal radius range is `[0.05, 2.0]`; other values return `E_NEIGHBOR_RADIUS`.

For every selected point, a confidence-weighted covariance is fitted:

```text
C_i = sum_j w_j (p_j - centroid)(p_j - centroid)^T / sum_j w_j
residual_i = sqrt(lambda_min(C_i))
normal_i = eigenvector(lambda_min), oriented to the old normal
```

The new confidence combines eigen-gap coherence, neighbour support and residual sharpness. The per-point normal angle bound combines fit residual and point quantization. Every selected point persists:

- neighbour radius used;
- number of participating neighbours;
- fit residual;
- normal error angle bound.

Only selected indices are written. All normals and confidence values outside the selected region are copied bit-for-bit, and the engine asserts this before committing.

## 3. Local surface patches and invalidation

A patch key is:

```text
axis sign cell_i cell_j cell_k
```

`axis/sign` comes from the dominant point normal. `i,j` are two in-plane grid cells and `k` isolates parallel surfaces such as the room floor and external ground. This prevents two distinct planes from being merged into one patch.

Within a patch, points are binned on a local triangulation lattice. Four occupied adjacent bins generate two triangles; three generate one. Triangle winding follows the patch orientation. The cache hash covers the patch key, float64 vertices and int32 triangles.

For an edit, root patches are selected by both point membership and distance from the edit region to the patch AABB, expanded by the declared support radius. Same-plane face neighbours sharing `i` or `j` boundaries are added. The final closure list is recorded before rebuilding and is asserted equal to the attempted recomputation list. Every patch outside the closure is reused as the same object and is verified to retain byte-equal vertices.

A global reconstruction parameter change is explicitly reported as `explicit-global-parameter-replacement`; it is the only operation allowed to replace the whole patch set.

### Viewport box selection

`viewport.py` projects points with the same camera matrices used by rendering. A screen rectangle selects exactly the visible projected points in that rectangle; `remove_viewport_box` removes those points and invalidates only their patches plus shared-boundary neighbours. It deliberately does not turn the rectangle into a world-axis AABB, because that would select unrelated points along the camera ray and could inflate the local rebuild into most of the model.

## 4. Error ledger

For each patch, the ledger stores four stages and one composed upper bound:

```text
q_i = ||0.5 * spacing_float32(p_i)||_2
q_patch = max_i q_i
r_patch = sqrt(lambda_min(covariance(points assigned to patch)))
theta = max(initial_normal_bound, atan2(r_patch + q_patch, 0.5 * radius))
n_patch = 2 * radius * sin(theta)
c_patch = max source-point distance from incident-triangle local planes
B_patch = q_patch + n_patch + c_patch
B_network = max_patch B_patch
```

All terms are world-space distances. The ledger includes formula text, layout/radius parameters, point/normal file hashes, and per-patch source-point and content hashes. `verify_ledger` rebuilds every patch from the point/normal files and compares hashes and every numeric field. The CLI `verify-ledger` uses only the two binary files and `ledger.json`.

## 5. Incremental/full consistency

Production reconstruction never uses the comparator result. After an incremental edit, the engine separately rebuilds from the complete edited point set. Both meshes are compared by patch key, vertex count, triangle topology and vertex positions. Two criteria must both pass:

1. maximum vertex deviation <= declared position tolerance;
2. every vertex deviation <= that patch's ledger upper bound plus tolerance.

The result includes maximum/mean/percentile deviation, threshold distribution and first divergent vertex with patch key, local/global index and triggering edit sequence. Failure raises `E_MESH_MISMATCH` and leaves the engine uncommitted.

## 6. Snapshot semantics

A `SceneSnapshot` contains one version stamp, immutable point arrays and one immutable mesh/ledger result. Publication validates array write flags and replaces a single pointer under a lock. A reader either gets the complete old scene or the complete new scene. Requested stale versions return `E_SNAPSHOT_STALE`.

`ThreadedReconstructionService` runs edits and reconstruction on a worker thread. The view-side `OffscreenViewport` acquires exactly one scene snapshot per call, so it cannot mix point version N with mesh version N+1.

## 7. Offscreen rendering

On Windows, `WglFboRenderer` creates an unshown `STATIC` Win32 window, chooses a pixel format, creates a WGL context, compiles GLSL 330 shaders, and attaches a color texture plus depth renderbuffer to an invisible FBO. It draws the mesh and overlays the point cloud, reads RGBA pixels back and records an image hash plus the source snapshot hash. PPM output flips the OpenGL bottom-up row order.

On unsupported hosts, or if WGL/FBO creation fails, rendering returns a structured `E_GL_CONTEXT`. CPU mesh, ledger, persistence and independent verification never depend on a GL context. A serialized `forced_unavailable_reason` parameter exercises this branch even on a capable Windows host.

## 8. Atomic persistence and recovery

Commit protocol:

1. create `staging/state-...`;
2. write every point, normal, normal-stat, source-id, patch-cache, ledger, history and state file through a temporary file plus `os.replace`;
3. write `COMMIT.json` containing hashes of all files;
4. atomically rename the staging directory to `states/state-NNNNNN`;
5. atomically replace `CURRENT`.

The production result becomes recoverable only when `CURRENT` references a complete state. Recovery reads only that state. State directories newer than `CURRENT`, missing/invalid commits and staging directories are quarantined under `orphans/recovery-*`; each move and reason is recorded in `recovery/open-*.json`.

Normal close writes a full `session_manifest.json` followed by `closed.json`. Opening a store archives the previous session manifest and removes the closed marker. A forced termination therefore has no valid close marker; the next open reports `clean_previous_exit=false` and never fabricates a successful exit report.

## 9. Reproducible fault injection

The six required codes are exercised by a serialized matrix:

| Code | Injection |
|---|---|
| `E_POINT_CLOUD_FORMAT` | XOR one byte in a copied `points.bin` magic |
| `E_NEIGHBOR_RADIUS` | radius `0.001` via operation parameters |
| `E_SNAPSHOT_STALE` | ask snapshot store for `current_version - 1` |
| `E_MESH_MISMATCH` | compare with a serialized `(0.01,0,0)` vertex offset |
| `E_CACHE_HASH` | XOR one byte in a copied committed patch cache |
| `E_GL_CONTEXT` | serialized forced-unavailable renderer parameter |

`fault_injection_config.json`, `fault_matrix_report.json` and the overall validation report retain the exact parameters and captured error contexts.
