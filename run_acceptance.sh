#!/usr/bin/env bash
# Deterministic workbench acceptance: full vs incremental, injections, recovery.
set -euo pipefail
cd "$(dirname "$0")"
PY=python3.11
WS=$(mktemp -d)
echo "workspace: $WS"

echo "[1/6] regenerate deterministic samples"
$PY -m dbench.sample_generators.generate_samples >/dev/null

echo "[2/6] full render + incremental rerun + pixel comparison"
$PY -m dbench run --config configs/valid_base.json \
    --store "$WS/full" --report "$WS/full.json" --output "$WS/full/final.png" >/dev/null
$PY -m dbench run --config configs/valid_base.json \
    --store "$WS/full" --report "$WS/inc.json" --output "$WS/full/final.png" >/dev/null
$PY -m dbench compare --report-a "$WS/inc.json" --report-b "$WS/full.json" \
    --label-a incremental --label-b full --out "$WS/compare.json" >/dev/null
$PY - "$WS/compare.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
assert r["equivalent"] and r["pixels_equal"], r
print("  full==incremental (pixel + every stage fingerprint): OK")
PY

echo "[3/6] image-to-image render"
$PY -m dbench run --config configs/valid_img2img.json \
    --store "$WS/i2i" --report "$WS/i2i.json" >/dev/null
echo "  img2img: OK"

echo "[4/6] fault injection: every family maps to a business code"
for pair in "inj_001:E_NON_FINITE_LATENT" "inj_003:E_NOISE_ENTRY_TRUNCATED" \
            "inj_004:E_INJECTED_PATCH_OUT_OF_RANGE"; do
  manifest=${pair%%:*}; code=${pair##*:}
  $PY -m dbench run --config configs/valid_base.json \
      --store "$WS/inj-$manifest" --report "$WS/inj-$manifest.json" \
      --injections "samples/injections/$manifest.json" >/dev/null || true
  $PY - "$WS/inj-$manifest.json" "$code" "$manifest" <<'PY'
import json, sys
r = json.load(open(sys.argv[1])); expected = sys.argv[2]; manifest = sys.argv[3]
codes = {f["code"] for f in r["failures"]}
ids = {h["injection_id"] for h in r["injection_hits"]}
assert expected in codes, (manifest, codes)
assert ids, (manifest, "no injection id recorded")
print(f"  {manifest} -> {expected} (id={sorted(ids)}): OK")
PY
done
$PY -m dbench run --config configs/valid_img2img.json \
    --store "$WS/inj-img" --report "$WS/inj-img.json" \
    --injections samples/injections/faults.json >/dev/null || true
$PY - "$WS/inj-img.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
assert "E_IMAGE_SIZE_MISMATCH" in {f["code"] for f in r["failures"]}
assert "INJ-002" in {h["injection_id"] for h in r["injection_hits"]}
print("  image size injection -> E_IMAGE_SIZE_MISMATCH (INJ-002): OK")
PY

echo "[5/6] crash recovery + snapshot continuity gate"
# 5a. Forced-kill leftovers (orphan data without manifest) are reconciled.
$PY - "$WS/full" <<'PY'
import os, sys, subprocess
store = sys.argv[1]
orphan = os.path.join(store, "encode_prompt", "v0002.data.npy")
with open(orphan, "wb") as handle:
    handle.write(b"partial-write")
r = subprocess.run(["python3.11", "-m", "dbench", "run",
                    "--config", "configs/valid_base.json",
                    "--store", store,
                    "--report", os.path.join(store, "ok.json")],
                   capture_output=True, text=True)
assert r.returncode == 0, r.stderr
assert not os.path.exists(orphan), "orphan data was not reconciled"
print("  forced-kill orphan cleanup + safe resume: OK")
PY
# 5b. A genuine committed-version gap is rejected with E_SNAPSHOT_VERSION.
$PY -m dbench run --config configs/valid_base.json \
    --store "$WS/gap" --report "$WS/gap-base.json" >/dev/null
$PY -m dbench run --config configs/valid_city.json \
    --store "$WS/gap" --report "$WS/gap-v2.json" >/dev/null
$PY - "$WS/gap" <<'PY'
import os, sys, subprocess
store = sys.argv[1]
os.remove(os.path.join(store, "init_latent", "v0001.manifest.json"))
os.remove(os.path.join(store, "init_latent", "v0001.data.npy"))
r = subprocess.run(["python3.11", "-m", "dbench", "run",
                    "--config", "configs/valid_city.json",
                    "--store", store,
                    "--report", os.path.join(store, "r.json")],
                   capture_output=True, text=True)
assert r.returncode == 2 and "E_SNAPSHOT_VERSION" in r.stdout, r.stdout
print("  committed version gap -> E_SNAPSHOT_VERSION, exit 2: OK")
PY

echo "[6/6] unit test suite"
$PY -m unittest discover -s tests >/dev/null
echo "  all unit tests: OK"

echo "ACCEPTANCE PASSED (workspace: $WS)"
