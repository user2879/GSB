#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export QT_QPA_PLATFORM=offscreen
echo "== [1/7] generate deterministic sample =="
python3 main.py generate --input samples/sample.json
echo "== [2/7] solve (chunked, bounded threads, atomic persistence) =="
python3 main.py solve --input samples/sample.json --state-dir state/accept --output out/accept 2>/dev/null
echo "== [3/7] verify state digests =="
python3 main.py verify --input samples/sample.json --state-dir state/accept
echo "== [4/7] serial vs parallel equivalence =="
python3 main.py serial-check --input samples/sample.json --state-dir state/accept_serial
echo "== [5/7] resume after SIGTERM =="
python3 - <<'PY'
import json, os, signal, subprocess, sys, time
p = subprocess.Popen([sys.executable,"main.py","solve","--input","samples/sample.json",
                      "--state-dir","state/accept_int","--output","out/accept_int"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
while not os.path.exists("state/accept_int/chunk_000000.json"):
    time.sleep(0.02)
p.send_signal(signal.SIGTERM)
rc = p.wait(timeout=20)
print("interrupt exit code:", rc)
PY
python3 main.py resume --input samples/sample.json --state-dir state/accept_int --output out/accept_int 2>/dev/null
echo "== [6/7] offscreen UI self-test =="
python3 main.py ui-selftest --input samples/sample.json --state-dir state/accept_ui \
  --state-export out/accept_ui_state.json --screenshot out/accept_ui.png
echo "== [7/7] invalid input business codes =="
python3 - <<'PY'
import copy, json, os, subprocess, sys
base = json.load(open("samples/sample.json"))
mutations = {
 "E_BONE": lambda d: d["skeleton"]["bones"][3].__setitem__("name","root"),
 "E_HIERARCHY": lambda d: d["skeleton"]["bones"][1].__setitem__("parent","head"),
 "E_LENGTH": lambda d: d["skeleton"]["bones"][2].__setitem__("length",0),
 "E_CURVE": lambda d: d["curves"]["rotations"].__setitem__("root", d["curves"]["rotations"]["root"][:-1]),
 "E_QUATERNION": lambda d: d["curves"]["rotations"]["spine"][5].__setitem__(2, float("inf")),
 "E_LIMIT": lambda d: d["constraints"]["angle_limits"][0].__setitem__("lower",9.0),
 "E_TARGET": lambda d: d["constraints"]["ik_targets"][0].__setitem__("frames",[999]),
 "E_CONTACT": lambda d: d["constraints"].__setitem__("contact_frames",[77]),
 "E_PARAM": lambda d: d["params"].__setitem__("damping",2.0),
}
seen = []
for expected, mutate in mutations.items():
    data = copy.deepcopy(base); mutate(data)
    json.dump(data, open("samples/_bad.json","w"))
    r = subprocess.run([sys.executable,"main.py","verify","--input","samples/_bad.json","--state-dir","state/_bad"],
                       capture_output=True, text=True)
    code = json.loads(r.stdout)["error"]["code"]
    seen.append(code); assert code == expected, (expected, code)
os.remove("samples/_bad.json")
print("all business codes correct:", seen)
PY
echo "ALL ACCEPTANCE STEPS PASSED"
