import json

from workbench import codes, loader


def schemas_of(dataset):
    return {n: t["param_schema"] for n, t in dataset["tools"].items()}


def test_valid_dataset_loads_clean(dataset):
    assert dataset["config"] is not None
    assert dataset["load_failures"] == []
    assert len(dataset["turns"]) == 24
    assert len(dataset["acoustics"]) == 24


def test_duplicate_turn_id(tmp_path, dataset):
    path = tmp_path / "turns.jsonl"
    turn = dict(dataset["turns"][0])
    path.write_text(json.dumps(turn) + "\n" + json.dumps(turn) + "\n")
    turns, failures = loader.load_turns(str(path), dataset["vocab"], schemas_of(dataset))
    assert len(turns) == 1
    assert failures[0]["code"] == codes.E_LOAD_TURN_DUP


def test_vocab_violation_and_processing_continues(tmp_path, dataset):
    path = tmp_path / "turns.jsonl"
    bad = dict(dataset["turns"][0], terms=["not_in_vocab"])
    good = dict(dataset["turns"][1])
    path.write_text(json.dumps(bad) + "\n" + json.dumps(good) + "\n")
    turns, failures = loader.load_turns(str(path), dataset["vocab"], schemas_of(dataset))
    assert [f["code"] for f in failures] == [codes.E_LOAD_VOCAB]
    assert [t["turn_id"] for t in turns] == [good["turn_id"]]


def test_tool_param_schema_mismatch(tmp_path, dataset):
    path = tmp_path / "turns.jsonl"
    turn = dict(dataset["turns"][0])
    turn["tool_calls"] = [{"tool": "sentiment_polarity",
                           "params": {"pos_count": "x", "neg_count": 1, "term_count": 3}}]
    path.write_text(json.dumps(turn) + "\n")
    schemas = {n: t["param_schema"] for n, t in dataset["tools"].items()}
    _, failures = loader.load_turns(str(path), dataset["vocab"], schemas)
    assert failures[0]["code"] == codes.E_LOAD_TOOL_PARAMS


def test_nonfinite_acoustic(tmp_path):
    path = tmp_path / "ac.jsonl"
    rec = {"turn_id": "T001", "pitch_mean": float("nan"), "pitch_var": 1.0,
           "energy_quantiles": {"0.5": 0.5}}
    path.write_text(json.dumps(rec) + "\n")
    _, failures = loader.load_acoustics(str(path))
    assert failures[0]["code"] == codes.E_LOAD_ACOUSTIC_NONFINITE


def test_cross_check_missing_and_orphan(dataset):
    turns = dataset["turns"][:2]
    acoustics = {dataset["turns"][0]["turn_id"]: {},
                 "T999": {}}
    failures = loader.cross_check(turns, acoustics)
    codes_seen = {f["code"] for f in failures}
    assert codes.E_LOAD_ACOUSTIC_MISSING in codes_seen
    assert codes.E_LOAD_ACOUSTIC_ORPHAN in codes_seen


def test_config_validation_codes(dataset):
    cfg = json.loads(json.dumps(dataset["config"]))
    cfg["tolerances"]["abs"] = -1
    cfg["precision"]["frac_bits"] = 7
    cfg["fusion_point_budget"] = 0
    cfg["interactive_latency_ms"] = -5
    failures = loader.validate_config(cfg)
    codes_seen = {f["code"] for f in failures}
    assert codes.E_LOAD_TOL in codes_seen
    assert codes.E_LOAD_PRECISION in codes_seen
    assert codes.E_LOAD_BUDGET in codes_seen


def test_failure_list_order_fixed(tmp_path, dataset):
    # failures from turns file precede acoustics file failures regardless
    # of discovery order in code
    turns_path = tmp_path / "turns.jsonl"
    bad_turn = dict(dataset["turns"][0], terms=["bad_term"])
    turns_path.write_text(json.dumps(bad_turn) + "\n")
    ac_path = tmp_path / "ac.jsonl"
    ac_path.write_text('{"turn_id": "T1", "pitch_mean": "NaN"}\n')
    _, tf = loader.load_turns(str(turns_path), dataset["vocab"], {})
    _, af = loader.load_acoustics(str(ac_path))
    ordered = tf + af
    assert ordered[0]["source"] == "turns"
    assert ordered[-1]["source"] == "acoustics"
