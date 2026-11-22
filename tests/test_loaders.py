"""加载校验层：非法输入与业务码映射。"""

import json

import pytest

from tlc import datagen
from tlc.errors import (
    BizError,
    E_CONTRIB_INVALID,
    E_ENCODER_SHAPE,
    E_FEATURE_MISSING,
    E_PLAN_INVALID,
    E_TASK_INVALID,
)
from tlc.loaders import (
    apply_replacements,
    load_contributions,
    load_encoder,
    load_features,
    load_replacement_plan,
    load_tasks,
    validate_task_samples,
)


@pytest.fixture()
def data_dir(tmp_path):
    return datagen.generate(str(tmp_path / "data"))


def _load_json(path):
    return json.load(open(path, encoding="utf-8"))


def _dump(path, obj):
    json.dump(obj, open(path, "w", encoding="utf-8"))


def test_duplicate_task_name(data_dir, tmp_path):
    tasks = _load_json(f"{data_dir}/tasks.json")
    tasks["tasks"].append(dict(tasks["tasks"][0]))
    path = tmp_path / "bad_tasks.json"
    _dump(path, tasks)
    with pytest.raises(BizError) as exc:
        load_tasks(str(path))
    assert exc.value.code == E_TASK_INVALID


def test_empty_classes(data_dir, tmp_path):
    tasks = _load_json(f"{data_dir}/tasks.json")
    tasks["tasks"][0]["classes"] = []
    path = tmp_path / "bad_tasks.json"
    _dump(path, tasks)
    with pytest.raises(BizError) as exc:
        load_tasks(str(path))
    assert exc.value.code == E_TASK_INVALID


def test_missing_sample_id(data_dir):
    tasks = load_tasks(f"{data_dir}/tasks.json")
    bank = load_features(f"{data_dir}/features.jsonl")
    tasks[0].train_ids[0] = "no_such_sample"
    with pytest.raises(BizError) as exc:
        validate_task_samples(tasks, bank)
    assert exc.value.code == E_FEATURE_MISSING


def test_label_outside_classes(data_dir):
    tasks = load_tasks(f"{data_dir}/tasks.json")
    bank = load_features(f"{data_dir}/features.jsonl")
    sid = tasks[0].train_ids[0]
    bank.samples[sid]["labels"][tasks[0].name] = "ZZZ"
    with pytest.raises(BizError) as exc:
        validate_task_samples(tasks, bank)
    assert exc.value.code == E_TASK_INVALID


def test_encoder_shape_mismatch(data_dir, tmp_path):
    enc = _load_json(f"{data_dir}/encoder.json")
    enc["layers"][1]["weight"] = [row[:-1] for row in enc["layers"][1]["weight"]]
    enc["layers"][1]["checksum"] = ""
    path = tmp_path / "bad_encoder.json"
    _dump(path, enc)
    with pytest.raises(BizError) as exc:
        load_encoder(str(path))
    assert exc.value.code == E_ENCODER_SHAPE


def test_encoder_checksum_mismatch(data_dir, tmp_path):
    enc = _load_json(f"{data_dir}/encoder.json")
    enc["layers"][0]["bias"][0] += 1.0
    path = tmp_path / "bad_encoder.json"
    _dump(path, enc)
    with pytest.raises(BizError) as exc:
        load_encoder(str(path))
    assert exc.value.code == E_ENCODER_SHAPE


def test_contrib_missing_entry(data_dir, tmp_path):
    contrib = _load_json(f"{data_dir}/contributions.json")
    contrib["entries"] = [e for e in contrib["entries"] if not (
        e["layer"] == "layer2" and e["task"] == "task_alpha")]
    path = tmp_path / "bad_contrib.json"
    _dump(path, contrib)
    enc = load_encoder(f"{data_dir}/encoder.json")
    tasks = load_tasks(f"{data_dir}/tasks.json")
    with pytest.raises(BizError) as exc:
        load_contributions(str(path), enc, tasks)
    assert exc.value.code == E_CONTRIB_INVALID


def test_contrib_non_finite(data_dir, tmp_path):
    contrib = _load_json(f"{data_dir}/contributions.json")
    contrib["entries"][0]["sensitivity"] = float("nan")
    path = tmp_path / "bad_contrib.json"
    _dump(path, contrib)
    enc = load_encoder(f"{data_dir}/encoder.json")
    tasks = load_tasks(f"{data_dir}/tasks.json")
    with pytest.raises(BizError) as exc:
        load_contributions(str(path), enc, tasks)
    assert exc.value.code == E_CONTRIB_INVALID


def test_plan_unknown_layer(data_dir):
    enc = load_encoder(f"{data_dir}/encoder.json")
    tasks = load_tasks(f"{data_dir}/tasks.json")
    contrib = load_contributions(f"{data_dir}/contributions.json", enc, tasks)
    with pytest.raises(BizError) as exc:
        apply_replacements(enc, [{"layer": "nope", "new_version": 2}], contrib)
    assert exc.value.code == E_PLAN_INVALID


def test_plan_version_decrease(data_dir):
    enc = load_encoder(f"{data_dir}/encoder.json")
    tasks = load_tasks(f"{data_dir}/tasks.json")
    contrib = load_contributions(f"{data_dir}/contributions.json", enc, tasks)
    with pytest.raises(BizError) as exc:
        apply_replacements(enc, [{"layer": "layer1", "new_version": 0}], contrib)
    assert exc.value.code == E_PLAN_INVALID


def test_plan_frozen_layer(data_dir):
    enc = load_encoder(f"{data_dir}/encoder.json")
    tasks = load_tasks(f"{data_dir}/tasks.json")
    contrib = load_contributions(f"{data_dir}/contributions.json", enc, tasks)
    for key in contrib:
        if key[0] == "layer1":
            contrib[key]["frozen"] = True
    with pytest.raises(BizError) as exc:
        apply_replacements(enc, [{"layer": "layer1", "new_version": 2}], contrib)
    assert exc.value.code == E_PLAN_INVALID


def test_plan_structure_ok(data_dir):
    batches = load_replacement_plan(f"{data_dir}/replacement_plan.json")
    assert len(batches) == 3
