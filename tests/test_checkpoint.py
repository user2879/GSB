import json
import os

import pytest

from fwaudit.engine import AuditEngine
from fwaudit.generator import build_image, build_patch
from fwaudit.patch import BudgetConfig
from fwaudit.storage import (
    file_lock, atomic_write_json, load_checkpoint,
    checkpoint_path, CHECKPOINT_STRUCT_VERSION,
)
from fwaudit.errors import AuditError, E_LOCK


def test_checkpoint_written_and_resumed(tmp_path, partitions, budget):
    v1 = build_image(partitions, base_image_id=100)
    patch = build_patch(
        [{"name": "kernel", "write_offset": 0, "write_length": 8}],
        base_image_id=100,
    )
    engine = AuditEngine(str(tmp_path / "state"), budget=budget)
    engine.run([("v1", v1)], [("p", patch)])
    assert os.path.exists(checkpoint_path(str(tmp_path / "state")))

    restored = AuditEngine(str(tmp_path / "state"), budget=budget)
    assert restored.restore() is True
    assert restored.generation_watermark == 1
    assert len(restored.patch_records) == 1


def test_policy_version_change_forces_rebuild(tmp_path, partitions):
    v1 = build_image(partitions, base_image_id=100)
    b1 = BudgetConfig(policy_version=1)
    state = str(tmp_path / "state")
    engine = AuditEngine(state, budget=b1)
    engine.run([("v1", v1)], [])

    b2 = BudgetConfig(policy_version=2)
    rebuilt = AuditEngine(state, budget=b2)
    assert rebuilt.restore() is False
    assert rebuilt.generation_watermark == 0
    assert rebuilt.images == []


def test_struct_version_change_forces_rebuild(tmp_path, partitions, budget):
    v1 = build_image(partitions, base_image_id=100)
    state = str(tmp_path / "state")
    engine = AuditEngine(state, budget=budget)
    engine.run([("v1", v1)], [])
    path = checkpoint_path(state)
    payload = json.loads(open(path).read())
    payload["struct_version"] = 999
    atomic_write_json(path, payload)

    rebuilt = AuditEngine(state, budget=budget)
    assert rebuilt.restore() is False


def test_lock_conflict_returns_e_lock(tmp_path):
    with file_lock(str(tmp_path)):
        with pytest.raises(AuditError) as exc:
            with file_lock(str(tmp_path)):
                pass
    assert exc.value.code == E_LOCK


def test_atomic_replace_uses_temp_and_rename(tmp_path):
    target = tmp_path / "report.json"
    atomic_write_json(str(target), {"a": 1})
    atomic_write_json(str(target), {"a": 2})
    assert json.loads(target.read_text()) == {"a": 2}
    leftovers = [
        name for name in os.listdir(str(tmp_path))
        if name.endswith(".tmp")
    ]
    assert leftovers == []
