from __future__ import annotations

import copy

import pytest

from faultisolation.errors import DivergenceError
from faultisolation.replay import verify_paths


def test_divergence_detected_on_tampered_outcome(small_bundle, monkeypatch):
    # 强制让 merge 路径产出不同槽位，验证 E_DIVERGE 能定位首个不一致事件与字段
    from faultisolation import replay as replay_mod
    original = replay_mod.replay_path

    def fake_replay(bundle, path):
        outcomes = original(bundle, path)
        if path == "merge":
            broken = []
            for outcome in outcomes:
                if outcome.seq == 1 and outcome.slots:
                    object.__setattr__(outcome, "slots", "999")
                broken.append(outcome)
            return broken
        return outcomes

    monkeypatch.setattr(replay_mod, "replay_path", fake_replay)
    with pytest.raises(DivergenceError) as exc:
        verify_paths(small_bundle)
    assert exc.value.code == "E_DIVERGE"
    assert exc.value.field == "slots"
    assert exc.value.instance  # 能定位实例标识
