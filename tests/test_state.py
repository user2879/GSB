from hdr_sdr_workbench.state import PipelineState, StateTable, default_state


def test_state_id_content_addressed():
    s1 = PipelineState("bt709", 1.0, "reinhard", 4.0, "srgb")
    s2 = PipelineState("bt709", 1.0, "reinhard", 4.0, "srgb")
    assert s1.state_id() == s2.state_id()
    s3 = PipelineState("bt709", 1.1, "reinhard", 4.0, "srgb")
    assert s1.state_id() != s3.state_id()


def test_closure_precision_and_noop():
    table = StateTable(6)
    op = table.apply([1, 2, 3], {"exposure_stops": 0.5})
    assert op["closure"] == [1, 2, 3]
    noop = table.apply([1, 2, 3], {"exposure_stops": 0.5})
    assert noop["closure"] == []
    op2 = table.apply([4], {"tonemap": "reinhard", "tonemap_param": 2.0})
    assert op2["closure"] == [4]


def test_range_vs_frame_reach_same_state():
    # 区间操作 vs 等价逐帧操作，最终每帧 state_id 必须一致
    t_range = StateTable(4)
    t_range.apply([0, 1, 2, 3], {"exposure_stops": 0.25})
    t_frame = StateTable(4)
    for i in range(4):
        t_frame.apply([i], {"exposure_stops": 0.25})
    assert t_range.state_ids() == t_frame.state_ids()


def test_replay_historical_state():
    table = StateTable(5)
    table.apply([0, 1], {"exposure_stops": 1.0})
    table.apply([2], {"colorspace": "bt709"})
    table.apply([0, 1, 2], {"tonemap": "reinhard", "tonemap_param": 3.0})
    at_v1 = table.replay_to(1)
    assert at_v1.states[0].exposure_stops == 1.0
    assert at_v1.states[2] == default_state()
    # 重放副本与主表隔离
    at_v1.apply([0], {"exposure_stops": 5.0})
    assert table.states[0].exposure_stops == 1.0
