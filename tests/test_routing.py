from __future__ import annotations

from shard_diag.canonical import canonical_bytes
from shard_diag.routing import compare_route, generate_sample_keys, sample_hit_shards


def test_known_drift_directions_and_offsets():
    from shard_diag.spec import validate_spec

    from conftest import make_spec_values

    spec = validate_spec(make_spec_values())
    old = spec.rule("orders", 1)
    new = spec.rule("orders", 2)
    right = compare_route(
        event_seq=3,
        db_id="db-b",
        table="orders",
        key_column="order_id",
        route_key=90,
        previous_rule=old,
        current_rule=new,
    )
    assert right["old_shard"] == "orders:s0"
    assert right["new_shard"] == "orders:s1"
    assert right["drift_direction"] == "right"
    assert right["boundary_index"] == 1
    assert right["boundary_offset"] == -20
    assert right["changed"] is True

    left = compare_route(
        event_seq=1,
        db_id="db-a",
        table="orders",
        key_column="order_id",
        route_key=210,
        previous_rule=old,
        current_rule=new,
    )
    assert left["drift_direction"] == "left"
    assert left["old_shard"] == "orders:s2"
    assert left["new_shard"] == "orders:s1"

    stable = compare_route(
        event_seq=5,
        db_id="db-a",
        table="orders",
        key_column="order_id",
        route_key=700,
        previous_rule=old,
        current_rule=new,
    )
    assert stable["changed"] is False
    assert stable["drift_direction"] == "none"


def test_route_records_are_byte_identical_regardless_of_input_order():
    from shard_diag.spec import validate_spec

    from conftest import make_spec_values

    spec = validate_spec(make_spec_values())

    def records(events):
        result = []
        for event in events:
            db = spec.database(event.database)
            rule = spec.event_rule(event)
            previous = spec.previous_rule(event.table, db.route_version)
            result.append(
                compare_route(
                    event_seq=event.seq,
                    db_id=event.database,
                    table=event.table,
                    key_column=rule.key_column,
                    route_key=event.route_key,
                    previous_rule=previous,
                    current_rule=rule,
                )
            )
        return sorted(result, key=lambda record: (record["event_seq"], record["db_id"], record["table"]))

    forward = records(spec.events)
    backward = records(list(reversed(spec.events)))
    assert canonical_bytes(forward) == canonical_bytes(backward)


def test_sample_keys_are_deterministic_and_include_event_key():
    from shard_diag.spec import validate_spec

    from conftest import make_spec_values

    spec = validate_spec(make_spec_values())
    rule = spec.rule("orders", 2)
    first = generate_sample_keys(seed=101, event_seq=1, route_key=210, rule=rule, sample_count=5)
    second = generate_sample_keys(seed=101, event_seq=1, route_key=210, rule=rule, sample_count=5)
    assert first == second
    assert first[0] == 210
    assert all(rule.key_min <= key < rule.key_max for key in first)
    assert sample_hit_shards(rule, first) == tuple(sorted(sample_hit_shards(rule, first)))


def test_no_previous_version_has_no_drift():
    from shard_diag.spec import validate_spec

    from conftest import make_spec_values

    spec = validate_spec(make_spec_values())
    current = spec.rule("orders", 1)
    record = compare_route(
        event_seq=1,
        db_id="db-a",
        table="orders",
        key_column="order_id",
        route_key=210,
        previous_rule=None,
        current_rule=current,
    )
    assert record["old_route_version"] is None
    assert record["old_shard"] is None
    assert record["new_shard"] == "orders:s2"
    assert record["changed"] is False
    assert record["drift_direction"] == "none"


def test_boundary_sequence_length_change_uses_first_new_boundary_index():
    from shard_diag.routing import first_boundary_diff
    from shard_diag.spec import Segment, TableRule

    old = TableRule("orders", "order_id", 1, (Segment("orders:s0", 0, 100), Segment("orders:s1", 100, 300)))
    new = TableRule(
        "orders",
        "order_id",
        2,
        (
            Segment("orders:s0", 0, 100),
            Segment("orders:s1", 100, 300),
            Segment("orders:s2", 300, 400),
        ),
    )
    diff = first_boundary_diff(old, new)
    assert diff.index == 3
    assert diff.old is None
    assert diff.new == 400
    assert diff.offset == 0
