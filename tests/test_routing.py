from gsb.routing import bucket_for, route_request
from gsb.spec import RouteRule

RULE = RouteRule(
    rule_id="r-main",
    stable_backend="b-stable-a",
    gray_backend="b-gray-a",
    weight=5000,
    rule_version=1,
)


def test_bucket_range_and_determinism():
    for i in range(200):
        bucket = bucket_for("r-main", f"client-{i}")
        assert 0 <= bucket < 10000
        assert bucket == bucket_for("r-main", f"client-{i}")


def test_same_client_same_route_under_same_weight():
    first = route_request(RULE, "q1", "ba", "c-alpha")
    for _ in range(10):
        again = route_request(RULE, "qX", "bb", "c-alpha")
        assert again.route == first.route
        assert again.bucket == first.bucket


def test_route_threshold_semantics():
    always_gray = RouteRule(**{**RULE.__dict__, "weight": 10000})
    always_stable = RouteRule(**{**RULE.__dict__, "weight": 0})
    for i in range(50):
        key = f"c-{i}"
        assert route_request(always_gray, "q", "b", key).route == "gray"
        assert route_request(always_stable, "q", "b", key).route == "stable"


def test_independent_of_dict_order():
    # 以不同构造顺序得到相同规则对象，分桶结论必须一致
    keys_a = {"rule_id": "r-main", "stable_backend": "b-stable-a",
              "gray_backend": "b-gray-a", "weight": 5000, "rule_version": 1}
    keys_b = dict(reversed(list(keys_a.items())))
    rule_a = RouteRule(**keys_a)
    rule_b = RouteRule(**keys_b)
    for i in range(20):
        key = f"c-{i}"
        assert route_request(rule_a, "q", "b", key).route == route_request(
            rule_b, "q", "b", key
        ).route


def test_hit_record_carries_ledger_fields():
    hit = route_request(RULE, "q-1", "batch-1", "c-alpha")
    assert hit.rule_version == 1
    assert hit.weight == 5000
    expected_backend = (
        "b-gray-a" if hit.route == "gray" else "b-stable-a"
    )
    assert hit.backend_id == expected_backend
