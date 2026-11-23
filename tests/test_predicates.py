from shardbench.predicates import (extract_constraints, normalize,
                                   predicate_digest)


def test_normalization_is_byte_identical_across_surface_differences():
    a = {"and": [
        {"in": [" user_id ", [1001, 7, 42, 7]]},
        {"eq": ["STATUS", "  Paid "]},
    ]}
    b = {"and": [
        {"eq": ["status", "paid"]},
        {"in": ["user_id", [42, 1001, 7]]},
    ]}
    na, nb = normalize(a), normalize(b)
    assert na == nb
    assert predicate_digest(a) == predicate_digest(b)


def test_single_element_in_collapses_to_eq():
    assert normalize({"in": ["c", [5]]}) == {"eq": ["c", 5]}


def test_duplicate_and_clauses_deduped():
    p = {"and": [{"eq": ["a", 1]}, {"eq": ["a", 1]}]}
    assert normalize(p) == {"eq": ["a", 1]}


def test_constraint_intersection():
    cons = extract_constraints({"and": [
        {"in": ["user_id", [1, 2, 3]]},
        {"in": ["user_id", [2, 3, 4]]},
        {"range": ["amount", 10, 50]},
    ]})
    assert cons["user_id"] == {"in": [2, 3]}
    assert cons["amount"] == {"range": (10, 50)}


def test_contradiction_yields_empty_set():
    cons = extract_constraints({"and": [
        {"eq": ["user_id", 1]}, {"eq": ["user_id", 2]}]})
    assert cons["user_id"] == {"in": []}
