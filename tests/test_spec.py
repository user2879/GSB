import unittest

from cfgpipe.errors import SpecError
from cfgpipe.spec import (apply_changeset, initial_state, validate_changeset,
                          validate_declaration)


def base_decl():
    return {
        "session": {"scale_exp": 3, "groups": 8, "global_concurrency": 2,
                    "node_concurrency": 1, "dispatch_timeout_ms": 500,
                    "backoff_ms": [10, 20]},
        "items": [
            {"key": "alpha", "value": 1.5, "version": 1, "expire_ms": 5000},
            {"key": "beta", "value": 0, "version": 2, "expire_ms": 5000,
             "derived_from": "alpha", "derived_coeff": 2.5},
            {"key": "gamma", "value": 3, "version": 3, "expire_ms": 4000},
        ],
        "nodes": [
            {"id": "n1", "host": "127.0.0.1", "port": 24001,
             "subscribes": ["alpha", "beta"], "seed": 7, "sample_size": 100},
            {"id": "n2", "host": "127.0.0.1", "port": 24002,
             "subscribes": ["gamma"], "seed": 8, "sample_size": 50},
        ],
    }


class TestDeclaration(unittest.TestCase):
    def test_valid(self):
        decl = validate_declaration(base_decl())
        self.assertEqual([it.key for it in decl.items],
                         ["alpha", "beta", "gamma"])
        self.assertEqual(decl.session.scale_exp, 3)

    def _expect(self, mutate, key, field):
        raw = base_decl()
        mutate(raw)
        with self.assertRaises(SpecError) as ctx:
            validate_declaration(raw)
        err = ctx.exception
        self.assertEqual(err.code, "E_SPEC")
        if key is not None:
            self.assertEqual(err.key, key)
        self.assertEqual(err.field, field)
        return err

    def test_illegal_key(self):
        self._expect(lambda r: r["items"][0].update({"key": "0bad key"}),
                     "0bad key", "key")

    def test_duplicate_key(self):
        self._expect(lambda r: r["items"][1].update({"key": "alpha"}),
                     "alpha", "key")

    def test_non_quantizable_value(self):
        self._expect(lambda r: r["items"][0].update({"value": "abc"}),
                     "alpha", "value")

    def test_version_not_monotonic(self):
        self._expect(lambda r: r["items"][2].update({"version": 2}),
                     "gamma", "version")

    def test_expire_not_positive(self):
        self._expect(lambda r: r["items"][0].update({"expire_ms": 0}),
                     "alpha", "expire_ms")

    def test_derived_target_missing(self):
        self._expect(lambda r: r["items"][1].update({"derived_from": "nope"}),
                     "beta", "derived_from")

    def test_derived_cycle(self):
        def mutate(r):
            r["items"][0].update({"derived_from": "beta",
                                  "derived_coeff": 1.0})
        self._expect(mutate, "alpha", "derived_from")

    def test_derived_coeff_without_from(self):
        self._expect(lambda r: r["items"][0].update({"derived_coeff": 1.0}),
                     "alpha", "derived_coeff")

    def test_node_host_not_loopback(self):
        self._expect(lambda r: r["nodes"][0].update({"host": "localhost"}),
                     "n1", "host")

    def test_node_duplicate_id(self):
        self._expect(lambda r: r["nodes"][1].update({"id": "n1"}),
                     "n1", "id")

    def test_node_bad_port(self):
        self._expect(lambda r: r["nodes"][0].update({"port": 80}),
                     "n1", "port")

    def test_subscribe_unknown_key(self):
        self._expect(lambda r: r["nodes"][0].update(
            {"subscribes": ["alpha", "zzz"]}), "n1", "subscribes")

    def test_first_violation_is_stable(self):
        raw = base_decl()
        raw["items"][0]["expire_ms"] = -1
        raw["items"][1]["expire_ms"] = -2
        with self.assertRaises(SpecError) as ctx:
            validate_declaration(raw)
        self.assertEqual(ctx.exception.key, "alpha")
        self.assertEqual(ctx.exception.field, "expire_ms")

    def test_scale_exp_bounds(self):
        for bad in (-1, 7, "3"):
            raw = base_decl()
            raw["session"]["scale_exp"] = bad
            with self.assertRaises(SpecError) as ctx:
                validate_declaration(raw)
            self.assertEqual(ctx.exception.field, "scale_exp")


class TestChangeset(unittest.TestCase):
    def setUp(self):
        self.decl = validate_declaration(base_decl())

    def test_valid_and_version_rules(self):
        # 首个变更集版本允许等于声明最大版本（不小于即可）
        cs = validate_changeset(
            {"version": 3, "changes": [{"key": "alpha", "value": 2.0}]},
            self.decl, None)
        self.assertEqual(cs.version, 3)
        with self.assertRaises(SpecError):
            validate_changeset(
                {"version": 2, "changes": [{"key": "alpha", "value": 2.0}]},
                self.decl, None)
        with self.assertRaises(SpecError):
            validate_changeset(
                {"version": 3, "changes": [{"key": "alpha", "value": 2.0}]},
                self.decl, 3)
        with self.assertRaises(SpecError):
            validate_changeset(
                {"version": 4, "changes": [{"key": "alpha", "value": 2.0}]},
                self.decl, 4)

    def test_unknown_key(self):
        with self.assertRaises(SpecError) as ctx:
            validate_changeset(
                {"version": 9, "changes": [{"key": "zzz", "value": 1}]},
                self.decl, 3)
        self.assertEqual(ctx.exception.key, "zzz")

    def test_coeff_on_non_derived(self):
        with self.assertRaises(SpecError) as ctx:
            validate_changeset(
                {"version": 9,
                 "changes": [{"key": "alpha", "derived_coeff": 2.0}]},
                self.decl, 3)
        self.assertEqual(ctx.exception.field, "derived_coeff")

    def test_subscriber_only_change_keeps_version(self):
        cs = validate_changeset(
            {"version": 9,
             "changes": [{"key": "gamma", "subscribers_add": ["n1"]}]},
            self.decl, 3)
        state = initial_state(self.decl)
        before = state.items["gamma"].version
        nxt = apply_changeset(state, cs)
        self.assertEqual(nxt.items["gamma"].version, before)
        self.assertEqual(nxt.subs["gamma"], ["n1", "n2"])

    def test_content_change_bumps_version(self):
        cs = validate_changeset(
            {"version": 9, "changes": [{"key": "gamma", "value": 5}]},
            self.decl, 3)
        nxt = apply_changeset(initial_state(self.decl), cs)
        self.assertEqual(nxt.items["gamma"].version, 9)


if __name__ == "__main__":
    unittest.main()
