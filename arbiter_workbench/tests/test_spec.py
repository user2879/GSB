import json
import unittest
from pathlib import Path

from arbiter.errors import SpecError
from arbiter.sample import DEFAULT_KEYS
from arbiter.spec import load_declarations

DECLS = Path(__file__).resolve().parents[1] / "decls"


class SpecValidationTest(unittest.TestCase):
    def test_loads_valid_declarations(self):
        decls = load_declarations(DECLS, DEFAULT_KEYS)
        self.assertEqual(len(decls.environments), 4)
        self.assertEqual(len(decls.instances), 6)
        self.assertEqual(len(decls.views), 3)
        self.assertTrue(all(e.host == "127.0.0.1" for e in decls.environments.values()))

    def _tmp_decls(self, tmp: Path, mutate) -> Path:
        import shutil
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(DECLS, tmp)
        mutate(tmp)
        return tmp

    def test_first_illegal_field_reported(self):
        tmp = Path("/tmp/wb_spec_bad")
        def mutate(path: Path):
            data = json.loads((path / "environments.json").read_text())
            data["environments"][0]["port"] = 80
            (path / "environments.json").write_text(json.dumps(data))
        decl_dir = self._tmp_decls(tmp, mutate)
        with self.assertRaises(SpecError) as ctx:
            load_declarations(decl_dir, DEFAULT_KEYS)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertEqual(ctx.exception.context.field, "port")
        self.assertIn("env-prod-a", ctx.exception.context.object_id)

    def test_non_loopback_host_rejected(self):
        tmp = Path("/tmp/wb_spec_host")
        def mutate(path: Path):
            data = json.loads((path / "environments.json").read_text())
            data["environments"][1]["host"] = "10.0.0.2"
            (path / "environments.json").write_text(json.dumps(data))
        with self.assertRaises(SpecError) as ctx:
            load_declarations(self._tmp_decls(tmp, mutate), DEFAULT_KEYS)
        self.assertEqual(ctx.exception.context.field, "host")

    def test_unknown_env_ref_in_instance(self):
        tmp = Path("/tmp/wb_spec_ref")
        def mutate(path: Path):
            data = json.loads((path / "instances.json").read_text())
            data["instances"][0]["env_ref"] = "env-nope"
            (path / "instances.json").write_text(json.dumps(data))
        with self.assertRaises(SpecError) as ctx:
            load_declarations(self._tmp_decls(tmp, mutate), DEFAULT_KEYS)
        self.assertEqual(ctx.exception.context.field, "env_ref")

    def test_bad_policy_and_duplicate_port(self):
        tmp = Path("/tmp/wb_spec_policy")
        def mutate(path: Path):
            data = json.loads((path / "views.json").read_text())
            data["views"][0]["policies"]["pool.max_size"] = "quorum"
            (path / "views.json").write_text(json.dumps(data))
        with self.assertRaises(SpecError) as ctx:
            load_declarations(self._tmp_decls(tmp, mutate), DEFAULT_KEYS)
        self.assertEqual(ctx.exception.context.field, "policies.pool.max_size")

    def test_negative_tolerance_rejected(self):
        tmp = Path("/tmp/wb_spec_tol")
        def mutate(path: Path):
            data = json.loads((path / "views.json").read_text())
            data["views"][0]["tolerance"] = -1
            (path / "views.json").write_text(json.dumps(data))
        with self.assertRaises(SpecError) as ctx:
            load_declarations(self._tmp_decls(tmp, mutate), DEFAULT_KEYS)
        self.assertEqual(ctx.exception.context.field, "tolerance")


if __name__ == "__main__":
    unittest.main()
