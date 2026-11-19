import os
import tempfile
import unittest

from rts_replay.generator import PROFILES, generate_all, load_rules
from rts_replay.hashing import sha256_file
from rts_replay.events import load_events


class GeneratorTests(unittest.TestCase):
    def test_byte_identical_regeneration(self):
        with tempfile.TemporaryDirectory() as dir_a, tempfile.TemporaryDirectory() as dir_b:
            manifests_a = generate_all(dir_a)
            manifests_b = generate_all(dir_b)
            self.assertEqual(
                [item["scenario"] for item in manifests_a],
                [item["scenario"] for item in manifests_b],
            )
            for manifest in manifests_a:
                name = manifest["scenario"]
                for filename in ("events.jsonl", "rules.json", "manifest.json"):
                    hash_a = sha256_file(os.path.join(dir_a, "samples", name, filename))
                    hash_b = sha256_file(os.path.join(dir_b, "samples", name, filename))
                    self.assertEqual(hash_a, hash_b, filename)

    def test_manifest_counts_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            manifests = generate_all(directory)
            for manifest in manifests:
                events = load_events(
                    os.path.join(
                        directory, "samples", manifest["scenario"], "events.jsonl"
                    )
                )
                self.assertEqual(len(events), manifest["event_count"])
                self.assertEqual(
                    sha256_file(
                        os.path.join(
                            directory,
                            "samples",
                            manifest["scenario"],
                            manifest["events_file"],
                        )
                    ),
                    manifest["events_sha256"],
                )

    def test_rules_load_and_validate(self):
        with tempfile.TemporaryDirectory() as directory:
            generate_all(directory)
            for profile in PROFILES:
                rules = load_rules(
                    os.path.join(directory, "samples", profile.name, "rules.json")
                )
                self.assertEqual(rules.match_seed, profile.seed)
                self.assertEqual(len(rules.factions), len(profile.faction_specs))


if __name__ == "__main__":
    unittest.main()
