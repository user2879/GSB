import json
import os

from tests._base import BaseCase
from vdu.pipeline import Pipeline
from vdu.snapshot import SnapshotStore
from vdu.synth import relabel_transcript


class SnapshotModeTests(BaseCase):
    def _store(self):
        return SnapshotStore(self.snapshots)

    def test_rebuild_incremental_resume_sequence(self):
        store = self._store()
        first = Pipeline(self.samples, self.config_path,
                         self.snapshots).run()
        self.assertEqual(first["mode"], "rebuild")

        # Relabel one segment: label fingerprint changes -> incremental.
        new_tr = relabel_transcript(self.transcript, 4, ["seg02"])
        self.write_transcript(new_tr)
        second = Pipeline(self.samples, self.config_path,
                          self.snapshots).run()
        self.assertEqual(second["mode"], "incremental")

        # No content change -> resume.
        third = Pipeline(self.samples, self.config_path,
                         self.snapshots).run()
        self.assertEqual(third["mode"], "resume")

    def test_config_hash_change_forces_rebuild(self):
        Pipeline(self.samples, self.config_path, self.snapshots).run()
        self.config["calibration_gain"] = 0.4
        with open(self.config_path, "w") as handle:
            json.dump(self.config, handle)
        mode, _ = self._store().detect_mode(self.config, self.transcript)
        self.assertEqual(mode, "rebuild")
