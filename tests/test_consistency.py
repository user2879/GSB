import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from segpos import ProbabilityTable
from tests._fixtures import TAGS, make_pipeline, write_workspace


def comparable(pipe):
    return {
        sid: {
            "tokens": [
                {
                    "start": t["start"],
                    "end": t["end"],
                    "text": t["text"],
                    "match_basis": t["match_basis"],
                    "hit": t["hit"],
                    "boundary_confidence": t["boundary_confidence"],
                    "pos": t["pos"],
                    "local_path_score": t["local_path_score"],
                }
                for t in payload["tokens"]
            ],
            "pos_sequence": payload["pos_sequence"],
            "sentence_logprob": payload["sentence_logprob"],
            "smoothing_hits": payload["smoothing_hits"],
            "low_confidence_words": payload["low_confidence_words"],
        }
        for sid, payload in sorted(pipe.results_by_id().items())
    }


def failures(pipe):
    return sorted(
        [
            (f["sentence_id"], f["stage"], f["code"], f["exception_type"])
            for f in pipe.failures
        ]
    )


class IncrementalConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.paths = write_workspace(self.root / "ws", workers=2, interval=2)
        self.snap = self.root / "snap"
        self.pipe = make_pipeline(self.paths, self.snap)
        self.pipe.run()

    def tearDown(self):
        self.tmp.cleanup()

    def _full_rebuild(self):
        full_root = self.root / "full"
        full_paths = write_workspace(full_root, workers=1, interval=10)
        final_dict = full_root / "dictionary.txt"
        final_dict.write_text(
            "\n".join(
                f"{e.word} {e.freq} {e.pos}"
                for e in self.pipe.dictionary.to_entries()
            )
            + "\n",
            encoding="utf-8",
        )
        final_table = full_root / "table.json"
        final_table.write_text(
            json.dumps(self._table_data, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        (full_root / "sentences.txt").write_text(
            "\n".join(
                f"{sid}\t{text}" for sid, text in sorted(self._sentences.items())
            )
            + "\n",
            encoding="utf-8",
        )
        full_pipe = make_pipeline(
            full_paths,
            full_root / "snapshot",
            table_version=self._table_version,
        )
        full_pipe.run()
        return full_pipe

    def _perturb_table(self):
        data = copy.deepcopy(self._table_data)
        for head, mapping in data["trans_logprob"].items():
            tags = list(mapping)
            values = [mapping[t] for t in tags]
            data["trans_logprob"][head] = dict(
                zip(tags, values[1:] + values[:1])
            )
        return data

    def test_full_edit_sequence_matches_blank_full_run(self):
        self._sentences = dict(self.pipe.cache._texts)
        self._table_data = json.loads(
            self.paths["table"].read_text(encoding="utf-8")
        )
        self._table_version = 1

        invalidated = self.pipe.add_word("训练模型", 40, "n")
        self.assertEqual(invalidated, [])  # 当前样本无连续子串，保持确定性
        outcome_add = self.pipe.run()
        self.assertEqual(outcome_add.recomputed, [])
        self.assertEqual(len(outcome_add.reused), 5)

        affected = self.pipe.update_freq("学习", 300)
        self.assertEqual(affected, ["S3"])
        outcome_freq = self.pipe.run()
        self.assertEqual(outcome_freq.recomputed, ["S3"])
        self.assertNotIn("S3", outcome_freq.reused)

        affected_del = self.pipe.remove_word("电脑")
        self.assertEqual(affected_del, ["S2"])
        outcome_del = self.pipe.run()
        self.assertEqual(outcome_del.recomputed, ["S2"])

        table_v2_data = self._perturb_table()
        self._table_data = table_v2_data
        self._table_version = 2
        affected_table = self.pipe.replace_prob_table(
            ProbabilityTable(table_v2_data, version=2)
        )
        self.assertEqual(sorted(affected_table), ["S1", "S2", "S3", "S4", "S5"])
        outcome_table = self.pipe.run()
        self.assertEqual(len(outcome_table.recomputed), 5)
        self.assertEqual(outcome_table.reused, [])

        self._sentences["S5"] = "白猫在屋顶看月亮"
        self.paths["sentences"].write_text(
            "\n".join(
                f"{sid}\t{text}" for sid, text in sorted(self._sentences.items())
            )
            + "\n",
            encoding="utf-8",
        )
        changed = self.pipe.replace_sentence_text("S5", self._sentences["S5"])
        self.assertTrue(changed)
        outcome_text = self.pipe.run()
        self.assertEqual(outcome_text.recomputed, ["S5"])

        full_pipe = self._full_rebuild()
        self.assertEqual(comparable(self.pipe), comparable(full_pipe))
        self.assertEqual(failures(self.pipe), failures(full_pipe))
        self.assertEqual(
            list(comparable(self.pipe)), list(comparable(full_pipe))
        )

    def test_reused_records_are_not_recomputed(self):
        before = self.pipe.results_by_id()
        outcome = self.pipe.run()
        after = self.pipe.results_by_id()
        self.assertEqual(sorted(outcome.reused), ["S1", "S2", "S3", "S4", "S5"])
        self.assertEqual(outcome.recomputed, [])
        for sid in before:
            self.assertEqual(before[sid]["cache_key"], after[sid]["cache_key"])
            self.assertTrue(after[sid]["reused"])



    def test_batch_edit_merges_invalidation_set(self):
        pipe = make_pipeline(self.paths, self.root / "snap_batch")
        pipe.run()
        affected = pipe.batch_edit(
            [
                {"action": "update_freq", "word": "学习", "freq": 250},
                {"action": "update_freq", "word": "电脑", "freq": 120},
            ]
        )
        self.assertEqual(sorted(affected), ["S2", "S3"])
        outcome = pipe.run()
        self.assertEqual(sorted(outcome.recomputed), ["S2", "S3"])
        self.assertEqual(
            sorted(outcome.reused), ["S1", "S4", "S5"]
        )
        self.assertEqual(pipe.config.dict_change_mode, "entry")


if __name__ == "__main__":
    unittest.main()
