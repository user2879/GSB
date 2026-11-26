from __future__ import annotations

import unittest

from gsb.checkpoint import CheckpointManager
from gsb.model import CheckpointSpec, EventSpec
from gsb.params import get_parameter_set
from tests.helpers import make_session, make_shard


def _manager(session):
    return CheckpointManager(
        session,
        lambda shard_id, version: get_parameter_set(
            session.shard(shard_id), version
        ).digest,
    )


class CheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.shard = make_shard(parameter_version=2)
        self.checkpoints = (
            CheckpointSpec("v1", self.shard.shard_id, 1, "1" * 64),
            CheckpointSpec("v2", self.shard.shard_id, 2, "2" * 64),
        )
        self.session = make_session((self.shard,), checkpoints=self.checkpoints)

    def test_commit_and_rollback_restore_version_and_generation(self) -> None:
        manager = _manager(self.session)
        manager.record_batch(
            sequence=1, shard_id=self.shard.shard_id, batch_id="b1", loss_digest="a" * 64
        )
        committed = manager.commit("v2", seq=2)
        self.assertTrue(committed.committed)
        self.assertEqual(committed.parameter_version, 2)
        self.assertEqual(committed.batch_sequences, ((1, "b1", "a" * 64),))
        rolled = manager.rollback("v1")
        self.assertEqual(manager.current_version(self.shard.shard_id), 1)
        self.assertEqual(manager.generation(self.shard.shard_id), 1)
        self.assertEqual(rolled.generation, 1)
        self.assertEqual(
            manager.current_parameter_digest(self.shard.shard_id),
            get_parameter_set(self.shard, 1).digest,
        )

    def test_batch_recording_order_does_not_change_checkpoint_bytes(self) -> None:
        first = _manager(self.session)
        second = _manager(self.session)
        first.record_batch(
            sequence=1, shard_id=self.shard.shard_id, batch_id="b1", loss_digest="a" * 64
        )
        first.record_batch(
            sequence=2, shard_id=self.shard.shard_id, batch_id="b2", loss_digest="b" * 64
        )
        second.record_batch(
            sequence=2, shard_id=self.shard.shard_id, batch_id="b2", loss_digest="b" * 64
        )
        second.record_batch(
            sequence=1, shard_id=self.shard.shard_id, batch_id="b1", loss_digest="a" * 64
        )
        event = EventSpec(3, "commit", self.shard.shard_id, checkpoint_id="v2")
        first.apply_event(event)
        second.apply_event(event)
        self.assertEqual(first.snapshot(), second.snapshot())


if __name__ == "__main__":
    unittest.main()

