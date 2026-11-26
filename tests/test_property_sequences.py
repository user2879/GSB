from __future__ import annotations

import random

import pytest

from topic_ledger.replay import replay_equivalence
from topic_ledger.samplegen import build_sample_bundle


def _random_sequence(seed: int) -> list[dict]:
    rng = random.Random(seed)
    bundle = build_sample_bundle()
    revisions = {doc["doc_id"]: 1 for doc in bundle["documents"]}
    doc_shards = {doc["doc_id"]: doc["shard_id"] for doc in bundle["documents"]}
    commands: list[dict] = [{"command_id": "start", "op": "run_sampling", "batch_id": "start"}]

    for step in range(8):
        command_id = f"seq-{seed}-{step}"
        choice = rng.randrange(7)
        if choice == 0:
            doc_id = rng.choice(sorted(revisions))
            revisions[doc_id] += 1
            terms = [rng.randrange(len(bundle["vocab"])) for _ in range(rng.randrange(5, 16))]
            commands.append(
                {
                    "command_id": command_id,
                    "op": "upsert_document",
                    "document": {
                        "doc_id": doc_id,
                        "shard_id": doc_shards[doc_id],
                        "revision": revisions[doc_id],
                        "term_ids": terms,
                    },
                }
            )
        elif choice == 1:
            commands.append({"command_id": command_id, "op": "run_sampling", "batch_id": f"run-{step}"})
        elif choice == 2:
            commands.append(
                {
                    "command_id": command_id,
                    "op": "set_config",
                    "patch": {"sample_point_budget": rng.choice([1, 20, 100_000])},
                }
            )
        elif choice == 3:
            commands.append({"command_id": command_id, "op": "advance_clock_ms", "milliseconds": 300})
        elif choice == 4:
            commands.append({"command_id": command_id, "op": "recover_leases"})
        elif choice == 5:
            commands.append(
                {
                    "command_id": command_id,
                    "op": "set_node_lease_limit",
                    "node_id": rng.choice(["node-a", "node-b", "node-c"]),
                    "lease_limit": rng.choice([1, 2, 3]),
                }
            )
        else:
            commands.append({"command_id": command_id, "op": "snapshot", "label": command_id})

    commands.extend(
        [
            {"command_id": "settle-budget", "op": "set_config", "patch": {"sample_point_budget": 100_000}},
            {"command_id": "settle-clock", "op": "advance_clock_ms", "milliseconds": 1000},
            {"command_id": "settle-recover", "op": "recover_leases"},
            {"command_id": "settle-run", "op": "run_sampling", "batch_id": "settle"},
        ]
    )
    return commands


@pytest.mark.parametrize("seed", list(range(12)))
def test_random_command_sequences_replay_identically(tmp_path, seed) -> None:
    bundle = build_sample_bundle()
    commands = _random_sequence(seed)
    result = replay_equivalence(
        bundle,
        commands,
        work_root=tmp_path / f"property-{seed}",
        concurrency_left=1,
        concurrency_right=4,
    )
    assert result["equal"] is True, "\n".join(result["comparison"]["differences"][:10])
