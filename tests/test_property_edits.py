import copy
import random

import pytest

from chaindiag.ledger import CacheStore
from chaindiag.pipeline import compare_reports, state_to_engine
from chaindiag.samplegen import default_bundle
from chaindiag.state import apply_command, apply_commands


def fresh_state():
    bundle = default_bundle()
    chain = copy.deepcopy(bundle["chain_e2"])
    return {
        "config": copy.deepcopy(bundle["config"]),
        "chain": chain,
        "operators": chain["operators"],
        "tokens": copy.deepcopy(bundle["tokens"]),
        "boundaries": copy.deepcopy(bundle["segments"]["boundaries"]),
    }


def random_valid_command(state: dict, rng: random.Random) -> dict:
    choices = ["replace", "param", "segments", "quant", "bits", "version"]
    if len(state["tokens"]) > 4:
        choices.append("delete")
    choice = rng.choice(choices)

    if choice == "replace":
        index = rng.randrange(len(state["tokens"]))
        return {"op": "replace_token", "token_uid": state["tokens"][index]["uid"], "token": rng.randrange(32)}
    if choice == "delete":
        index = rng.randrange(len(state["tokens"]))
        return {"op": "delete_token", "token_uid": state["tokens"][index]["uid"]}
    if choice == "param":
        op_seq = rng.randrange(len(state["operators"]))
        return {"op": "update_op_param", "op_seq": op_seq, "path": "scale_float",
                "value": rng.choice([0.8, 0.9375, 1.0, 1.0625, 1.125, 1.25])}
    if choice == "segments":
        count = len(state["tokens"])
        cuts = sorted(rng.sample(range(1, count), k=rng.randrange(1, min(4, count))))
        return {"op": "set_segments", "boundaries": [0, *cuts, count]}
    if choice == "quant":
        return {"op": "set_quant", "quant_tier": rng.choice(["q0", "q1", "q2", "q3"])}
    if choice == "bits":
        return {"op": "set_frac_bits", "frac_bits": rng.choice([8, 16, 24])}
    return {"op": "set_version", "exchange_version": rng.choice(["e1", "e2"])}


@pytest.mark.parametrize("seed", [39101, 39102, 39103, 39104, 39105, 39106, 39107, 39108])
def test_random_edit_sequences_match_blank_rebuild(seed):
    rng = random.Random(seed)
    incremental = fresh_state()
    store = CacheStore()
    state_to_engine(incremental, store=store, serial=True).run(with_width_reference=False)

    commands = []
    engine = None
    for _ in range(14):
        command = random_valid_command(incremental, rng)
        commands.append(command)
        apply_command(incremental, command)
        engine = state_to_engine(incremental, store=store, serial=True)
        report = engine.run(with_width_reference=False)
        assert report["status"] == "ok", report["diagnostics"]

    rerun = engine.run(with_width_reference=False)
    assert all(entry["action"] == "reused" for entry in rerun["reuse_ledger"])

    clean = fresh_state()
    apply_commands(clean, commands)
    full = state_to_engine(clean, store=CacheStore(), serial=True).run(with_width_reference=False)
    comparison = compare_reports(rerun, full)
    assert comparison["equal"], {"seed": seed, "mismatch": comparison["first_mismatch"], "commands": commands}
