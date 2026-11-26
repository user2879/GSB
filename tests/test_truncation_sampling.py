from decchain.config import DecodeConfig
from decchain.kernel import run_kernel
from decchain.truncation import truncate
from decchain.sampling import sample
from decchain.rng import DeterministicRng, derive_seed


TOKENS = [100, 101, 102, 103, 104]
LOGPROBS = [0.0, -0.4, -1.1, -2.0, -3.1]


def _prep(config):
    kr = run_kernel(TOKENS, LOGPROBS, 1.0, config)
    tr = truncate(kr, config)
    return kr, tr


def test_deterministic_ordering_and_topk(config):
    kr, tr = _prep(config)
    assert tr.sorted_tokens == sorted(
        TOKENS, key=lambda t: (-kr.quant[t], t))
    assert tr.topk_tokens == tr.sorted_tokens[: config.top_k]


def test_nucleus_keeps_minimum_prefix(config):
    kr, tr = _prep(config)
    assert tr.kept_tokens
    assert tr.kept_total >= tr.target_int - tr.half_step
    # Kept prefix is contiguous in the deterministic ordering.
    assert tr.kept_tokens == tr.sorted_tokens[: len(tr.kept_tokens)]


def test_boundary_diagnostics_present(config):
    kr, tr = _prep(config)
    assert tr.boundaries
    boundary = tr.boundaries[0]
    assert boundary.attribution in (
        "TEMP_ROUNDING", "ACCUM_ORDER", "BOTH", "NONE")
    assert boundary.ref_int_scaled >= 0


def test_temperature_moves_boundary(config):
    cold_kr = run_kernel(TOKENS, LOGPROBS, 0.5, config)
    hot_kr = run_kernel(TOKENS, LOGPROBS, 2.5, config)
    cold = truncate(cold_kr, config)
    hot = truncate(hot_kr, config)
    assert cold.kept_tokens != hot.kept_tokens or \
        cold.kept_quant != hot.kept_quant


def test_sampling_is_deterministic_and_bounded(config):
    kr, tr = _prep(config)
    cfg_fp = config.fingerprint()
    first = sample("R001", 1, 1, "a" * 64, cfg_fp, kr, tr)
    second = sample("R001", 1, 1, "a" * 64, cfg_fp, kr, tr)
    assert first.selected_token == second.selected_token
    assert first.rng_unit == second.rng_unit
    assert first.selected_token in tr.kept_tokens
    lo, hi = first.cumulative_intervals[first.selected_token]
    assert lo <= first.rng_target < hi


def test_step_index_changes_rng_stream(config):
    kr, tr = _prep(config)
    cfg_fp = config.fingerprint()
    a = sample("R001", 1, 1, "a" * 64, cfg_fp, kr, tr)
    b = sample("R001", 1, 2, "a" * 64, cfg_fp, kr, tr)
    assert a.seed != b.seed


def test_rng_has_no_global_state():
    rng_a = DeterministicRng("ab" * 16)
    rng_b = DeterministicRng("ab" * 16)
    assert [rng_a.next_unit() for _ in range(8)] == \
           [rng_b.next_unit() for _ in range(8)]
    seed = derive_seed("fp", 1, 1, "cfg")
    assert isinstance(seed, str) and len(seed) == 64
