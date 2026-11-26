"""Latent initialization and the per-step denoise kernel (business rule 4).

The kernel is deterministic integer math.  Every random draw comes from an
instance-local :class:`DeterministicRandom` keyed by content fingerprints and
the configured seed; Python's global random state is never touched.

Frame dependency: frame ``f`` initialization mixes the fused-vector fingerprint
of frame ``f-1`` (frame 0 uses a fixed genesis marker), so editing one frame's
conditions changes exactly that frame's init material and the next frame's
init material, which the ledger keys track precisely.  Within a frame every
step's output is the next step's input, preserving the sequential dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .budget import FrameAllocation
from .config import RenderConfig
from .fingerprints import fingerprint
from .fixedpoint import QuantTrace, q_div_round, q_mul, q_scale
from .fusion import FusionResult
from .rng import DeterministicRandom

GENESIS = "genesis-frame-minus-one"
LATENT_AMPLITUDE = 256       # Q amplitude bound for latent random walks
NOISE_AMPLITUDE = 48         # Q amplitude bound for noise table entries


@dataclass
class StepRecord:
    frame: int
    step: int
    kind: str                  # "denoise" or "fill"
    input_fp: str
    output: list[int]
    noise_entry: list[int]
    guidance_q: int
    trace: list[QuantTrace] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "frame": self.frame, "step": self.step, "kind": self.kind,
            "input_fp": self.input_fp, "output": self.output,
            "noise_entry": self.noise_entry, "guidance_q": self.guidance_q,
            "trace": [t.to_dict() for t in self.trace],
        }


@dataclass
class FrameLatent:
    frame: int
    init_material: str
    initial: list[int]
    steps: list[StepRecord]
    final: list[int]
    fill_basis: list[str]

    def to_dict(self) -> dict:
        return {
            "frame": self.frame, "init_material": self.init_material,
            "initial": self.initial,
            "steps": [s.to_dict() for s in self.steps],
            "final": self.final, "fill_basis": self.fill_basis,
        }


def init_material(frame: int, fusion: FusionResult,
                  prev_fusion_fp: str, config: RenderConfig) -> str:
    fusion_fp = fingerprint(fusion.vector, f"fusion-vector-f{frame}")
    config_fp = fingerprint(config.canonical_json(), "render-config")
    return "|".join([
        "latent-init",
        f"seed={config.noise_seed}",
        f"frame={frame}",
        f"fusion_fp={fusion_fp}",
        f"prev_fusion_fp={prev_fusion_fp}",
        f"config_fp={config_fp}",
    ])


def initial_latent(material: str, config: RenderConfig) -> list[int]:
    rng = DeterministicRandom(material)
    return [rng.next_signed(LATENT_AMPLITUDE)
            for _ in range(config.latent_dim)]


def _noise_table(material: str, config: RenderConfig,
                 length: int) -> list[list[int]]:
    rng = DeterministicRandom(material + "|noise-table")
    return [rng.noise_table(config.latent_dim, NOISE_AMPLITUDE)
            for _ in range(length)]


def _guidance_projection(cond_vector: list[int], index: int,
                         config: RenderConfig) -> int:
    """Map the fused condition onto latent slot ``index`` deterministically."""
    dim = config.condition_dim
    return cond_vector[index % dim]


def run_frame(frame: int, fusion: FusionResult,
              allocation: FrameAllocation, prev_fusion_fp: str,
              config: RenderConfig) -> FrameLatent:
    material = init_material(frame, fusion, prev_fusion_fp, config)
    latent = initial_latent(material, config)
    scale = q_scale(config.frac_bits)
    guidance = config.guidance_q()
    total_steps = allocation.demand
    noise = _noise_table(material, config, total_steps)
    records: list[StepRecord] = []
    fill_basis: list[str] = []
    current = list(latent)

    for step in range(total_steps):
        kind = "denoise" if step < allocation.allocated else "fill"
        input_fp = fingerprint(current, f"step-input-f{frame}-s{step}")
        entry = noise[step]
        nxt: list[int] = []
        trace: list[QuantTrace] = []
        # Fixed operation order per latent slot: noise first, then guidance
        # scaled condition, then accumulate onto the previous latent value.
        for slot in range(config.latent_dim):
            cond_raw = _guidance_projection(fusion.vector, slot, config)
            # Normalize the fused condition by the encoder feature scale so the
            # guided term stays in the latent Q domain instead of saturating.
            cond = q_div_round(cond_raw, config.condition_dim * 256,
                               config.qmode)
            if kind == "denoise":
                scaled, mul_trace = q_mul(guidance, cond, config.frac_bits,
                                          config.qmode,
                                          op=f"guidance-f{frame}-s{step}-{slot}")
            else:
                # Filled steps use a reduced, deterministic guidance (half of
                # the configured value) so the missing budget is recoverable
                # but clearly distinguished from real denoising.
                half_guidance = q_div_round(guidance, 2, "trunc")
                scaled, mul_trace = q_mul(half_guidance, cond,
                                          config.frac_bits, config.qmode,
                                          op=f"fill-guidance-f{frame}-s{step}-{slot}")
            trace.append(mul_trace)
            value = current[slot] + entry[slot] + scaled
            # Keep the walk bounded in the fixed point domain.
            bound = 16 * scale
            value = max(-bound, min(bound, value))
            nxt.append(value)
        if kind == "fill":
            fill_basis.append(
                f"frame={frame} step={step} fill: budget truncated by 1; "
                "half-guidance continuation; same chained latent input/output"
            )
        records.append(StepRecord(frame=frame, step=step, kind=kind,
                                  input_fp=input_fp, output=nxt,
                                  noise_entry=entry, guidance_q=guidance,
                                  trace=trace))
        current = nxt

    return FrameLatent(frame=frame, init_material=material,
                       initial=latent, steps=records, final=current,
                       fill_basis=fill_basis)
