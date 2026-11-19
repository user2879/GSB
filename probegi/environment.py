"""Deterministic synthetic lighting environment.

The environment is fully procedural and seeded:

* a fixed number of directional lights (smooth, narrow Gaussian lobes so the
  finite quadrature can actually observe them),
* a sky-dome vertical gradient term,
* a fixed-seed procedural "noise" term built from a sum of seeded cosine
  lobes (band-limited, reproducible, no external assets).

All randomness comes from one ``numpy.random.Generator`` constructed from the
seed, consumed in a fixed order, so two constructions with the same seed
produce byte-identical radiance samples.
"""
import numpy as np

N_DIRECTIONAL_LIGHTS = 3
N_NOISE_LOBES = 8
N_EMITTERS = 6
EMITTER_REGION = 4.0


class LightingEnvironment:
    def __init__(self, seed: int):
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        # Directional lights: directions, colours, sharpness (fixed order).
        self.light_dirs = _normalised(rng.standard_normal((N_DIRECTIONAL_LIGHTS, 3)))
        self.light_colors = rng.uniform(2.0, 6.0, size=(N_DIRECTIONAL_LIGHTS, 3))
        self.light_sharpness = rng.uniform(8.0, 16.0, size=N_DIRECTIONAL_LIGHTS)
        # Sky gradient endpoints (z = -1 .. +1).
        self.sky_ground = rng.uniform(0.02, 0.10, size=3)
        self.sky_zenith = rng.uniform(0.20, 0.60, size=3)
        # Procedural noise: seeded cosine lobes.
        self.lobe_dirs = _normalised(rng.standard_normal((N_NOISE_LOBES, 3)))
        self.lobe_colors = rng.uniform(0.05, 0.50, size=(N_NOISE_LOBES, 3))
        self.lobe_sharpness = rng.uniform(2.0, 24.0, size=N_NOISE_LOBES)
        # Spatial noise: seeded glowing sphere emitters.  These make the
        # incident radiance field position-dependent, which is what gives
        # probe interpolation a non-trivial error landscape.
        self.emitter_pos = rng.uniform(-EMITTER_REGION, EMITTER_REGION,
                                       size=(N_EMITTERS, 3))
        self.emitter_colors = rng.uniform(1.0, 4.0, size=(N_EMITTERS, 3))
        self.emitter_radius = rng.uniform(0.3, 1.2, size=N_EMITTERS)

    def radiance(self, directions: np.ndarray) -> np.ndarray:
        """Incident radiance (W/sr/m^2-ish) for unit directions (N, 3) -> (N, 3)."""
        d = np.asarray(directions, dtype=np.float64)
        t = (d[:, 2:3] + 1.0) * 0.5
        out = self.sky_ground * (1.0 - t) + self.sky_zenith * t
        for i in range(N_DIRECTIONAL_LIGHTS):
            c = np.clip(d @ self.light_dirs[i], 0.0, None) ** self.light_sharpness[i]
            out = out + c[:, None] * self.light_colors[i]
        for i in range(N_NOISE_LOBES):
            c = np.clip(d @ self.lobe_dirs[i], 0.0, None) ** self.lobe_sharpness[i]
            out = out + c[:, None] * self.lobe_colors[i]
        return out

    def radiance_at(self, positions: np.ndarray,
                    directions: np.ndarray) -> np.ndarray:
        """Position-dependent incident radiance.

        positions (P, 3), directions (D, 3) -> (P, D, 3).  The distant terms
        (sky, directional lights, lobes) are shared by all probes; the
        seeded emitters vary with probe position.
        """
        p = np.asarray(positions, dtype=np.float64)
        d = np.asarray(directions, dtype=np.float64)
        base = self.radiance(d)  # (D, 3)
        out = np.broadcast_to(base, (p.shape[0],) + base.shape).copy()
        for i in range(N_EMITTERS):
            vec = self.emitter_pos[i][None, :] - p  # (P, 3)
            dist = np.linalg.norm(vec, axis=1, keepdims=True)
            to = vec / np.maximum(dist, 1e-12)  # (P, 3)
            cosang = np.einsum("pd,nd->pn", to, d)  # (P, D)
            half = np.arctan(self.emitter_radius[i]
                             / np.maximum(dist, 1e-12))  # (P, 1)
            cos_half = np.cos(2.0 * half)
            win = np.clip((cosang - cos_half)
                          / np.maximum(1.0 - cos_half, 1e-9), 0.0, 1.0) ** 2
            falloff = 1.0 / (1.0 + dist * dist)  # (P, 1)
            out += (win * falloff)[:, :, None] * self.emitter_colors[i]
        return out

    def describe(self) -> dict:
        return {
            "seed": self.seed,
            "n_directional_lights": N_DIRECTIONAL_LIGHTS,
            "n_noise_lobes": N_NOISE_LOBES,
            "sky_ground": self.sky_ground.tolist(),
            "sky_zenith": self.sky_zenith.tolist(),
            "n_emitters": N_EMITTERS,
        }


def _normalised(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=1, keepdims=True)
