"""Deterministic synthetic spectra, sample pairs, and material scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .canonical import canonical_bytes, sha256_bytes, sha256_obj
from .cmf import WAVELENGTHS, get_observer
from .config import RunConfig
from .spectra import (
    IntegrationConvention,
    dense_grid,
    integrate_reflectance,
)

GENERATOR_VERSION = "spectral-scene-v1"


@dataclass(frozen=True)
class GaussianMixture:
    """Smooth analytic spectrum used to generate both truth and 5 nm samples."""

    floor: float
    centers: tuple[float, ...]
    widths: tuple[float, ...]
    amplitudes: tuple[float, ...]
    lower: float = 0.005
    upper: float = 0.995

    def at(self, wavelengths: np.ndarray) -> np.ndarray:
        wavelengths = np.asarray(wavelengths, dtype=float)
        values = np.full_like(wavelengths, self.floor, dtype=float)
        for center, width, amplitude in zip(self.centers, self.widths, self.amplitudes):
            values += amplitude * np.exp(-0.5 * ((wavelengths - center) / width) ** 2)
        return np.clip(values, self.lower, self.upper)


@dataclass
class GeneratedScene:
    config: RunConfig
    payload: dict[str, Any]

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.payload)

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.canonical_bytes)

    @property
    def source_reflectances(self) -> np.ndarray:
        return np.asarray(self.payload["sample_reflectance_5nm"], dtype=float)

    @property
    def material_reflectances(self) -> np.ndarray:
        return np.asarray(self.payload["material_reflectance_5nm"], dtype=float)

    @property
    def source_xyz(self) -> np.ndarray:
        return np.asarray([pair["source"]["xyz"] for pair in self.payload["sample_pairs"]], dtype=float)

    @property
    def target_xyz(self) -> np.ndarray:
        return np.asarray([pair["target"]["xyz"] for pair in self.payload["sample_pairs"]], dtype=float)

    @property
    def material_source_xyz(self) -> np.ndarray:
        return np.asarray([material["source_xyz"] for material in self.payload["materials"]], dtype=float)

    @property
    def material_target_xyz(self) -> np.ndarray:
        return np.asarray([material["target_xyz"] for material in self.payload["materials"]], dtype=float)


def _random_mixture(
    rng: np.random.Generator,
    *,
    floor_range: tuple[float, float],
    center_range: tuple[float, float],
    width_range: tuple[float, float],
    amplitude_range: tuple[float, float],
    components: int,
    lower: float,
    upper: float,
) -> GaussianMixture:
    centers = sorted(float(value) for value in rng.uniform(*center_range, size=components))
    widths = tuple(float(value) for value in rng.uniform(*width_range, size=components))
    amplitudes = tuple(float(value) for value in rng.uniform(*amplitude_range, size=components))
    floor = float(rng.uniform(*floor_range))
    return GaussianMixture(
        floor=floor,
        centers=tuple(centers),
        widths=widths,
        amplitudes=amplitudes,
        lower=lower,
        upper=upper,
    )


def _make_source_spd(rng: np.random.Generator) -> GaussianMixture:
    # Cool, broad source illuminant.  The small seeded variation changes the
    # absolute white point without changing the scenario topology.
    phase = float(rng.uniform(-0.02, 0.02))
    return GaussianMixture(
        floor=0.24 + phase,
        centers=(451.0, 557.0, 644.0),
        widths=(38.0, 142.0, 105.0),
        amplitudes=(0.48, 1.12, 0.38),
        lower=0.02,
        upper=40.0,
    )


def _make_camera_spd(rng: np.random.Generator) -> GaussianMixture:
    # Warmer camera-side white point, deliberately far enough from the source
    # white point to make the inferred matrix observable.
    phase = float(rng.uniform(-0.02, 0.02))
    return GaussianMixture(
        floor=0.18 - phase,
        centers=(472.0, 604.0, 690.0),
        widths=(62.0, 168.0, 82.0),
        amplitudes=(0.54, 0.93, 0.29),
        lower=0.02,
        upper=40.0,
    )


def _reference_xyz(
    reflectance: np.ndarray,
    spd: np.ndarray,
    convention: IntegrationConvention,
    wavelengths: np.ndarray,
) -> Any:
    return integrate_reflectance(
        reflectance,
        spd,
        convention,
        wavelengths_nm=wavelengths,
    )


def generate_scene(config: RunConfig) -> GeneratedScene:
    """Generate a scene exclusively from the configured seed.

    Analytic continuous spectra are sampled at 5 nm for the declared material
    definition and at ``reference_density_nm`` for the truth integration.
    Consequently the truth does not come from re-integrating the 5 nm table.
    """

    rng = np.random.default_rng(config.seed)
    observer = get_observer(config.observer)
    base_wavelengths = np.asarray(WAVELENGTHS, dtype=float)
    reference_wavelengths = dense_grid(config.reference_density_nm)

    source_spd_model = _make_source_spd(rng)
    camera_spd_model = _make_camera_spd(rng)

    sample_models = [
        _random_mixture(
            rng,
            floor_range=(0.015, 0.15),
            center_range=(415.0, 705.0),
            width_range=(22.0, 72.0),
            amplitude_range=(0.13, 0.58),
            components=4,
            lower=0.01,
            upper=0.98,
        )
        for _ in range(config.sample_count)
    ]
    material_models = [
        _random_mixture(
            rng,
            floor_range=(0.01, 0.18),
            center_range=(410.0, 710.0),
            width_range=(20.0, 85.0),
            amplitude_range=(0.12, 0.62),
            components=4,
            lower=0.005,
            upper=0.985,
        )
        for _ in range(config.material_count)
    ]

    sample_5nm = np.vstack([model.at(base_wavelengths) for model in sample_models])
    material_5nm = np.vstack([model.at(base_wavelengths) for model in material_models])
    source_spd_5nm = source_spd_model.at(base_wavelengths)
    camera_spd_5nm = camera_spd_model.at(base_wavelengths)

    sample_dense = np.vstack([model.at(reference_wavelengths) for model in sample_models])
    material_dense = np.vstack([model.at(reference_wavelengths) for model in material_models])
    source_spd_dense = source_spd_model.at(reference_wavelengths)
    camera_spd_dense = camera_spd_model.at(reference_wavelengths)

    source_convention = IntegrationConvention(
        observer=config.observer,
        density_nm=config.reference_density_nm,
        quadrature=config.quadrature,
        normalization=config.normalization,
    )
    target_convention = IntegrationConvention(
        observer=config.observer,
        density_nm=config.reference_density_nm,
        quadrature=config.quadrature,
        normalization=config.normalization,
    )
    source_result = _reference_xyz(sample_dense, source_spd_dense, source_convention, reference_wavelengths)
    target_result = _reference_xyz(sample_dense, camera_spd_dense, target_convention, reference_wavelengths)
    material_source_result = _reference_xyz(
        material_dense, source_spd_dense, source_convention, reference_wavelengths
    )
    material_target_result = _reference_xyz(
        material_dense, camera_spd_dense, target_convention, reference_wavelengths
    )

    reference_metadata = {
        "method": "analytic_continuous_gaussian_mixtures+dense_trapezoid",
        "density_nm": config.reference_density_nm,
        "quadrature": config.quadrature,
        "observer": config.observer,
        "observer_name": observer.name,
        "normalization": config.normalization,
        "wavelength_range_nm": [380.0, 780.0],
        "material_definition_step_nm": 5.0,
        "dense_grid_points": int(reference_wavelengths.size),
        "truth_hash_sha256": sha256_obj(
            {
                "source": source_result.xyz.tolist(),
                "target": target_result.xyz.tolist(),
                "material_source": material_source_result.xyz.tolist(),
                "material_target": material_target_result.xyz.tolist(),
            }
        ),
    }

    sample_pairs: list[dict[str, Any]] = []
    for index, model in enumerate(sample_models):
        sample_pairs.append(
            {
                "index": index,
                "name": f"sample_{index:02d}",
                "source": {
                    "illuminant": "source",
                    "xyz": source_result.xyz[index].tolist(),
                    "truth": {
                        "source": "continuous_spectrum_reference_integral",
                        "density_nm": config.reference_density_nm,
                        "observer": config.observer,
                        "normalization": config.normalization,
                    },
                },
                "target": {
                    "illuminant": "camera",
                    "xyz": target_result.xyz[index].tolist(),
                    "truth": {
                        "source": "continuous_spectrum_reference_integral",
                        "density_nm": config.reference_density_nm,
                        "observer": config.observer,
                        "normalization": config.normalization,
                    },
                },
                "analytic_model": {
                    "floor": model.floor,
                    "centers_nm": list(model.centers),
                    "widths_nm": list(model.widths),
                    "amplitudes": list(model.amplitudes),
                    "lower": model.lower,
                    "upper": model.upper,
                },
            }
        )

    materials: list[dict[str, Any]] = []
    for index, model in enumerate(material_models):
        materials.append(
            {
                "index": index,
                "name": f"material_{index:02d}",
                "source_xyz": material_source_result.xyz[index].tolist(),
                "target_xyz": material_target_result.xyz[index].tolist(),
                "target_truth": {
                    "source": "continuous_spectrum_reference_integral",
                    "density_nm": config.reference_density_nm,
                    "observer": config.observer,
                    "normalization": config.normalization,
                },
                "analytic_model": {
                    "floor": model.floor,
                    "centers_nm": list(model.centers),
                    "widths_nm": list(model.widths),
                    "amplitudes": list(model.amplitudes),
                    "lower": model.lower,
                    "upper": model.upper,
                },
            }
        )

    payload: dict[str, Any] = {
        "schema_version": config.schema_version,
        "generator_version": GENERATOR_VERSION,
        "seed": config.seed,
        "wavelengths_nm": list(WAVELENGTHS),
        "reference_truth": reference_metadata,
        "config": config.to_dict(),
        "source": {
            "name": "source_illuminant",
            "spd_5nm": source_spd_5nm.tolist(),
            "analytic_model": {
                "floor": source_spd_model.floor,
                "centers_nm": list(source_spd_model.centers),
                "widths_nm": list(source_spd_model.widths),
                "amplitudes": list(source_spd_model.amplitudes),
                "lower": source_spd_model.lower,
                "upper": source_spd_model.upper,
            },
            "spd_truth_density_sha256": sha256_obj(source_spd_dense.tolist()),
            "white_xyz": source_result.white_xyz.tolist(),
            "integration": source_result.to_metadata(),
        },
        "camera": {
            "name": "camera_illuminant",
            "spd_5nm": camera_spd_5nm.tolist(),
            "analytic_model": {
                "floor": camera_spd_model.floor,
                "centers_nm": list(camera_spd_model.centers),
                "widths_nm": list(camera_spd_model.widths),
                "amplitudes": list(camera_spd_model.amplitudes),
                "lower": camera_spd_model.lower,
                "upper": camera_spd_model.upper,
            },
            "spd_truth_density_sha256": sha256_obj(camera_spd_dense.tolist()),
            "white_xyz": target_result.white_xyz.tolist(),
            "integration": target_result.to_metadata(),
        },
        "sample_reflectance_5nm": sample_5nm.tolist(),
        "material_reflectance_5nm": material_5nm.tolist(),
        "sample_pairs": sample_pairs,
        "materials": materials,
    }
    return GeneratedScene(config=config, payload=payload)
