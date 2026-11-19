"""Dependency-scoped convention changes and cache invalidation closure."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np

from .attribution import compute_attribution, verify_attribution
from .canonical import array_digest, sha256_obj
from .config import RunConfig
from .errors import DiagnosticError, ErrorCode
from .matrix import MatrixSolution, solve_adaptation
from .scene import GaussianMixture, GeneratedScene, generate_scene
from .spectra import IntegrationConvention, dense_grid, integrate_reflectance

LEAVES = (
    "scene_inputs",
    "injections",
    "observer",
    "normalization",
    "reference_density",
    "quadrature",
    "quantization",
    "attribution_settings",
    "solve_settings",
)


TOPOLOGY = (
    "scene",
    "integration.source_weights",
    "integration.camera_weights",
    "samples.source_xyz",
    "samples.target_xyz",
    "whitepoints",
    "materials.source_xyz",
    "materials.target_xyz",
    "matrix",
    "attribution",
    "materials.adapted_xyz",
)


DEPENDENCIES: dict[str, set[str]] = {
    "scene": {"scene_inputs", "injections"},
    "integration.source_weights": {
        "scene",
        "observer",
        "normalization",
        "reference_density",
        "quadrature",
    },
    "integration.camera_weights": {
        "scene",
        "observer",
        "normalization",
        "reference_density",
        "quadrature",
    },
    "samples.source_xyz": {"integration.source_weights", "scene"},
    "samples.target_xyz": {"integration.camera_weights", "scene"},
    "whitepoints": {"integration.source_weights", "integration.camera_weights"},
    "materials.source_xyz": {"integration.source_weights", "scene"},
    "materials.target_xyz": {"integration.camera_weights", "scene"},
    "matrix": {"samples.source_xyz", "samples.target_xyz", "solve_settings"},
    "attribution": {
        "matrix",
        "samples.source_xyz",
        "samples.target_xyz",
        "quantization",
        "attribution_settings",
    },
    "materials.adapted_xyz": {"materials.source_xyz", "matrix"},
}


def dependency_closure(changed_leaves: set[str]) -> set[str]:
    closure = set(changed_leaves)
    changed = True
    while changed:
        changed = False
        for artifact in TOPOLOGY:
            if artifact in closure:
                continue
            if DEPENDENCIES[artifact] & closure:
                closure.add(artifact)
                changed = True
    return {artifact for artifact in closure if artifact in TOPOLOGY}


class ConventionSession:
    """Run convention-scoped artifacts with an explicit dependency cache."""

    def __init__(self, config: RunConfig) -> None:
        if config.injections:
            # Closure experiments must not mix arbitrary mutations with cache
            # identity tests.  Injection runs still work through the pipeline.
            raise DiagnosticError(
                code=ErrorCode.CONFIG_INVALID,
                message="convention sessions require a non-injected scene",
                details={"injections": list(config.injections)},
            )
        self.config = config
        self._scene: GeneratedScene | None = None
        self._cache: dict[str, dict[str, dict[str, Any]]] = {artifact: {} for artifact in TOPOLOGY}
        self._previous_leaves: dict[str, Any] | None = None
        self.values: dict[str, Any] = {}
        self.hashes: dict[str, str] = {}
        self.last_report: dict[str, Any] | None = None
        self._first_hashes_by_context: dict[str, dict[str, str]] = {}

    def _leaves(self, config: RunConfig) -> dict[str, Any]:
        return {
            "scene_inputs": {
                "seed": config.seed,
                "sample_count": config.sample_count,
                "material_count": config.material_count,
                "schema_version": config.schema_version,
            },
            "injections": list(config.injections),
            "observer": config.observer,
            "normalization": config.normalization,
            "reference_density": config.reference_density_nm,
            "quadrature": config.quadrature,
            "quantization": [config.quant_bits, list(config.quant_range)],
            "attribution_settings": [config.attribution_density_nm, config.attribution_drop_count],
            "solve_settings": config.cond_threshold,
        }

    def _changed_leaves(self, leaves: dict[str, Any]) -> set[str]:
        if self._previous_leaves is None:
            return set(LEAVES)
        return {key for key in LEAVES if leaves[key] != self._previous_leaves[key]}

    def _context(self, artifact: str, leaves: dict[str, Any]) -> dict[str, Any]:
        context: dict[str, Any] = {}
        for dependency in sorted(DEPENDENCIES[artifact]):
            if dependency in LEAVES:
                context[dependency] = leaves[dependency]
            else:
                context[dependency] = self.hashes[dependency]
        return context

    def _get_or_compute(
        self,
        artifact: str,
        leaves: dict[str, Any],
        compute: Callable[[], tuple[Any, str]],
        closure: set[str],
        report: dict[str, Any],
    ) -> Any:
        context = self._context(artifact, leaves)
        key = sha256_obj(context)
        cache = self._cache[artifact]
        if key in cache:
            value = cache[key]["value"]
            digest = cache[key]["hash"]
            if artifact in closure:
                report["cache_hits"].append(artifact)
        else:
            value, digest = compute()
            cache[key] = {"value": value, "hash": digest, "context": context}
            report["recomputed"].append(artifact)
            if artifact not in closure:
                report["outside_closure"].append(artifact)
        self.values[artifact] = value
        self.hashes[artifact] = digest
        return value

    def run(
        self,
        *,
        observer: str | None = None,
        normalization: str | None = None,
        reference_density_nm: float | None = None,
        quadrature: str | None = None,
        quant_bits: int | None = None,
        quant_range: tuple[float, float] | None = None,
        cond_threshold: float | None = None,
        attribution_density_nm: float | None = None,
        attribution_drop_count: int | None = None,
    ) -> dict[str, Any]:
        changes: dict[str, Any] = {}
        if observer is not None:
            changes["observer"] = observer
        if normalization is not None:
            changes["normalization"] = normalization
        if reference_density_nm is not None:
            changes["reference_density_nm"] = reference_density_nm
        if quadrature is not None:
            changes["quadrature"] = quadrature
        if quant_bits is not None:
            changes["quant_bits"] = quant_bits
        if quant_range is not None:
            changes["quant_range"] = quant_range
        if cond_threshold is not None:
            changes["cond_threshold"] = cond_threshold
        if attribution_density_nm is not None:
            changes["attribution_density_nm"] = attribution_density_nm
        if attribution_drop_count is not None:
            changes["attribution_drop_count"] = attribution_drop_count
        self.config = replace(self.config, **changes)
        config = self.config
        leaves = self._leaves(config)
        changed_leaves = self._changed_leaves(leaves)
        closure = dependency_closure(changed_leaves)
        report: dict[str, Any] = {
            "changed_leaves": sorted(changed_leaves),
            "dependency_closure": [artifact for artifact in TOPOLOGY if artifact in closure],
            "recomputed": [],
            "cache_hits": [],
            "outside_closure": [],
        }

        def scene_compute() -> tuple[GeneratedScene, str]:
            scene = generate_scene(config)
            return scene, scene.sha256

        scene = self._get_or_compute("scene", leaves, scene_compute, closure, report)
        assert isinstance(scene, GeneratedScene)
        self._scene = scene

        source_convention = IntegrationConvention(
            observer=config.observer,
            density_nm=config.reference_density_nm,
            quadrature=config.quadrature,
            normalization=config.normalization,
        )
        reference_grid = dense_grid(config.reference_density_nm)

        def mixture(data: dict[str, Any]) -> GaussianMixture:
            return GaussianMixture(
                floor=float(data["floor"]),
                centers=tuple(float(value) for value in data["centers_nm"]),
                widths=tuple(float(value) for value in data["widths_nm"]),
                amplitudes=tuple(float(value) for value in data["amplitudes"]),
                lower=float(data["lower"]),
                upper=float(data["upper"]),
            )

        source_spd = mixture(scene.payload["source"]["analytic_model"]).at(reference_grid)
        camera_spd = mixture(scene.payload["camera"]["analytic_model"]).at(reference_grid)
        sample_spectra = np.vstack(
            [mixture(item["analytic_model"]).at(reference_grid) for item in scene.payload["sample_pairs"]]
        )
        material_spectra = np.vstack(
            [mixture(item["analytic_model"]).at(reference_grid) for item in scene.payload["materials"]]
        )

        def weights_compute(spd: np.ndarray) -> Callable[[], tuple[Any, str]]:
            def compute() -> tuple[Any, str]:
                result = integrate_reflectance(
                    np.ones((1, reference_grid.size), dtype=float),
                    spd,
                    source_convention,
                    wavelengths_nm=reference_grid,
                )
                value = {"metadata": result.to_metadata(), "weights": result.weights.tolist()}
                return value, sha256_obj(value)

            return compute

        self._get_or_compute(
            "integration.source_weights", leaves, weights_compute(source_spd), closure, report
        )
        self._get_or_compute(
            "integration.camera_weights", leaves, weights_compute(camera_spd), closure, report
        )

        def sample_compute(group: str) -> Callable[[], tuple[Any, str]]:
            def compute() -> tuple[Any, str]:
                spd = source_spd if group == "source" else camera_spd
                result = integrate_reflectance(
                    sample_spectra,
                    spd,
                    source_convention,
                    wavelengths_nm=reference_grid,
                )
                value = {"xyz": result.xyz.tolist(), "metadata": result.to_metadata()}
                return value, array_digest(result.xyz)

            return compute

        source_samples = self._get_or_compute(
            "samples.source_xyz", leaves, sample_compute("source"), closure, report
        )
        target_samples = self._get_or_compute(
            "samples.target_xyz", leaves, sample_compute("camera"), closure, report
        )

        def whitepoints_compute() -> tuple[Any, str]:
            value = {
                "source": source_samples["metadata"]["white_xyz"],
                "camera": target_samples["metadata"]["white_xyz"],
            }
            return value, sha256_obj(value)

        self._get_or_compute("whitepoints", leaves, whitepoints_compute, closure, report)

        def material_compute(group: str) -> Callable[[], tuple[Any, str]]:
            def compute() -> tuple[Any, str]:
                spd = source_spd if group == "source" else camera_spd
                result = integrate_reflectance(
                    material_spectra,
                    spd,
                    source_convention,
                    wavelengths_nm=reference_grid,
                )
                value = {"xyz": result.xyz.tolist(), "metadata": result.to_metadata()}
                return value, array_digest(result.xyz)

            return compute

        material_source = self._get_or_compute(
            "materials.source_xyz", leaves, material_compute("source"), closure, report
        )
        material_target = self._get_or_compute(
            "materials.target_xyz", leaves, material_compute("camera"), closure, report
        )

        def matrix_compute() -> tuple[Any, str]:
            solution = solve_adaptation(
                np.asarray(source_samples["xyz"], dtype=float),
                np.asarray(target_samples["xyz"], dtype=float),
                cond_threshold=config.cond_threshold,
                allow_prune=True,
            )
            if not solution.accepted:
                raise DiagnosticError(
                    code=ErrorCode.MATRIX_SINGULAR,
                    message="convention run produced a singular matrix",
                    details=solution.to_dict(),
                )
            value = solution.to_dict()
            return value, value["solution_sha256"]

        matrix_value = self._get_or_compute("matrix", leaves, matrix_compute, closure, report)

        def attribution_compute() -> tuple[Any, str]:
            solution = MatrixSolution(
                status=str(matrix_value["status"]),
                matrix=np.asarray(matrix_value["matrix"], dtype=float),
                used_indices=tuple(matrix_value["used_indices"]),
                condition_number=matrix_value["condition_number"],
                matrix_condition_number=matrix_value["matrix_condition_number"],
                residuals=np.asarray(matrix_value["residuals"], dtype=float),
                residual_componentwise=np.asarray(matrix_value["residual_componentwise"], dtype=float),
                residual_norm=np.asarray(matrix_value["residual_norm"], dtype=float),
                residual_rmse=float(matrix_value["residual_rmse"]),
                residual_max_abs=float(matrix_value["residual_max_abs"]),
                diagnostics=solve_adaptation(
                    np.asarray(source_samples["xyz"], dtype=float),
                    np.asarray(target_samples["xyz"], dtype=float),
                    cond_threshold=config.cond_threshold,
                    allow_prune=True,
                ).diagnostics,
                prune_trace=(),
            )
            ledger = compute_attribution(
                scene,
                solution,
                observer=config.observer,
                normalization=config.normalization,
                quadrature=config.quadrature,
                cond_threshold=config.cond_threshold,
                quant_bits=config.quant_bits,
                quant_range=config.quant_range,
                integration_density_nm=config.attribution_density_nm,
                distribution_drop_count=config.attribution_drop_count,
                base_indices=solution.used_indices,
                reference_source_xyz=np.asarray(source_samples["xyz"], dtype=float),
                reference_target_xyz=np.asarray(target_samples["xyz"], dtype=float),
                material_source_xyz=np.asarray(material_source["xyz"], dtype=float),
                material_target_xyz=np.asarray(material_target["xyz"], dtype=float),
                reference_density_nm=config.reference_density_nm,
            )
            verify_attribution(ledger)
            return ledger, ledger["attribution_sha256"]

        self._get_or_compute("attribution", leaves, attribution_compute, closure, report)

        def adapted_compute() -> tuple[Any, str]:
            matrix = np.asarray(matrix_value["matrix"], dtype=float)
            values = np.asarray(material_source["xyz"], dtype=float) @ matrix
            value = {"xyz": values.tolist()}
            return value, array_digest(values)

        self._get_or_compute("materials.adapted_xyz", leaves, adapted_compute, closure, report)

        self._previous_leaves = leaves
        post_cache_closure = [artifact for artifact in report["dependency_closure"] if artifact not in report["cache_hits"]]
        report["invalidation_closure"] = post_cache_closure
        report["recomputed"] = [artifact for artifact in TOPOLOGY if artifact in report["recomputed"]]
        report["cache_hits"] = [artifact for artifact in TOPOLOGY if artifact in report["cache_hits"]]
        report["closure_reconciled"] = report["recomputed"] == post_cache_closure and not report["outside_closure"]
        if not report["closure_reconciled"]:
            raise DiagnosticError(
                code=ErrorCode.ATTRIB_MISMATCH,
                message="actual recomputation set differs from the invalidation closure",
                details=report,
            )
        report["hashes"] = {artifact: self.hashes[artifact] for artifact in TOPOLOGY}
        report["observer"] = config.observer
        report["normalization"] = config.normalization
        report["reference_density_nm"] = config.reference_density_nm
        self.last_report = report
        return {
            "report": report,
            "scene_sha256": self.hashes["scene"],
            "sample_source_xyz_sha256": self.hashes["samples.source_xyz"],
            "sample_target_xyz_sha256": self.hashes["samples.target_xyz"],
            "matrix_sha256": self.hashes["matrix"],
            "attribution_sha256": self.hashes["attribution"],
            "material_adapted_sha256": self.hashes["materials.adapted_xyz"],
            "matrix": matrix_value,
            "attribution": self.values["attribution"],
            "materials": [
                {
                    "index": index,
                    "source_xyz": material_source["xyz"][index],
                    "adapted_xyz": self.values["materials.adapted_xyz"]["xyz"][index],
                }
                for index in range(len(material_source["xyz"]))
            ],
        }
