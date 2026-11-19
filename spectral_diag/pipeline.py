"""End-to-end deterministic diagnostic pipeline."""

from __future__ import annotations

import platform
from dataclasses import replace
from typing import Any

import numpy as np

from . import __version__
from .attribution import compute_attribution, verify_attribution
from .canonical import array_digest, sha256_obj
from .config import SCHEMA_VERSION, RunConfig
from .errors import DiagnosticError, ErrorCode
from .injection import (
    locate_first_affected,
    matrix_singular_error,
    normalize_injections,
    prepare_scene,
)
from .matrix import quantize_xyz, solve_adaptation
from .scene import GeneratedScene, generate_scene

PIPELINE_VERSION = "spectral-diagnostic-pipeline-v1"


def version_stamp() -> dict[str, Any]:
    return {
        "application_version": __version__,
        "pipeline_version": PIPELINE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
    }


def _fallback_scene(config: RunConfig) -> GeneratedScene:
    safe_observer = config.observer
    if safe_observer not in {"cie1931_2", "cie1964_10"}:
        safe_observer = "cie1931_2"
    safe_density = config.reference_density_nm
    if not np.isfinite(safe_density) or safe_density <= 0 or safe_density > 1.0:
        safe_density = 0.5
    safe_config = replace(
        config,
        observer=safe_observer,
        reference_density_nm=safe_density,
        injections=(),
    )
    return generate_scene(safe_config)


def _clean_details(details: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in details.items() if not key.startswith("_")}


def _error_result(
    config: RunConfig,
    error: DiagnosticError,
    *,
    scene: GeneratedScene | None,
    location: dict[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "rejected",
        "version_stamp": version_stamp(),
        "config": config.to_dict(),
        "injections": [dict(value) for value in normalize_injections(config.injections)],
        "injection_sha256": sha256_obj(list(config.injections)),
        "error": {
            "code": str(error.code),
            "message": error.message,
            "details": _clean_details(dict(error.details)),
        },
        "first_affected": None if location is None else location["first_affected"],
        "location": location,
        "scene_sha256": None if scene is None else scene.sha256,
        "scene": None if scene is None else scene.payload,
        "matrix": None,
        "attribution": None,
        "materials": None,
        "integration": None,
    }
    payload["result_sha256"] = sha256_obj(payload)
    return payload


def execute_pipeline(config: RunConfig) -> dict[str, Any]:
    """Run generation, integration, inversion, attribution, and adaptation.

    Business failures are returned as a rejected report with a stable error
    code and first-affected observation.  They never contain an accepted matrix.
    """

    scene: GeneratedScene | None = None
    try:
        normalize_injections(config.injections)
        try:
            scene = prepare_scene(config)
        except DiagnosticError as error:
            # Scene-level injections attach their modified arrays privately so
            # locating the impact does not require guessing at the mutation.
            location_scene = scene
            source = error.details.get("_scene_source_xyz")
            target = error.details.get("_scene_target_xyz")
            if source is not None and target is not None:
                location = locate_first_affected(
                    error,
                    source_xyz=np.asarray(source, dtype=float),
                    target_xyz=np.asarray(target, dtype=float),
                    quant_step=config.quant_step,
                )
            else:
                fallback = _fallback_scene(config)
                location = locate_first_affected(
                    error,
                    source_xyz=fallback.source_xyz,
                    target_xyz=fallback.target_xyz,
                    quant_step=config.quant_step,
                )
            return _error_result(config, error, scene=location_scene, location=location)

        solution = solve_adaptation(
            scene.source_xyz,
            scene.target_xyz,
            cond_threshold=config.cond_threshold,
            allow_prune=True,
        )
        if not solution.accepted or solution.matrix is None:
            error = matrix_singular_error(solution, scene.source_xyz, scene.target_xyz)
            location = locate_first_affected(
                error,
                source_xyz=scene.source_xyz,
                target_xyz=scene.target_xyz,
                matrix=None,
                quant_step=config.quant_step,
            )
            payload = _error_result(config, error, scene=scene, location=location)
            payload["matrix"] = solution.to_dict()
            payload["matrix_rejected"] = True
            payload["result_sha256"] = sha256_obj({k: v for k, v in payload.items() if k != "result_sha256"})
            return payload

        quantized_source, quant_source_meta = quantize_xyz(
            scene.source_xyz,
            quant_range=config.quant_range,
            quant_bits=config.quant_bits,
        )
        quantized_target, quant_target_meta = quantize_xyz(
            scene.target_xyz,
            quant_range=config.quant_range,
            quant_bits=config.quant_bits,
        )

        attribution = compute_attribution(
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
            quantized_source_xyz=quantized_source,
            quantized_target_xyz=quantized_target,
            base_indices=solution.used_indices,
        )
        verify_attribution(attribution)

        adapted = scene.material_source_xyz @ solution.matrix
        material_residual = adapted - scene.material_target_xyz
        source_white = np.asarray(scene.payload["source"]["white_xyz"], dtype=float)
        camera_white = np.asarray(scene.payload["camera"]["white_xyz"], dtype=float)
        mapped_white = source_white @ solution.matrix
        white_residual = mapped_white - camera_white
        materials = []
        for index, material in enumerate(scene.payload["materials"]):
            materials.append(
                {
                    "index": index,
                    "name": material["name"],
                    "source_xyz": scene.material_source_xyz[index].tolist(),
                    "adapted_xyz": adapted[index].tolist(),
                    "target_truth_xyz": scene.material_target_xyz[index].tolist(),
                    "residual": material_residual[index].tolist(),
                    "residual_l2": float(np.linalg.norm(material_residual[index])),
                }
            )

        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "accepted" if solution.status == "accepted" else "accepted_after_pruning",
            "version_stamp": version_stamp(),
            "config": config.to_dict(),
            "injections": [dict(value) for value in config.injections],
            "injection_sha256": sha256_obj(list(config.injections)),
            "scene_sha256": scene.sha256,
            "scene": scene.payload,
            "reference_truth": scene.payload["reference_truth"],
            "integration": {
                "source": scene.payload["source"]["integration"],
                "camera": scene.payload["camera"]["integration"],
                "sample_source_xyz_sha256": array_digest(scene.source_xyz),
                "sample_target_xyz_sha256": array_digest(scene.target_xyz),
                "material_source_xyz_sha256": array_digest(scene.material_source_xyz),
                "material_target_xyz_sha256": array_digest(scene.material_target_xyz),
                "normalization": config.normalization,
                "density_nm": config.reference_density_nm,
            },
            "samples": {
                "source_xyz": scene.source_xyz.tolist(),
                "target_xyz": scene.target_xyz.tolist(),
                "quantized_source_xyz": quantized_source.tolist(),
                "quantized_target_xyz": quantized_target.tolist(),
                "quantization": {
                    "source": quant_source_meta,
                    "target": quant_target_meta,
                },
            },
            "matrix": solution.to_dict(),
            "white_point_mapping": {
                "source_white_xyz": source_white.tolist(),
                "camera_white_xyz": camera_white.tolist(),
                "mapped_source_white_xyz": mapped_white.tolist(),
                "residual": white_residual.tolist(),
                "residual_l2": float(np.linalg.norm(white_residual)),
            },
            "attribution": attribution,
            "materials": materials,
            "material_adapted_sha256": array_digest(adapted),
            "first_affected": None,
            "location": None,
            "error": None,
        }
        payload["result_sha256"] = sha256_obj({key: value for key, value in payload.items() if key != "result_sha256"})
        return payload
    except DiagnosticError as error:
        if scene is not None:
            location = locate_first_affected(
                error,
                source_xyz=scene.source_xyz,
                target_xyz=scene.target_xyz,
                quant_step=config.quant_step,
            )
        else:
            fallback = _fallback_scene(config)
            location = locate_first_affected(
                error,
                source_xyz=fallback.source_xyz,
                target_xyz=fallback.target_xyz,
                quant_step=config.quant_step,
            )
        return _error_result(config, error, scene=scene, location=location)


def rejected_result_to_error(payload: dict[str, Any]) -> DiagnosticError:
    error = payload.get("error") or {}
    return DiagnosticError(
        code=ErrorCode(error.get("code", ErrorCode.INPUT_INVALID)),
        message=str(error.get("message", "diagnostic pipeline rejected the run")),
        details=dict(error.get("details", {})),
    )
