"""Per-stage computation, real fault injection and atomic commits."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from .cache import content_fingerprint
from .config import Config, IMAGE_SIZE
from .dag import (
    DECODE_STAGE,
    ENCODE_STAGE,
    IMAGE_ENCODE_STAGE,
    INIT_STAGE,
    denoise_step_of,
)
from .errors import (
    E_IMAGE_SIZE_MISMATCH,
    E_INJECTED_PATCH_OUT_OF_RANGE,
    E_NOISE_ENTRY_TRUNCATED,
    E_NON_FINITE_LATENT,
    E_PATCH_OUT_OF_RANGE,
    WorkbenchError,
)
from .hashing import array_summary, fingerprint_array
from .injection import (
    Injection,
    apply_non_finite_latent,
    build_out_of_range_patch,
    matching_injections,
    truncate_noise_entry,
)
from .stages import (
    apply_patch,
    assert_finite,
    decode_latent,
    denoise_step,
    encode_initial_image,
    encode_prompt,
    initialize_latent,
    load_vocab,
    write_png,
)


def _summary(array: np.ndarray) -> dict:
    return array_summary(array)


def compute_stage(stage: str, config: Config, plan_entry: dict,
                  upstream_arrays: dict[str, np.ndarray],
                  injections: tuple[Injection, ...],
                  output_png: str | None) -> tuple[np.ndarray, dict, list[dict]]:
    """Compute one stage. Returns (array, manifest-extra, injection-hits).

    Raises WorkbenchError for injected or structural faults; the orchestrator
    converts those into isolated failure records.
    """
    hits: list[dict] = []
    node = plan_entry["node"]
    step = node.step
    extra: dict[str, Any] = {"step": step}

    if stage == ENCODE_STAGE:
        vocab = load_vocab(config.vocab_file)
        array, params = encode_prompt(config.prompt, vocab)
        extra["params"] = params

    elif stage == INIT_STAGE:
        embed = upstream_arrays[ENCODE_STAGE]
        array, params = initialize_latent(
            config, fingerprint_array(embed), tuple(config.latent_shape)
        )
        extra["params"] = params

    elif stage == IMAGE_ENCODE_STAGE:
        # Real injection: point the encoder at a wrong-sized image.
        image_path = config.initial_image
        for inj in matching_injections(injections, stage, None):
            if inj.fault_type == "image_size_mismatch":
                candidate = inj.params.get("wrong_image", config.initial_image)
                if candidate and not os.path.isabs(candidate) and config.source_path:
                    candidate = os.path.normpath(
                        os.path.join(os.path.dirname(config.source_path), candidate)
                    )
                image_path = candidate
                hits.append({"injection_id": inj.injection_id,
                             "code": E_IMAGE_SIZE_MISMATCH, "stage": stage,
                             "step": None})
        try:
            array, params = encode_initial_image(
                image_path, config.effective_fingerprint(), IMAGE_SIZE
            )
        except WorkbenchError as err:
            err.injection_hits = hits
            raise
        extra["params"] = params

    elif step is not None:  # denoise step
        embedding = upstream_arrays[ENCODE_STAGE]
        replaced = node.extra.get("replaced", False)
        if replaced:
            # img2img: early steps start from the encoded image latent.
            previous = upstream_arrays[IMAGE_ENCODE_STAGE].copy()
        else:
            deps = node.deps
            source = deps[1]
            previous = upstream_arrays[source].copy()

        # Real injection: truncate the noise entry for this step.
        noise_entry = list(config.noise_table[step - 1])
        for inj in matching_injections(injections, stage, step):
            if inj.fault_type == "noise_entry_truncated":
                noise_entry = truncate_noise_entry(noise_entry, inj)
                hits.append({"injection_id": inj.injection_id,
                             "code": E_NOISE_ENTRY_TRUNCATED, "stage": stage,
                             "step": step})
        if len(noise_entry) != len(config.noise_table[step - 1]):
            err = WorkbenchError(
                E_NOISE_ENTRY_TRUNCATED,
                "noise table entry truncated by injection",
                {"stage": stage, "step": step,
                 "actual": len(noise_entry),
                 "expected": len(config.noise_table[step - 1]),
                 "injection_id": hits[-1]["injection_id"]},
            )
            return None, extra, hits, err
        local_config = _config_with_noise_entry(config, step, noise_entry)
        array, params = denoise_step(step, previous, embedding, local_config)

        # Configured local latent patches (validated up-front).
        for patch in config.patches_for_step(step):
            array = apply_patch(array, patch)
            extra.setdefault("applied_patches", []).append({
                "step": step, "top": patch.top, "left": patch.left,
                "height": patch.height, "width": patch.width,
            })

        # Real injection: force an out-of-range patch for this step.
        for inj in matching_injections(injections, stage, step):
            if inj.fault_type == "patch_out_of_range":
                bad_patch = build_out_of_range_patch(inj, config.latent_shape[0])
                hits.append({"injection_id": inj.injection_id,
                             "code": E_INJECTED_PATCH_OUT_OF_RANGE, "stage": stage,
                             "step": step})
                err = _guard_patch_bounds(bad_patch, config, inj.injection_id)
                if err is not None:
                    return None, extra, hits, err

        # Real injection: corrupt a latent element, then finite-check.
        for inj in matching_injections(injections, stage, step):
            if inj.fault_type == "non_finite_latent":
                array = apply_non_finite_latent(array, inj)
                hits.append({"injection_id": inj.injection_id,
                             "code": E_NON_FINITE_LATENT, "stage": stage,
                             "step": step})
        try:
            assert_finite(array, stage, step,
                          hits[-1]["injection_id"] if hits and
                          hits[-1]["code"] == E_NON_FINITE_LATENT else None)
        except WorkbenchError as err:
            return None, extra, hits, err
        extra["params"] = params
        extra["replaced_from_image"] = replaced

    elif stage == DECODE_STAGE:
        final_stage = node.deps[1]
        array, params = decode_latent(upstream_arrays[final_stage], config)
        extra["params"] = params
        if output_png:
            write_png(array, output_png)
            extra["png_path"] = os.path.abspath(output_png)
    else:  # pragma: no cover
        raise WorkbenchError("E_CONFIG_INVALID", f"unknown stage {stage}")

    return array, extra, hits, None


def _config_with_noise_entry(config: Config, step: int, entry: list[float]) -> Config:
    import dataclasses

    table = [list(row) for row in config.noise_table]
    table[step - 1] = entry
    return dataclasses.replace(
        config, noise_table=tuple(tuple(row) for row in table)
    )


def _guard_patch_bounds(patch, config: Config, injection_id: str):
    _, lh, lw = config.latent_shape
    if (patch.top < 0 or patch.left < 0 or
            patch.top + patch.height > lh or patch.left + patch.width > lw):
        return WorkbenchError(
            E_INJECTED_PATCH_OUT_OF_RANGE,
            "injected patch rectangle is out of range",
            {"stage": "denoise", "step": patch.step, "top": patch.top,
             "left": patch.left, "height": patch.height, "width": patch.width,
             "latent_height": lh, "latent_width": lw,
             "injection_id": injection_id},
        )
    return None
