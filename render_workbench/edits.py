"""Batch edit plans applied as transactions over config + prompt inputs.

Each batch is validated against the current state before anything is mutated.
If any edit fails validation (for example an out-of-range patch), the whole
batch is rejected via a structured failure record and the pre-batch state is
preserved.
"""

import copy
import json
import os

from .config import _validate_patches, _normalize_patch
from .errors import (
    E_CONFIG_INVALID,
    E_INPUT_MISSING,
    E_PATCH_OUT_OF_RANGE,
    E_TOKEN_OUT_OF_VOCAB,
)
from .png_codec import decode_png


def load_plan(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _err(code, message, **extra):
    record = {"code": code, "message": message}
    record.update(extra)
    return record


class EditPlanner:
    def __init__(self, cfg, prompts, base_dir):
        self.cfg = cfg
        self.prompts = {prompt["id"]: prompt for prompt in prompts}
        self.base_dir = base_dir

    # ------------------------------------------------------------- transaction
    def apply_batch(self, batch):
        """Validate then apply one batch. Returns a list of failure records."""
        snapshot_cfg = copy.deepcopy(self.cfg)
        snapshot_prompts = copy.deepcopy(self.prompts)
        errors = []
        if not isinstance(batch, dict) or not isinstance(batch.get("edits"), list):
            return [_err(E_CONFIG_INVALID, "batch must contain an edits list")]
        for index, edit in enumerate(batch["edits"]):
            try:
                self._apply_edit(edit)
            except _EditError as exc:
                errors.append(exc.record)
                break
        if errors:
            self.cfg = snapshot_cfg
            self.prompts = snapshot_prompts
        return errors

    def prompt_list(self):
        return [self.prompts[pid] for pid in sorted(self.prompts)]

    # ------------------------------------------------------------------ edits
    def _apply_edit(self, edit):
        if not isinstance(edit, dict) or not isinstance(edit.get("op"), str):
            raise _EditError(_err(E_CONFIG_INVALID, "edit needs a string op"))
        op = edit["op"]
        handler = {
            "replace_prompt_tokens": self._replace_tokens,
            "set_seed": self._set_seed,
            "set_guidance_scale": self._set_guidance,
            "set_denoise_steps": self._set_steps,
            "add_tier": self._add_tier,
            "remove_tier": self._remove_tier,
            "replace_init_image": self._replace_init_image,
            "add_patch": self._add_patch,
            "remove_patch": self._remove_patch,
        }.get(op)
        if handler is None:
            raise _EditError(_err(E_CONFIG_INVALID, "unknown op %s" % op))
        handler(edit)

    def _require_prompt(self, edit):
        prompt_id = edit.get("prompt_id")
        if not isinstance(prompt_id, str) or prompt_id not in self.prompts:
            raise _EditError(_err(E_CONFIG_INVALID,
                                  "unknown prompt_id %r" % prompt_id))
        return self.prompts[prompt_id]

    def _replace_tokens(self, edit):
        prompt = self._require_prompt(edit)
        tokens = edit.get("tokens")
        if not isinstance(tokens, list) or not tokens:
            raise _EditError(_err(E_CONFIG_INVALID,
                                  "tokens must be a non-empty list"))
        vocab_set = set(self.cfg["vocab"])
        for token in tokens:
            if not isinstance(token, str) or token not in vocab_set:
                raise _EditError(_err(
                    E_TOKEN_OUT_OF_VOCAB,
                    "token outside vocab: %r" % token,
                    prompt_id=prompt["id"]))
        prompt["tokens"] = list(tokens)

    def _set_seed(self, edit):
        seed = edit.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool) \
                or not (self.cfg["seed_min"] <= seed <= self.cfg["seed_max"]):
            raise _EditError(_err(E_CONFIG_INVALID,
                                  "seed out of allowed range"))
        self.cfg["seed"] = seed

    def _set_guidance(self, edit):
        value = edit.get("guidance_scale")
        if not isinstance(value, (int, float)) or isinstance(value, bool) \
                or value < 0:
            raise _EditError(_err(E_CONFIG_INVALID,
                                  "guidance_scale must be non-negative"))
        self.cfg["guidance_scale"] = float(value)

    def _set_steps(self, edit):
        steps = edit.get("denoise_steps")
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise _EditError(_err(E_CONFIG_INVALID,
                                  "denoise_steps must be a positive integer"))
        replace_steps = self.cfg["img2img_replace_steps"]
        if replace_steps > steps:
            raise _EditError(_err(E_CONFIG_INVALID,
                                  "denoise_steps must be >= "
                                  "img2img_replace_steps"))
        for tier in self.cfg["tiers"]:
            full_noise = tier.get("full_noise", tier["noise"])
            full_guidance = tier.get("full_guidance", tier["guidance"])
            tier["full_noise"] = list(full_noise)
            tier["full_guidance"] = list(full_guidance)
            if len(full_noise) < steps or len(full_guidance) < steps:
                raise _EditError(_err(
                    E_CONFIG_INVALID,
                    "tier %s noise/guidance tables cannot be extended to "
                    "length %d without add_tier data" % (tier["id"], steps)))
            tier["noise"] = list(full_noise)[:steps]
            tier["guidance"] = list(full_guidance)[:steps]
        self.cfg["denoise_steps"] = steps

    def _add_tier(self, edit):
        tier_id = edit.get("id")
        width = edit.get("width")
        height = edit.get("height")
        noise = edit.get("noise")
        guidance = edit.get("guidance")
        steps = self.cfg["denoise_steps"]
        if not isinstance(tier_id, str) or any(t["id"] == tier_id
                                               for t in self.cfg["tiers"]):
            raise _EditError(_err(E_CONFIG_INVALID,
                                  "tier id missing or duplicate"))
        if not isinstance(width, int) or isinstance(width, bool) \
                or not isinstance(height, int) or isinstance(height, bool) \
                or width <= 0 or height <= 0:
            raise _EditError(_err(E_CONFIG_INVALID, "invalid tier geometry"))
        if self.cfg["tiers"]:
            last = self.cfg["tiers"][-1]
            if not (width > last["width"] and height > last["height"]):
                raise _EditError(_err(E_CONFIG_INVALID,
                                      "new tier must be strictly larger"))
        if not isinstance(noise, list) or not isinstance(guidance, list) \
                or len(noise) != steps or len(guidance) != steps:
            raise _EditError(_err(E_CONFIG_INVALID,
                                  "tier tables must have length denoise_steps"))
        if any(not isinstance(v, (int, float)) or isinstance(v, bool)
               for v in noise + guidance):
            raise _EditError(_err(E_CONFIG_INVALID,
                                  "table entries must be numeric"))
        table_noise = [float(v) for v in noise]
        table_guidance = [float(v) for v in guidance]
        self.cfg["tiers"].append({
            "id": tier_id, "width": width, "height": height,
            "noise": table_noise,
            "guidance": table_guidance,
            "full_noise": list(table_noise),
            "full_guidance": list(table_guidance),
        })

    def _remove_tier(self, edit):
        tier_id = edit.get("id")
        before = len(self.cfg["tiers"])
        self.cfg["tiers"] = [t for t in self.cfg["tiers"] if t["id"] != tier_id]
        if len(self.cfg["tiers"]) == before:
            raise _EditError(_err(E_CONFIG_INVALID,
                                  "unknown tier %s" % tier_id))

    def _replace_init_image(self, edit):
        prompt = self._require_prompt(edit)
        rel_path = edit.get("init_image")
        if rel_path is None:
            prompt.update(init_image=None, init_image_path=None,
                          init_pixels=None)
            return
        if not isinstance(rel_path, str):
            raise _EditError(_err(E_CONFIG_INVALID,
                                  "init_image must be a path or null"))
        path = rel_path if os.path.isabs(rel_path) \
            else os.path.join(self.base_dir, rel_path)
        if not os.path.exists(path):
            raise _EditError(_err(E_INPUT_MISSING,
                                  "init image not found: %s" % rel_path,
                                  prompt_id=prompt["id"]))
        try:
            with open(path, "rb") as handle:
                pixels = decode_png(handle.read())
        except Exception as exc:  # noqa: BLE001
            raise _EditError(_err(E_CONFIG_INVALID,
                                  "init image unreadable: %s" % exc,
                                  prompt_id=prompt["id"]))
        prompt.update(init_image=rel_path, init_image_path=path,
                      init_pixels=pixels)

    def _add_patch(self, edit):
        prompt = self._require_prompt(edit)
        patch = edit.get("patch")
        errors = _validate_patches([patch], self.cfg, prompt["id"])
        if errors:
            code = errors[0]["code"]
            if code != E_PATCH_OUT_OF_RANGE:
                code = E_CONFIG_INVALID
            raise _EditError(dict(errors[0], code=code))
        normalized = _normalize_patch(patch)
        prompt.setdefault("patches", [])
        prompt["patches"] = [
            existing for existing in prompt["patches"]
            if not _same_region(existing, normalized)
        ]
        prompt["patches"].append(normalized)

    def _remove_patch(self, edit):
        prompt = self._require_prompt(edit)
        region = edit.get("region")
        step = edit.get("step")
        if not isinstance(region, dict):
            raise _EditError(_err(E_CONFIG_INVALID, "region object required"))
        try:
            x, y = int(region["x"]), int(region["y"])
            width, height = int(region["width"]), int(region["height"])
        except (KeyError, TypeError, ValueError):
            raise _EditError(_err(E_CONFIG_INVALID, "invalid region geometry"))
        if not isinstance(step, int) or isinstance(step, bool):
            raise _EditError(_err(E_CONFIG_INVALID, "step must be integer"))
        before = len(prompt.get("patches", []))
        prompt["patches"] = [
            patch for patch in prompt.get("patches", [])
            if not (patch["step"] == step and patch["x"] == x
                    and patch["y"] == y and patch["width"] == width
                    and patch["height"] == height)
        ]
        if len(prompt["patches"]) == before:
            raise _EditError(_err(E_CONFIG_INVALID,
                                  "no matching patch to remove"))


def _same_region(a, b):
    return (a["step"] == b["step"] and a["x"] == b["x"] and a["y"] == b["y"]
            and a["width"] == b["width"] and a["height"] == b["height"])


class _EditError(Exception):
    def __init__(self, record):
        super().__init__(record["message"])
        self.record = record
