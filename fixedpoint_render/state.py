"""Editable render state and deterministic command application.

The state is a plain JSON-serializable snapshot of every input plus the current
config overrides.  Commands mutate a deep copy in a fixed order so an arbitrary
command sequence always converges on the same state.
"""
from __future__ import annotations

import copy
import json
import os

from PIL import Image

from .config import CONDITION_TYPES, RenderConfig
from .errors import E_COMMAND_TARGET, ErrorBundle, RenderError
from .fixedpoint import parse_decimal_q


class RenderState:
    def __init__(self, config: RenderConfig, frame_count: int,
                 prompts: dict, images: dict, motions: dict):
        self.config = config
        self.frame_count = frame_count
        # Store image pixel arrays (mode, size, bytes) so state is serializable
        # and identical across save/load cycles.
        self.prompts = {f: list(terms) for f, terms in prompts.items()}
        self.image_payloads = {}
        for frame, image in images.items():
            self.image_payloads[frame] = {
                "mode": image.mode, "size": list(image.size),
                "bytes": list(image.tobytes()),
            }
        self.motions = {f: copy.deepcopy(data) for f, data in motions.items()}
        self.frame_budgets: dict[int, int] = {}
        self.image_present = {f: True for f in range(frame_count)}

    # -- (de)serialization -------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "frame_count": self.frame_count,
            "prompts": {str(k): v for k, v in self.prompts.items()},
            "images": {str(k): v for k, v in self.image_payloads.items()},
            "motions": {str(k): v for k, v in self.motions.items()},
            "frame_budgets": {str(k): v for k, v in self.frame_budgets.items()},
            "image_present": {str(k): v for k, v in self.image_present.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RenderState":
        config_dict = data["config"]
        config = RenderConfig(
            vocab=tuple(config_dict["vocab"]),
            width=config_dict["width"], height=config_dict["height"],
            pixel_mode=config_dict["pixel_mode"],
            channels=config_dict["channels"],
            tile_rows=config_dict["tile_rows"],
            condition_dim=config_dict["condition_dim"],
            latent_dim=config_dict["latent_dim"],
            frac_bits=config_dict["frac_bits"], qmode=config_dict["qmode"],
            weights=dict(config_dict["weights"]),
            guidance=config_dict["guidance"],
            per_frame_budget=config_dict["per_frame_budget"],
            global_step_budget=config_dict["global_step_budget"],
            noise_seed=config_dict["noise_seed"],
            max_workers=config_dict["max_workers"],
            snapshot_versions=config_dict.get("snapshot_versions", 1),
        )
        state = cls.__new__(cls)
        state.config = config
        state.frame_count = data["frame_count"]
        state.prompts = {int(k): list(v) for k, v in data["prompts"].items()}
        state.image_payloads = {int(k): v for k, v in data["images"].items()}
        state.motions = {int(k): copy.deepcopy(v)
                         for k, v in data["motions"].items()}
        state.frame_budgets = {int(k): v
                               for k, v in data["frame_budgets"].items()}
        state.image_present = {int(k): v
                               for k, v in data["image_present"].items()}
        return state

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, sort_keys=True, indent=1)

    @classmethod
    def load(cls, path: str) -> "RenderState":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def image(self, frame: int):
        if not self.image_present.get(frame, False):
            return None
        payload = self.image_payloads[frame]
        return Image.frombytes(payload["mode"], tuple(payload["size"]),
                               bytes(payload["bytes"]))

    # -- commands ----------------------------------------------------------
    def apply(self, command: dict) -> None:
        verb = command["verb"]
        frame = command.get("frame")
        if verb == "set_prompt":
            terms = [t.strip() for t in str(command["prompt"]).replace(",", " ").split()
                     if t.strip()]
            vocab = set(self.config.vocab)
            bad = [t for t in terms if t not in vocab]
            if bad:
                raise ErrorBundle([RenderError(
                    E_COMMAND_TARGET, "commands", f"frame={frame}.prompt",
                    f"terms outside vocab: {bad}")])
            self.prompts[frame] = terms
        elif verb == "set_image":
            payload = command["image"]
            self.image_payloads[frame] = {
                "mode": payload["mode"], "size": list(payload["size"]),
                "bytes": list(payload["bytes"]),
            }
            self.image_present[frame] = True
        elif verb == "delete_image":
            self.image_present[frame] = False
        elif verb == "set_motion":
            self.motions[frame] = copy.deepcopy(command["motion"])
        elif verb == "set_weight":
            condition = command["condition"]
            if condition not in CONDITION_TYPES:
                raise ErrorBundle([RenderError(
                    E_COMMAND_TARGET, "commands", "weight.condition",
                    f"bad condition {condition!r}")])
            value = str(command["value"])
            parse_decimal_q(value, self.config.frac_bits, self.config.qmode)
            self.config = RenderConfig(
                **{**self.config.to_dict(),
                   "weights": {**self.config.weights, condition: value}})
        elif verb == "set_frame_budget":
            self.frame_budgets[frame] = int(command["budget"])
        elif verb == "set_qmode":
            self.config = RenderConfig(
                **{**self.config.to_dict(), "qmode": command["qmode"]})
        elif verb == "set_frac_bits":
            self.config = RenderConfig(
                **{**self.config.to_dict(),
                   "frac_bits": int(command["frac_bits"])})
        else:  # validated earlier, defensive guard
            raise ErrorBundle([RenderError(
                E_COMMAND_TARGET, "commands", verb, "unsupported verb")])
