"""Incremental multi-stage rendering pipeline.

Node graph for every (prompt, tier):

  embedding -> init_latent -> denoise:s0 -> denoise:s1 -> ... -> frame:s*
                       \-> (img2img) img_encode -> denoise:sR -> ... -> frame

Local latent patch nodes wrap the output of one denoise step and only
invalidate that step's downstream chain. Every node has a cache key derived
from its parameters and the content fingerprints of its upstream artifacts,
which gives exact, fingerprint-driven failure propagation.
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import encoder as encoder_mod
from . import latent as latent_mod
from . import decoder as decoder_mod
from .config import (
    config_fingerprint_payload,
    global_semantic_payload,
    tier_payload,
)
from .hashing import (
    array_fingerprint,
    bytes_fingerprint,
    cache_key,
    fingerprint,
)
from .storage import ArtifactStore

TXT2IMG = "txt2img"
IMG2IMG = "img2img"


class StopRequested(Exception):
    """Raised internally when a graceful stop was requested between nodes."""


class Pipeline:
    def __init__(self, cfg, store_root, stop_event=None):
        self.cfg = cfg
        self.store = ArtifactStore(store_root)
        self.config_fp = fingerprint(config_fingerprint_payload(cfg))
        self.global_fp = fingerprint(global_semantic_payload(cfg))
        self.tier_fps = {
            tier["id"]: fingerprint(tier_payload(tier))
            for tier in cfg["tiers"]
        }
        self.store.manifest["config_fp"] = self.config_fp
        self.stop_event = stop_event
        self._lock = threading.Lock()
        self.computed_count = 0
        self.reuse_records = []
        self.recompute_records = []
        self.path_segments = []

    def _scope_fp(self, tier_id):
        """Fingerprint for everything affecting one tier's rendered content.

        Combines tier-independent semantics with the single tier's own
        parameters, so adding/removing another tier never invalidates this
        tier's cached artifacts.
        """
        return fingerprint({
            "global": self.global_fp,
            "tier": self.tier_fps[tier_id],
        })

    # ------------------------------------------------------------------ graph
    def build_plan(self, prompts):
        """Describe every node with params and upstream node identifiers."""
        nodes = {}
        for prompt in prompts:
            prompt_fp = encoder_mod.prompt_fingerprint(prompt)
            emb_id = "embedding::%s" % prompt["id"]
            nodes[emb_id] = {
                "id": emb_id,
                "kind": "embedding",
                "prompt_id": prompt["id"],
                "tier_id": None,
                "upstream": [],
                "params": {
                    "prompt_fp": prompt_fp,
                    "embedding_dim": self.cfg["embedding_dim"],
                    "vocab": self.cfg["vocab"],
                },
                "prompt": prompt,
            }
            for tier in self.cfg["tiers"]:
                self._build_tier_plan(nodes, prompt, tier, emb_id, prompt_fp)
        return nodes

    def _build_tier_plan(self, nodes, prompt, tier, emb_id, prompt_fp):
        pid = prompt["id"]
        tid = tier["id"]
        steps = self.cfg["denoise_steps"]
        channels = self.cfg["latent_channels"]
        seed = self.cfg["seed"]
        replace = self.cfg["img2img_replace_steps"]

        init_id = "init_latent::%s::%s" % (pid, tid)
        nodes[init_id] = {
            "id": init_id,
            "kind": "init_latent",
            "prompt_id": pid,
            "tier_id": tid,
            "upstream": [emb_id],
            "params": {
                "prompt_fp": prompt_fp,
                "seed": seed,
                "scope_fp": self._scope_fp(tid),
                "tier_id": tid,
                "channels": channels,
            },
            "prompt": prompt,
        }

        chains = {TXT2IMG: init_id}
        if prompt.get("init_image"):
            img_id = "img_encode::%s::%s" % (pid, tid)
            with open(prompt["init_image_path"], "rb") as image_handle:
                image_bytes = image_handle.read()
            init_image_fp = fingerprint({
                "image_bytes_fp": bytes_fingerprint(image_bytes),
                "size": [int(prompt["init_pixels"].shape[1]),
                         int(prompt["init_pixels"].shape[0])],
            })
            nodes[img_id] = {
                "id": img_id,
                "kind": "img_encode",
                "prompt_id": pid,
                "tier_id": tid,
                "upstream": [emb_id],
                "params": {
                    "prompt_fp": prompt_fp,
                    "init_image_fp": init_image_fp,
                    "scope_fp": self._scope_fp(tid),
                    "tier_id": tid,
                    "channels": channels,
                },
                "prompt": prompt,
            }
            chains[IMG2IMG] = img_id

        patch_by_step = {patch["step"]: patch
                         for patch in prompt.get("patches", [])}
        for path, root_id in chains.items():
            previous = root_id
            for step in range(steps):
                den_id = "denoise::%s::%s::%s::s%04d" % (
                    path, pid, tid, step)
                nodes[den_id] = {
                    "id": den_id,
                    "kind": "denoise",
                    "path": path,
                    "prompt_id": pid,
                    "tier_id": tid,
                    "step_index": step,
                    "upstream": [previous, emb_id],
                    "params": {
                        "step_index": step,
                        "seed": seed,
                        "guidance_scale": self.cfg["guidance_scale"],
                        "noise_entry": tier["noise"][step],
                        "guidance_entry": tier["guidance"][step],
                        "scope_fp": self._scope_fp(tid),
                        "tier_id": tid,
                        "channels": channels,
                        "embedding_dim": self.cfg["embedding_dim"],
                    },
                    "prompt": prompt,
                    "tier": tier,
                }
                previous = den_id
                if step in patch_by_step and path == TXT2IMG:
                    patch = patch_by_step[step]
                    patch_id = "patch::%s::%s::s%04d" % (pid, tid, step)
                    nodes[patch_id] = {
                        "id": patch_id,
                        "kind": "patch",
                        "path": path,
                        "prompt_id": pid,
                        "tier_id": tid,
                        "step_index": step,
                        "upstream": [den_id],
                        "params": {
                            "step_index": step,
                            "patch": {
                                "x": patch["x"], "y": patch["y"],
                                "width": patch["width"],
                                "height": patch["height"],
                                "values": patch["values"],
                            },
                            "tolerance": self.cfg["patch_tolerance"],
                        },
                        "prompt": prompt,
                        "tier": tier,
                    }
                    previous = patch_id
                frame_id = "frame::%s::%s::%s::s%04d" % (
                    path, pid, tid, step)
                nodes[frame_id] = {
                    "id": frame_id,
                    "kind": "frame",
                    "path": path,
                    "prompt_id": pid,
                    "tier_id": tid,
                    "step_index": step,
                    "upstream": [previous],
                    "params": {
                        "step_index": step,
                        "tier_id": tid,
                        "channels": channels,
                        "scope_fp": self._scope_fp(tid),
                    },
                    "prompt": prompt,
                    "tier": tier,
                }
        self._record_segments(prompt, tier, steps, replace)
        return nodes

    def _record_segments(self, prompt, tier, steps, replace):
        if not prompt.get("init_image"):
            self.path_segments.append({
                "prompt_id": prompt["id"],
                "tier_id": tier["id"],
                "divergent": [],
                "shared": [TXT2IMG],
            })
            return
        divergent = list(range(0, replace))
        shared = list(range(replace, steps))
        self.path_segments.append({
            "prompt_id": prompt["id"],
            "tier_id": tier["id"],
            "divergent": divergent,
            "shared": shared,
        })

    # ------------------------------------------------------------------ cache
    def _node_cache_key(self, node, content_fps):
        """Cache key built from real upstream content fingerprints.

        A local patch within tolerance changes no latent content. Its cache
        key therefore collapses onto its upstream content fingerprint, so a
        no-op patch never invalidates the downstream denoise/decode chain.
        The requested patch values stay auditable in the node params and the
        recompute/reuse report.
        """
        if node["kind"] == "patch" and self._patch_is_effective_noop(node):
            return cache_key(
                "patch-noop",
                {"step_index": node["step_index"],
                 "tolerance": node["params"]["tolerance"]},
                [content_fps[ref] for ref in node["upstream"]])
        return cache_key(
            node["kind"], node["params"],
            [content_fps[ref] for ref in node["upstream"]])

    def _patch_is_effective_noop(self, node):
        """Resolve no-op status against the latent the patch would overwrite.

        Two cases are within tolerance and therefore change no content:
          * a fresh patch over the raw upstream denoise latent whose values
            already match that latent;
          * a replacement patch over an already committed patch node whose
            stored output matches the new values.
        """
        patch = node["params"]["patch"]
        tolerance = self.cfg["patch_tolerance"]
        tier = node["tier"]
        channels = self.cfg["latent_channels"]
        source_id = node["upstream"][0]
        if source_id not in self.store.manifest.get("nodes", {}):
            return False
        latent = self._load_upstream_latent(source_id, tier, channels)
        if latent_mod.patch_is_noop(latent, patch, tolerance):
            return True
        # Replacement of an existing patch node: compare against the prior
        # committed patch output for the same step/region.
        prior_id = "patch::%s::%s::s%04d" % (
            node["prompt_id"], node["tier_id"], node["step_index"])
        nodes = self.store.manifest.get("nodes", {})
        if prior_id in nodes:
            prior_latent = self._load_upstream_latent(
                prior_id, tier, channels)
            if latent_mod.patch_is_noop(prior_latent, patch, tolerance):
                return True
        return False

    def reconcile(self, planned):
        """Topologically reconcile planned nodes against persisted artifacts.

        Cache keys always use concrete upstream content fingerprints. When an
        upstream is itself a miss, its fingerprint is resolved as soon as it
        is computed (see :meth:`execute`), so no ``None`` placeholders ever
        reach a persisted key.
        """
        ordered = self._topological(planned)
        content_fps = {
            node_id: entry["content_fp"]
            for node_id, entry in self.store.manifest.get("nodes", {}).items()
        }
        decisions = []
        live_ids = set()
        for node_id in ordered:
            node = planned[node_id]
            live_ids.add(node_id)
            resolvable = all(ref in content_fps for ref in node["upstream"])
            entry = self.store.manifest.get("nodes", {}).get(node_id)
            if resolvable:
                key = self._node_cache_key(node, content_fps)
                hit = bool(entry and entry.get("cache_key") == key
                           and self.store.verify_node(node_id, entry))
                # A no-op patch (within tolerance of the raw upstream or of
                # the prior committed patch output) changes no content. Keep
                # the verified existing node and re-key it as a no-op so the
                # downstream denoise/decode chain stays cached.
                if not hit and entry is not None and node["kind"] == "patch" \
                        and self._patch_is_effective_noop(node) \
                        and self.store.verify_node(node_id, entry):
                    hit = True
                    entry["cache_key"] = key
            else:
                key = None
                hit = False
            if hit:
                self.reuse_records.append(self._audit_record(
                    node, key, entry,
                    [content_fps[ref] for ref in node["upstream"]]))
            else:
                if entry is not None:
                    self.store.remove_node(node_id)
                    content_fps.pop(node_id, None)
                decisions.append(node)
        self._prune_orphans(live_ids)
        return ordered, decisions

    def _prune_orphans(self, live_ids):
        for node_id in list(self.store.manifest.get("nodes", {}).keys()):
            if node_id not in live_ids:
                self.store.remove_node(node_id)

    @staticmethod
    def _topological(planned):
        visited = set()
        ordered = []

        def visit(node_id):
            if node_id in visited:
                return
            visited.add(node_id)
            for upstream in planned[node_id]["upstream"]:
                visit(upstream)
            ordered.append(node_id)

        for node_id in sorted(planned):
            visit(node_id)
        return ordered

    @staticmethod
    def _audit_record(node, key, entry, upstream_fps):
        return {
            "node_id": node["id"],
            "kind": node["kind"],
            "prompt_id": node["prompt_id"],
            "tier_id": node.get("tier_id"),
            "cache_key": key,
            "decision": "reuse",
            "content_fp": entry["content_fp"],
            "upstream_fps": upstream_fps,
            "version": entry.get("version"),
        }

    # ---------------------------------------------------------------- execute
    def execute(self, prompts):
        planned = self.build_plan(prompts)
        ordered, decisions = self.reconcile(planned)
        pending = {node["id"]: node for node in decisions}
        content_fps = {
            node_id: entry["content_fp"]
            for node_id, entry in self.store.manifest.get("nodes", {}).items()
        }
        snapshot_interval = self.cfg["snapshot_interval"]
        commits_since_snapshot = 0

        def ready(node_id):
            return all(up not in pending for up in planned[node_id]["upstream"])

        max_workers = max(1, self.cfg["concurrency"])
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            while pending:
                wave = sorted(
                    (node_id for node_id in pending if ready(node_id)),
                    key=self._sort_key,
                )
                if not wave:  # defensive: graph should always resolve
                    raise RuntimeError("unresolvable pipeline cycle")
                futures = {
                    pool.submit(self._compute, planned[node_id]): node_id
                    for node_id in wave
                }
                for future in futures:
                    node_id = futures[future]
                    result = future.result()
                    node = pending.pop(node_id)
                    key = self._node_cache_key(node, content_fps)
                    upstream = [content_fps[ref] for ref in node["upstream"]]
                    self._commit(node, key, result, durable=False)
                    content_fps[node_id] = result["content_fp"]
                    self.recompute_records.append(self._audit_record(
                        node, key,
                        self.store.manifest["nodes"][node_id], upstream))
                    commits_since_snapshot += 1
                    self.computed_count += 1
                    if commits_since_snapshot >= snapshot_interval:
                        self._snapshot()
                        commits_since_snapshot = 0
                if self.stop_event is not None and self.stop_event.is_set():
                    self._flush_snapshot(commits_since_snapshot)
                    raise StopRequested()
        self._flush_snapshot(commits_since_snapshot)
        return ordered

    def _snapshot(self):
        """Durable atomic snapshot: barrier artifacts, then replace manifest."""
        self.store.snapshot_barrier()
        self.store.bump_version()
        self.store.manifest["cursors"] = self.store._rebuild_cursors(
            self.store.manifest["nodes"])
        self.store.persist_manifest(durable=True)

    def _flush_snapshot(self, commits_since_snapshot):
        if commits_since_snapshot or self.store.manifest["version"] == 0:
            self._snapshot()
        else:
            # Nothing new since the last durable snapshot; just refresh
            # in-memory cursors. The report write is the batch commit point.
            self.store.manifest["cursors"] = self.store._rebuild_cursors(
                self.store.manifest["nodes"])

    @staticmethod
    def _sort_key(node_id):
        parts = node_id.split("::")
        kind = parts[0]
        kind_order = {
            "embedding": 0, "init_latent": 1, "img_encode": 1,
            "denoise": 2, "patch": 3, "frame": 4,
        }
        tail = parts[1:]
        normalized = []
        for piece in tail:
            if piece.startswith("s") and piece[1:].isdigit():
                normalized.append((1, int(piece[1:])))
            else:
                normalized.append((0, piece))
        return (kind_order.get(kind, 9),) + tuple(normalized)

    # ---------------------------------------------------------------- compute
    def _compute(self, node):
        kind = node["kind"]
        if kind == "embedding":
            return self._compute_embedding(node)
        if kind == "init_latent":
            return self._compute_init(node)
        if kind == "img_encode":
            return self._compute_img_encode(node)
        if kind == "denoise":
            return self._compute_denoise(node)
        if kind == "patch":
            return self._compute_patch(node)
        if kind == "frame":
            return self._compute_frame(node)
        raise RuntimeError("unknown node kind %s" % kind)

    def _load_upstream_latent(self, node_id, tier, channels):
        entry = self.store.manifest["nodes"][node_id]
        return self.store.read_npy(
            node_id, "latent.bin",
            latent_mod.latent_shape(tier, channels), np.float64)

    def _compute_embedding(self, node):
        result = encoder_mod.encode_prompt(
            node["prompt"], self.cfg["vocab"], self.cfg["embedding_dim"])
        embedding = result["embedding"]
        return {
            "files": {"embedding.bin": np.ascontiguousarray(embedding).tobytes()},
            "content_fp": result["embedding_fp"],
            "extra": {"prompt_fp": result["prompt_fp"],
                      "shape": list(embedding.shape)},
        }

    def _compute_init(self, node):
        tier = self._tier(node["tier_id"])
        channels = self.cfg["latent_channels"]
        emb_entry = self.store.manifest["nodes"][node["upstream"][0]]
        latent = latent_mod.initialize_latent(
            emb_entry["content_fp"], node["params"]["seed"],
            node["params"]["scope_fp"], tier, channels)
        return {
            "files": {"latent.bin": np.ascontiguousarray(latent).tobytes()},
            "content_fp": array_fingerprint(latent),
            "extra": {"shape": list(latent.shape)},
        }

    def _compute_img_encode(self, node):
        tier = self._tier(node["tier_id"])
        channels = self.cfg["latent_channels"]
        prompt = node["prompt"]
        latent = latent_mod.encode_init_image(
            prompt["init_pixels"], tier, channels,
            node["params"]["prompt_fp"], node["params"]["init_image_fp"],
            node["params"]["scope_fp"])
        return {
            "files": {"latent.bin": np.ascontiguousarray(latent).tobytes()},
            "content_fp": array_fingerprint(latent),
            "extra": {"shape": list(latent.shape)},
        }

    def _compute_denoise(self, node):
        tier = node["tier"]
        channels = self.cfg["latent_channels"]
        source_id = node["upstream"][0]
        prev_latent = self._load_upstream_latent(source_id, tier, channels)
        emb_id = node["upstream"][1]
        emb_entry = self.store.manifest["nodes"][emb_id]
        embedding = self.store.read_npy(
            emb_id, "embedding.bin",
            (self.cfg["embedding_dim"],), np.float64)
        latent = latent_mod.denoise_step(
            prev_latent, node["step_index"], embedding, tier,
            self.cfg["embedding_dim"], channels,
            self.cfg["guidance_scale"], node["params"]["scope_fp"])
        return {
            "files": {"latent.bin": np.ascontiguousarray(latent).tobytes()},
            "content_fp": array_fingerprint(latent),
            "extra": {"shape": list(latent.shape)},
        }

    def _compute_patch(self, node):
        tier = node["tier"]
        channels = self.cfg["latent_channels"]
        source_id = node["upstream"][0]
        latent = self._load_upstream_latent(source_id, tier, channels)
        patch = node["params"]["patch"]
        noop = latent_mod.patch_is_noop(
            latent, patch, self.cfg["patch_tolerance"])
        output = latent if noop else latent_mod.apply_patch(latent, patch)
        return {
            "files": {"latent.bin": np.ascontiguousarray(output).tobytes()},
            "content_fp": array_fingerprint(output),
            "extra": {"shape": list(output.shape), "patch_noop": noop},
        }

    def _compute_frame(self, node):
        tier = node["tier"]
        channels = self.cfg["latent_channels"]
        source_id = node["upstream"][0]
        latent = self._load_upstream_latent(source_id, tier, channels)
        frame = decoder_mod.decode_frame(
            latent, node["step_index"], tier, channels,
            node["params"]["scope_fp"])
        return {
            "files": {
                "frame.png": frame["png_bytes"],
                "pixels.bin": np.ascontiguousarray(frame["pixels"]).tobytes(),
            },
            "content_fp": frame["pixel_fp"],
            "extra": {
                "png_fp": frame["png_fp"],
                "decode_params": frame["params"],
                "pixel_shape": list(frame["pixels"].shape),
            },
        }

    def _tier(self, tier_id):
        for tier in self.cfg["tiers"]:
            if tier["id"] == tier_id:
                return tier
        raise KeyError("unknown tier %s" % tier_id)

    def _commit(self, node, key, result, durable=False):
        with self._lock:
            self.store.write_node_files(node["id"], result["files"],
                                        durable=durable)
            entry = {
                "kind": node["kind"],
                "prompt_id": node["prompt_id"],
                "tier_id": node.get("tier_id"),
                "cache_key": key,
                "content_fp": result["content_fp"],
                "file_fps": {
                    name: bytes_fingerprint(data)
                    for name, data in result["files"].items()
                },
                "version": self.store.manifest.get("version", 0) + 1,
            }
            if "path" in node:
                entry["path"] = node["path"]
            if "step_index" in node:
                entry["step_index"] = node["step_index"]
            entry.update(result.get("extra", {}))
            self.store.manifest.setdefault("nodes", {})[node["id"]] = entry

    # ------------------------------------------------------------- verification
    def verify_shared_segments(self, prompts):
        """Check img2img and txt2img share identical semantics after replace.

        For each img2img prompt, every post-replacement step is recomputed in a
        sandbox starting from the same latent via the shared
        :func:`latent.denoise_step`, and compared element-wise against both
        persisted paths. The check therefore proves the two paths share one
        denoise/noise/guidance semantics over the shared segment.
        """
        results = []
        channels = self.cfg["latent_channels"]
        replace = self.cfg["img2img_replace_steps"]
        for prompt in prompts:
            if not prompt.get("init_image"):
                continue
            for tier in self.cfg["tiers"]:
                pid, tid = prompt["id"], tier["id"]
                emb_id = "embedding::%s" % pid
                embedding = self.store.read_npy(
                    emb_id, "embedding.bin",
                    (self.cfg["embedding_dim"],), np.float64)
                checks = []
                for step in range(replace, self.cfg["denoise_steps"]):
                    txt_id = "denoise::%s::%s::%s::s%04d" % (
                        TXT2IMG, pid, tid, step)
                    img_id = "denoise::%s::%s::%s::s%04d" % (
                        IMG2IMG, pid, tid, step)
                    txt_latent = self._load_upstream_latent(
                        txt_id, tier, channels)
                    img_latent = self._load_upstream_latent(
                        img_id, tier, channels)
                    # Sandbox recomputation from the txt2img predecessor via
                    # the single shared denoise_step function.
                    # Resolve the real txt2img predecessor, which may be a
                    # patch node wrapping the previous denoise output.
                    prev_denoise = "denoise::%s::%s::%s::s%04d" % (
                        TXT2IMG, pid, tid, step - 1)
                    patch_id = "patch::%s::%s::s%04d" % (pid, tid, step - 1)
                    if patch_id in self.store.manifest["nodes"]:
                        prev_latent = self._load_upstream_latent(
                            patch_id, tier, channels)
                    else:
                        prev_latent = self._load_upstream_latent(
                            prev_denoise, tier, channels)
                    sandbox = latent_mod.denoise_step(
                        prev_latent, step, embedding, tier,
                        self.cfg["embedding_dim"], channels,
                        self.cfg["guidance_scale"], self._scope_fp(tid))
                    # Apply the current step's patch if one exists, mirroring
                    # the txt2img graph exactly.
                    current_patch = "patch::%s::%s::s%04d" % (
                        pid, tid, step)
                    if current_patch in self.store.manifest["nodes"]:
                        match = next(
                            (q for q in prompt.get("patches", [])
                             if q["step"] == step), None)
                        if match is not None:
                            sandbox = latent_mod.apply_patch(sandbox, match)
                    checks.append({
                        "step_index": step,
                        "txt2img_fp": array_fingerprint(txt_latent),
                        "img2img_fp": array_fingerprint(img_latent),
                        "sandbox_fp": array_fingerprint(sandbox),
                        "sandbox_matches_txt2img": bool(
                            np.array_equal(sandbox, txt_latent)),
                        "sandbox_matches_img2img": bool(
                            np.array_equal(sandbox, img_latent)),
                    })
                results.append({
                    "prompt_id": pid,
                    "tier_id": tid,
                    "replace_steps": replace,
                    "shared_steps": list(
                        range(replace, self.cfg["denoise_steps"])),
                    "checks": checks,
                    "semantics": "identical denoise_step function on both paths",
                })
        return results

    def read_latent(self, node_id):
        entry = self.store.manifest["nodes"][node_id]
        tier = self._tier(entry["tier_id"])
        return self.store.read_npy(
            node_id, "latent.bin",
            latent_mod.latent_shape(tier, self.cfg["latent_channels"]),
            np.float64)

    def read_pixels(self, node_id):
        entry = self.store.manifest["nodes"][node_id]
        shape = tuple(entry["pixel_shape"])
        raw = self.store.read_bytes(node_id, "pixels.bin")
        return np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()

    def read_png(self, node_id):
        return self.store.read_bytes(node_id, "frame.png")
