from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .clock import Clock, MonotonicClock
from .config import WorkbenchConfig
from .engine import WorkbenchEngine
from .hashing import fingerprint
from .ledger import Ledger
from .synth import SampleSpec, build_sample, write_sample_workspace


class Studio:
    def __init__(
        self,
        root: str | Path,
        clock: Clock | None = None,
        engine: WorkbenchEngine | None = None,
    ):
        self.root = Path(root)
        self.clock = clock or MonotonicClock()
        self.engine = engine or WorkbenchEngine(self.root, clock=self.clock)
        self.ledger = Ledger(self.root / "ledger" / "ledger.jsonl", self.clock)
        self.global_version = 0

    @classmethod
    def create(
        cls,
        root: str | Path,
        config: WorkbenchConfig | None = None,
        clock: Clock | None = None,
    ) -> "Studio":
        config = config or WorkbenchConfig()
        write_sample_workspace(root, config)
        studio = cls(root, clock=clock)
        studio.global_version = 1
        studio.engine.global_version = 1
        studio.ledger.append("WORKSPACE_INITIALIZED", {"sample_count": config.sample_count})
        return studio

    def config_path(self) -> Path:
        return self.root / "configs" / "workbench.json"

    def read_config_raw(self) -> dict[str, Any]:
        return json.loads(self.config_path().read_text(encoding="utf-8"))

    def write_config_raw(self, raw: dict[str, Any]) -> None:
        self.config_path().write_text(
            json.dumps(raw, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )

    def current_config(self) -> WorkbenchConfig:
        return WorkbenchConfig.from_dict(self.read_config_raw())

    def load_specs(self) -> list[dict[str, Any]]:
        path = self.root / "samples" / "specs.json"
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    def save_specs(self, specs: list[dict[str, Any]]) -> None:
        (self.root / "samples" / "specs.json").write_text(
            json.dumps(specs, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )

    def _write_sample_from_spec(self, spec: dict[str, Any]) -> dict[str, Any]:
        config = WorkbenchConfig.from_dict(self.read_config_raw())
        sample_spec = SampleSpec(
            image_id=spec["image_id"],
            seed=int(spec["seed"]),
            column_counts=tuple(int(v) for v in spec["column_counts"]),
            glyph_offset=int(spec.get("glyph_offset", 0)),
        )
        bundle = build_sample(sample_spec, config)
        (self.root / "samples" / f"{spec['image_id']}.png").write_bytes(bundle["png_bytes"])
        (self.root / "truth" / f"{spec['image_id']}.json").write_text(
            json.dumps(bundle["truth"], ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return bundle["truth"]

    def _current_annotation_version(self) -> str:
        path = self.root / "annotations" / "keypoint_versions.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8")).get("version_id", "kv-manual")
        return "kv-manual"

    def rewrite_annotations(
        self,
        version_id: str,
        jitter: int = 0,
        jitter_by_image: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        jitter_by_image = jitter_by_image or {}
        documents: dict[str, Any] = {}
        truth_paths = sorted((self.root / "truth").glob("*.json"), key=lambda p: p.stem)
        for index, truth_path in enumerate(truth_paths, start=1):
            truth = json.loads(truth_path.read_text(encoding="utf-8"))
            image_id = truth["image_id"]
            image_jitter = int(jitter_by_image.get(image_id, jitter))
            import random

            rng = random.Random(f"annotation:{version_id}:{image_id}:{image_jitter}")
            points = []
            for point in truth["keypoints"]:
                dx = rng.randint(-image_jitter, image_jitter) if image_jitter else 0
                dy = rng.randint(-image_jitter, image_jitter) if image_jitter else 0
                item = {
                    "keypoint_id": point["keypoint_id"],
                    "x": int(point["x"] + dx),
                    "y": int(point["y"] + dy),
                    "column_rank": point["column_rank"],
                    "reading_order": point["reading_order"],
                }
                points.append(item)
            doc = {"version_id": f"{version_id}-doc{index:03d}", "keypoints": points}
            doc["content_digest"] = fingerprint(points)
            documents[image_id] = doc

        bundle = {
            "schema": "vertical-keypoint-versions/v1",
            "version_id": version_id,
            "documents": documents,
        }
        bundle["content_digest"] = fingerprint(
            {image_id: doc["content_digest"] for image_id, doc in sorted(documents.items())}
        )
        (self.root / "annotations" / "keypoint_versions.json").write_text(
            json.dumps(bundle, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        return bundle

    def apply(self, command: dict[str, Any], run: bool = True) -> dict[str, Any] | None:
        op = command["op"]
        self.global_version += 1
        self.engine.global_version = self.global_version
        mutation_summary = self._mutate(command)
        self.ledger.append(
            "COMMAND_APPLIED",
            {"global_version": self.global_version, "op": op, "command": command, "summary": mutation_summary},
        )
        if not run:
            return None
        result = self.engine.run(self.global_version)
        self.ledger.append(
            "RUN_COMPLETED",
            {
                "global_version": self.global_version,
                "keypoint_count": len(result.get("keypoints", [])),
                "failure_count": len(result.get("failures", [])),
                "pending_count": len(result.get("pending_queue", [])),
                "config_hash": result.get("config_hash"),
            },
        )
        return result

    def _mutate(self, command: dict[str, Any]) -> dict[str, Any]:
        op = command["op"]
        if op in ("update_config", "set_config"):
            raw = self.read_config_raw()
            raw.update(command.get("patch", {}))
            self.write_config_raw(raw)
            return {"updated_keys": sorted(command.get("patch", {}))}

        if op == "set_budget":
            raw = self.read_config_raw()
            patch = command.get("patch", command)
            for key in ("global_point_budget", "column_point_quota", "max_component_comparisons"):
                if key in patch:
                    raw[key] = int(patch[key])
            self.write_config_raw(raw)
            return {"budget": {key: raw[key] for key in ("global_point_budget", "column_point_quota", "max_component_comparisons")}}

        if op == "set_lease_limit":
            raw = self.read_config_raw()
            raw["max_active_leases"] = int(command["max_active_leases"])
            self.write_config_raw(raw)
            return {"max_active_leases": raw["max_active_leases"]}

        specs = self.load_specs()
        by_id = {spec["image_id"]: spec for spec in specs}

        if op == "replace_image":
            image_id = command["image_id"]
            spec = dict(by_id.get(image_id, {"image_id": image_id, "glyph_offset": 0}))
            spec["seed"] = int(command.get("seed", spec.get("seed", 1000)))
            columns = int(command.get("columns", len(spec.get("column_counts", (3, 4, 3)))))
            if "counts" in command:
                counts = [int(v) for v in command["counts"]]
            else:
                old_counts = spec.get("column_counts", [6] * columns)
                counts = [int(old_counts[i]) if i < len(old_counts) else 6 for i in range(columns)]
            if len(counts) != columns:
                raise ValueError("counts length must equal columns")
            spec["column_counts"] = counts
            by_id[image_id] = spec
            self.save_specs(list(by_id.values()))
            self._write_sample_from_spec(spec)
            self.rewrite_annotations(self._current_annotation_version())
            return {"image_id": image_id, "spec": spec}

        if op == "add_image":
            image_id = command.get("image_id") or f"sample_{len(specs) + 1:03d}"
            if image_id in by_id:
                image_id = f"sample_{len(specs) + 1:03d}-{self.global_version}"
            columns = int(command.get("columns", 4))
            counts = [int(v) for v in command.get("counts", [6] * columns)]
            if len(counts) != columns:
                raise ValueError("counts length must equal columns")
            spec = {
                "image_id": image_id,
                "seed": int(command.get("seed", 9000 + len(specs) * 97)),
                "column_counts": counts,
                "glyph_offset": int(command.get("glyph_offset", len(specs) * 3)),
            }
            specs.append(spec)
            self.save_specs(specs)
            self._write_sample_from_spec(spec)
            self.rewrite_annotations(self._current_annotation_version())
            return {"image_id": image_id, "spec": spec}

        if op == "delete_image":
            image_id = command["image_id"]
            specs = [spec for spec in specs if spec["image_id"] != image_id]
            self.save_specs(specs)
            png = self.root / "samples" / f"{image_id}.png"
            truth = self.root / "truth" / f"{image_id}.json"
            if png.exists():
                png.unlink()
            if truth.exists():
                truth.unlink()
            annotation_path = self.root / "annotations" / "keypoint_versions.json"
            bundle = json.loads(annotation_path.read_text(encoding="utf-8"))
            bundle.get("documents", {}).pop(image_id, None)
            bundle["content_digest"] = fingerprint(
                {
                    doc_id: doc.get("content_digest")
                    for doc_id, doc in sorted(bundle.get("documents", {}).items())
                }
            )
            annotation_path.write_text(
                json.dumps(bundle, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
            )
            return {"deleted": image_id}

        if op == "replace_annotation_version":
            version_id = command.get("version_id", f"kv-{self.global_version + 1:04d}")
            bundle = self.rewrite_annotations(
                version_id,
                jitter=int(command.get("jitter", 0)),
                jitter_by_image=command.get("jitter_by_image"),
            )
            return {"version_id": version_id, "digest": bundle["content_digest"]}

        raise ValueError(f"unknown command op: {op}")
