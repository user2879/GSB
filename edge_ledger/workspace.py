from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from .errors import Failure
from .loader import RawWorkspace, load_raw_workspace, read_yaml, write_yaml


class CommandError(ValueError):
    def __init__(self, message: str, *, command_id: str = "") -> None:
        super().__init__(message)
        self.command_id = command_id


@dataclass
class WorkspaceEditor:
    root: Path
    templates: list[dict[str, Any]]
    sample_batches: list[dict[str, Any]]
    config: dict[str, Any]
    injections: list[dict[str, Any]]
    sample_files: list[Path] = field(default_factory=list)
    next_batch_number: int = 1

    @classmethod
    def from_root(cls, root: Path) -> "WorkspaceEditor":
        raw = load_raw_workspace(Path(root))
        next_batch = len(raw.sample_files) + 1
        return cls(
            root=Path(root),
            templates=copy.deepcopy(raw.templates),
            sample_batches=copy.deepcopy(raw.sample_batches),
            config=copy.deepcopy(raw.config),
            injections=copy.deepcopy(raw.injections),
            sample_files=[Path(path) for path in raw.sample_files],
            next_batch_number=next_batch,
        )

    def _find_sample(self, sample_id: str) -> tuple[int, int] | None:
        for batch_index, batch in enumerate(self.sample_batches):
            for sample_index, sample in enumerate(batch.get("samples", []) or []):
                if isinstance(sample, Mapping) and sample.get("sample_id") == sample_id:
                    return batch_index, sample_index
        return None

    def _find_template_revision(
        self, template_id: str, revision: int
    ) -> tuple[int, dict[str, Any]] | None:
        for index, item in enumerate(self.templates):
            if item.get("template_id") == template_id and item.get("revision") == revision:
                return index, item
        return None

    def apply_command(self, command: Mapping[str, Any], *, command_id: str = "") -> None:
        operation = command.get("op", command.get("operation"))
        if not isinstance(operation, str):
            raise CommandError("command has no op", command_id=command_id)
        if operation == "add_samples":
            self._add_samples(command, command_id)
        elif operation in {"replace_sample", "upsert_sample"}:
            self._replace_sample(command, command_id, insert=operation == "upsert_sample")
        elif operation == "delete_sample":
            sample_id = command.get("sample_id")
            location = self._find_sample(str(sample_id)) if sample_id else None
            if location is None:
                raise CommandError(f"sample not found: {sample_id}", command_id=command_id)
            batch_index, sample_index = location
            del self.sample_batches[batch_index]["samples"][sample_index]
        elif operation == "upsert_template_revision":
            revision = command.get("revision_data", command.get("revision"))
            if not isinstance(revision, Mapping):
                raise CommandError("upsert_template_revision requires revision_data", command_id)
            record = copy.deepcopy(dict(revision))
            template_id = record.get("template_id")
            number = record.get("revision")
            if not isinstance(template_id, str) or not isinstance(number, int):
                raise CommandError(
                    "revision_data requires template_id and integer revision",
                    command_id=command_id,
                )
            found = self._find_template_revision(template_id, number)
            if found is None:
                self.templates.append(record)
            else:
                self.templates[found[0]] = record
        elif operation == "delete_template_revision":
            template_id = command.get("template_id")
            number = command.get("revision")
            if not isinstance(template_id, str) or not isinstance(number, int):
                raise CommandError(
                    "delete_template_revision requires template_id and revision",
                    command_id=command_id,
                )
            found = self._find_template_revision(template_id, number)
            if found is None:
                raise CommandError(
                    f"template revision not found: {template_id}@{number}",
                    command_id=command_id,
                )
            to_delete = {number}
            if command.get("cascade", True):
                changed = True
                while changed:
                    changed = False
                    for item in self.templates:
                        if (
                            item.get("template_id") == template_id
                            and item.get("parent_revision") in to_delete
                            and item.get("revision") not in to_delete
                        ):
                            to_delete.add(item.get("revision"))
                            changed = True
            self.templates = [
                item
                for item in self.templates
                if not (
                    item.get("template_id") == template_id
                    and item.get("revision") in to_delete
                )
            ]
        elif operation == "rollback_template":
            template_id = command.get("template_id")
            target = command.get("to_revision", command.get("revision"))
            if not isinstance(template_id, str) or not isinstance(target, int):
                raise CommandError(
                    "rollback_template requires template_id and to_revision",
                    command_id=command_id,
                )
            if self._find_template_revision(template_id, target) is None:
                raise CommandError(
                    f"rollback target not found: {template_id}@{target}",
                    command_id=command_id,
                )
            descendants = {target}
            changed = True
            while changed:
                changed = False
                for item in self.templates:
                    if (
                        item.get("template_id") == template_id
                        and item.get("parent_revision") in descendants
                        and item.get("revision") not in descendants
                    ):
                        descendants.add(item.get("revision"))
                        changed = True
            self.templates = [
                item
                for item in self.templates
                if not (
                    item.get("template_id") == template_id
                    and item.get("revision") in descendants
                    and item.get("revision") != target
                )
            ]
        elif operation in {"set_slot_constraint", "update_slot_constraint"}:
            self._set_slot_constraint(command, command_id)
        elif operation == "set_dedupe_threshold":
            value = command.get("value", command.get("threshold_permille"))
            if not isinstance(value, int) or isinstance(value, bool):
                raise CommandError("dedupe threshold must be an integer", command_id=command_id)
            self.config.setdefault("dedupe", {})["threshold_permille"] = value
        elif operation in {"set_quantization", "set_fixed_point"}:
            fixed = self.config.setdefault("fixed_point", {})
            if "quantization" in command:
                fixed["quantization"] = command["quantization"]
            if "decimal_places" in command:
                fixed["decimal_places"] = command["decimal_places"]
        elif operation == "set_scheduler":
            scheduler = self.config.setdefault("scheduler", {})
            for key in (
                "max_workers",
                "render_point_budget",
                "source_quota_points",
                "points_per_render",
            ):
                if key in command:
                    scheduler[key] = command[key]
        elif operation == "set_quality":
            quality = self.config.setdefault("quality", {})
            for key in (
                "answer_soft_limit_chars",
                "answer_decay_per_char",
                "answer_floor",
                "weights_permille",
            ):
                if key in command:
                    quality[key] = copy.deepcopy(command[key])
        else:
            raise CommandError(f"unsupported operation: {operation}", command_id=command_id)

    def _add_samples(self, command: Mapping[str, Any], command_id: str) -> None:
        samples = command.get("samples")
        batch_id = command.get("batch_id")
        if samples is None and command.get("batch_file"):
            path = self.root / str(command["batch_file"])
            payload = read_yaml(path, default={})
            if not isinstance(payload, Mapping):
                raise CommandError(f"batch file is not a mapping: {path}", command_id=command_id)
            samples = payload.get("samples", [])
            batch_id = batch_id or payload.get("batch_id")
        if not isinstance(samples, list) or not all(isinstance(item, Mapping) for item in samples):
            raise CommandError("add_samples requires a samples list", command_id=command_id)
        if not isinstance(batch_id, str) or not batch_id:
            batch_id = f"batch-{self.next_batch_number:03d}"
        self.sample_batches.append(
            {"batch_id": batch_id, "samples": [copy.deepcopy(dict(item)) for item in samples]}
        )
        if len(self.sample_files) < len(self.sample_batches):
            self.sample_files.append(
                self.root
                / "data"
                / "samples"
                / f"batch-{self.next_batch_number:03d}.yaml"
            )
        self.next_batch_number += 1

    def _replace_sample(
        self, command: Mapping[str, Any], command_id: str, *, insert: bool
    ) -> None:
        replacement = command.get("sample", command.get("data"))
        if not isinstance(replacement, Mapping):
            raise CommandError("replace_sample requires a sample mapping", command_id=command_id)
        replacement = copy.deepcopy(dict(replacement))
        sample_id = command.get("sample_id", replacement.get("sample_id"))
        if not isinstance(sample_id, str) or not sample_id:
            raise CommandError("replacement sample_id is required", command_id=command_id)
        replacement["sample_id"] = sample_id
        location = self._find_sample(sample_id)
        if location is not None:
            batch_index, sample_index = location
            self.sample_batches[batch_index]["samples"][sample_index] = replacement
        elif insert:
            if not self.sample_batches:
                self.sample_batches.append({"batch_id": "batch-001", "samples": []})
            self.sample_batches[-1]["samples"].append(replacement)
        else:
            raise CommandError(f"sample not found: {sample_id}", command_id=command_id)

    def _set_slot_constraint(self, command: Mapping[str, Any], command_id: str) -> None:
        template_id = command.get("template_id")
        revision = command.get("revision")
        slot = command.get("slot", command.get("slot_name"))
        fields = command.get("fields", command.get("constraint"))
        if (
            not isinstance(template_id, str)
            or not isinstance(revision, int)
            or not isinstance(slot, str)
            or not isinstance(fields, Mapping)
        ):
            raise CommandError(
                "set_slot_constraint requires template_id, revision, slot and fields",
                command_id=command_id,
            )
        found = self._find_template_revision(template_id, revision)
        if found is None:
            raise CommandError(
                f"template revision not found: {template_id}@{revision}",
                command_id=command_id,
            )
        record = found[1]
        constraints = record.setdefault("constraints", {})
        current = constraints.setdefault(slot, {})
        current.update(copy.deepcopy(dict(fields)))

    def save(self) -> None:
        self.templates.sort(key=lambda item: (item.get("template_id", ""), item.get("revision", 0)))
        write_yaml(
            self.root / "data" / "templates.yaml",
            {"schema_version": 1, "templates": self.templates},
        )
        write_yaml(self.root / "config" / "pipeline.yaml", self.config)
        sample_dir = self.root / "data" / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(sample_dir.glob("*.yaml"))
        for index, batch in enumerate(self.sample_batches):
            if index < len(self.sample_files):
                path = Path(self.sample_files[index])
                if not path.is_absolute():
                    path = self.root / path
            elif index < len(existing):
                path = existing[index]
            else:
                path = sample_dir / f"batch-{index + 1:03d}.yaml"
            write_yaml(path, batch)
        # Remove files from a previously larger batch set only after all new files
        # have been successfully written.
        for stale in existing[len(self.sample_batches) :]:
            try:
                stale.unlink()
            except OSError:
                pass


def read_commands(path: Path) -> list[dict[str, Any]]:
    payload = read_yaml(path, default={})
    if isinstance(payload, Mapping):
        commands = payload.get("commands", [])
    elif isinstance(payload, list):
        commands = payload
    else:
        commands = []
    if not isinstance(commands, list) or not all(isinstance(item, Mapping) for item in commands):
        raise CommandError("commands file must contain a commands list")
    return [dict(item) for item in commands]


def apply_command_file(
    root: Path, command_path: Path
) -> tuple[WorkspaceEditor | None, list[Failure]]:
    editor = WorkspaceEditor.from_root(root)
    failures: list[Failure] = []
    commands = read_commands(command_path)
    staged = copy.deepcopy(editor)
    for index, command in enumerate(commands):
        command_id = str(command.get("id", f"command-{index + 1:03d}"))
        try:
            staged.apply_command(command, command_id=command_id)
        except CommandError as exc:
            failures.append(
                Failure(
                    code="E_COMMAND_INVALID",
                    stage="load_config",
                    message=str(exc),
                    target_kind="command",
                    target_id=command_id,
                )
            )
    if failures:
        return None, sorted(failures, key=Failure.sort_key)
    staged.save()
    return staged, []
