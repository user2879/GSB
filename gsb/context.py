"""Resolve and load a local project configuration, templates and dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from gsb.config import RunConfig, load_config
from gsb.constants import E_CONFIG_INVALID
from gsb.errors import GsbError, ValidationIssue
from gsb.fonts import TemplateLibrary
from gsb.models import DatasetState
from gsb.pipeline import load_templates_for_config
from gsb.samples import load_dataset


@dataclass(frozen=True)
class ProjectContext:
    root: Path
    config_path: Path
    dataset_path: Path
    templates_path: Path
    config: RunConfig
    templates: TemplateLibrary | None
    dataset: DatasetState
    template_issues: tuple[ValidationIssue, ...]
    dataset_issues: tuple[ValidationIssue, ...]

    def require_templates(self) -> TemplateLibrary:
        if self.templates is None:
            first = self.template_issues[0] if self.template_issues else None
            raise GsbError(
                first.code if first is not None else E_CONFIG_INVALID,
                first.message if first is not None else "template library is unavailable",
                source=first.source if first is not None else str(self.templates_path),
            )
        return self.templates


def resolve_template_path(
    *,
    root: str | Path,
    config: RunConfig,
    templates_path: str | Path | None = None,
) -> Path:
    root_path = Path(root).resolve()
    if templates_path is not None:
        candidate = Path(templates_path)
        return candidate if candidate.is_absolute() else root_path / candidate
    registry_path = root_path / "configs" / "template_paths.yaml"
    try:
        registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GsbError(
            E_CONFIG_INVALID,
            "template path registry is missing",
            source=str(registry_path),
        ) from exc
    except yaml.YAMLError as exc:
        raise GsbError(
            E_CONFIG_INVALID,
            f"template path registry is invalid: {exc}",
            source=str(registry_path),
        ) from exc
    if not isinstance(registry, dict):
        raise GsbError(
            E_CONFIG_INVALID,
            "template path registry must be a mapping",
            source=str(registry_path),
        )
    selected = registry.get(config.recognition.template_version)
    if not isinstance(selected, str) or not selected:
        raise GsbError(
            E_CONFIG_INVALID,
            f"template version {config.recognition.template_version!r} is not declared",
            source=str(registry_path),
        )
    return root_path / selected


def load_project(
    *,
    root: str | Path,
    config_path: str | Path = "configs/default.yaml",
    dataset_path: str | Path = "samples/dataset.yaml",
    templates_path: str | Path | None = None,
) -> ProjectContext:
    root_path = Path(root).resolve()
    resolved_config = Path(config_path)
    if not resolved_config.is_absolute():
        resolved_config = root_path / resolved_config
    resolved_dataset = Path(dataset_path)
    if not resolved_dataset.is_absolute():
        resolved_dataset = root_path / resolved_dataset
    config = load_config(resolved_config)
    resolved_templates = resolve_template_path(
        root=root_path,
        config=config,
        templates_path=templates_path,
    )
    templates, template_issues = load_templates_for_config(config, resolved_templates)
    dataset, dataset_issues = load_dataset(resolved_dataset, config=config)
    return ProjectContext(
        root=root_path,
        config_path=resolved_config,
        dataset_path=resolved_dataset,
        templates_path=resolved_templates,
        config=config,
        templates=templates,
        dataset=dataset,
        template_issues=tuple(template_issues),
        dataset_issues=tuple(dataset_issues),
    )


def issues_payload(issues: tuple[ValidationIssue, ...] | list[ValidationIssue]) -> list[dict[str, Any]]:
    return [issue.to_dict() for issue in issues]
