"""Command-stream execution orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import (
    PRECISION_ALIASES,
    QUANTIZATION_ALIASES,
    SCHEME_ALIASES,
    SCHEME_DISPLAY,
    WorkbenchConfig,
    apply_config_patch,
)
from .engine import RunResult, WorkbenchEngine
from .models import Issue, Sample
from .reporting import build_diagnostic_report, build_visual_summary, write_reports
from .sensitivity import SensitivityAnalyzer, sensitivity_cache_key


@dataclass
class ExecutionResult:
    exit_code: int
    config: WorkbenchConfig
    engine: WorkbenchEngine
    reports: dict[str, str]
    report: dict[str, Any] | None
    run_results: list[RunResult] = field(default_factory=list)
    runtime_issues: list[Issue] = field(default_factory=list)
    command_log: list[dict[str, Any]] = field(default_factory=list)


def _normalize_list(values: Any, aliases: dict[str, str], allowed: set[str], field_name: str) -> tuple[list[str], list[Issue]]:
    issues: list[Issue] = []
    if not isinstance(values, list):
        return [], [Issue("E_COMMAND_PARAM", None, field_name, "must be an array")]
    result: list[str] = []
    for index, value in enumerate(values):
        normalized = aliases.get(value, None) if isinstance(value, str) else None
        if normalized not in allowed:
            issues.append(
                Issue(
                    "E_COMMAND_PARAM",
                    None,
                    f"{field_name}[{index}]",
                    f"unsupported value {value!r}",
                )
            )
        elif normalized not in result:
            result.append(normalized)
    return result, issues


class CommandExecutor:
    def __init__(
        self,
        config: WorkbenchConfig,
        samples: list[Sample],
        commands: list[dict[str, Any]],
        *,
        state_dir: str | Path = "state",
        reports_dir: str | Path = "reports",
        sample_content_hash: str | None = None,
        validation_issues: list[Issue] | None = None,
        command_issues: list[Issue] | None = None,
    ):
        self.engine = WorkbenchEngine(
            config,
            samples,
            str(state_dir),
            sample_content_hash=sample_content_hash,
        )
        self.commands = commands
        self.state_dir = Path(state_dir)
        self.reports_dir = Path(reports_dir)
        self.validation_issues = list(validation_issues or [])
        self.command_issues = list(command_issues or [])
        self.runtime_issues: list[Issue] = []
        self.command_log: list[dict[str, Any]] = []
        self.run_results: list[RunResult] = []
        self.current_report: dict[str, Any] | None = None
        self.current_sensitivity: dict[str, Any] | None = None
        self.report_paths: dict[str, str] = {}

    def _issue(self, code: str, field_name: str, message: str, line: int | None = None) -> Issue:
        return Issue(code=code, sample_id=None, field=field_name, message=message, line=line)

    def _normalize_sensitivity_options(
        self, params: dict[str, Any], line: int | None
    ) -> tuple[dict[str, Any], list[Issue]]:
        options: dict[str, Any] = {}
        issues: list[Issue] = []
        if "scheme_candidates" in params:
            values, value_issues = _normalize_list(
                params["scheme_candidates"],
                SCHEME_ALIASES,
                set(SCHEME_DISPLAY),
                "params.scheme_candidates",
            )
            options["scheme_candidates"] = values
            issues.extend(value_issues)
        if "precision_levels" in params:
            values, value_issues = _normalize_list(
                params["precision_levels"],
                PRECISION_ALIASES,
                {"fp16", "fp32", "fp64"},
                "params.precision_levels",
            )
            options["precision_levels"] = values
            issues.extend(value_issues)
        if "quantization_levels" in params:
            values, value_issues = _normalize_list(
                params["quantization_levels"],
                QUANTIZATION_ALIASES,
                {"fine", "balanced", "coarse"},
                "params.quantization_levels",
            )
            options["quantization_levels"] = values
            issues.extend(value_issues)
        if "fixed_decimal_widths" in params:
            widths = params["fixed_decimal_widths"]
            if (
                not isinstance(widths, list)
                or not widths
                or any(
                    not isinstance(value, int) or isinstance(value, bool) or value <= 0
                    for value in widths
                )
            ):
                issues.append(
                    self._issue(
                        "E_COMMAND_PARAM",
                        "params.fixed_decimal_widths",
                        "must be a non-empty array of positive integers",
                        line,
                    )
                )
            else:
                options["fixed_decimal_widths"] = sorted(set(int(value) for value in widths))
        if "data_scales" in params:
            scales = params["data_scales"]
            if (
                not isinstance(scales, list)
                or not scales
                or any(
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not 0 < float(value) <= 1
                    for value in scales
                )
            ):
                issues.append(
                    self._issue(
                        "E_COMMAND_PARAM",
                        "params.data_scales",
                        "must be a non-empty array in (0, 1]",
                        line,
                    )
                )
            else:
                options["data_scales"] = sorted(set(float(value) for value in scales))
        for issue in issues:
            issue.line = line
        return options, issues

    def _get_sensitivity(
        self,
        options: dict[str, Any] | None = None,
        line: int | None = None,
    ) -> tuple[dict[str, Any] | None, list[Issue]]:
        options = options or {}
        key = sensitivity_cache_key(
            self.engine.config,
            self.engine.sample_content_hash,
            options,
        )
        cached = self.engine.cached_sensitivity.get(key)
        if cached is not None:
            self.current_sensitivity = cached
            return cached, []
        try:
            result = SensitivityAnalyzer(
                self.engine.config,
                self.engine.samples,
                options,
            ).analyze()
        except Exception as exc:  # diagnostic surface must survive analysis failures
            return None, [
                self._issue(
                    "E_SENSITIVITY_ANALYSIS",
                    "sensitivity",
                    f"sensitivity analysis failed: {type(exc).__name__}: {exc}",
                    line,
                )
            ]
        self.engine.cache_sensitivity(key, result)
        self.current_sensitivity = result
        return result, []

    def _default_sensitivity_options(self) -> dict[str, Any]:
        config = self.engine.config
        return {
            "scheme_candidates": list(config.scheme_candidates),
            "precision_levels": list(config.precision_levels),
            "quantization_levels": list(config.quantization_levels),
            "fixed_decimal_widths": list(config.fixed_decimal_widths),
            "data_scales": [0.25, 0.5, 1.0],
        }

    def _write_report(self) -> None:
        if self.engine.last_run is None:
            return
        sensitivity = self.current_sensitivity
        if sensitivity is None:
            sensitivity, issues = self._get_sensitivity(self._default_sensitivity_options())
            self.runtime_issues.extend(issues)
        report = build_diagnostic_report(
            self.engine,
            self.engine.last_run,
            sensitivity,
            self.validation_issues,
            [*self.command_issues, *self.runtime_issues],
        )
        visual = build_visual_summary(report)
        self.report_paths = write_reports(self.reports_dir, report, visual)
        self.current_report = report

    def _ensure_run(self, line: int | None = None) -> list[Issue]:
        if self.engine.last_run is not None:
            return []
        result = self.engine.run(command_cursor=line)
        self.run_results.append(result)
        return []

    def execute(self) -> ExecutionResult:
        for command in self.commands:
            name = command["command"]
            params = command.get("params", {})
            line = command.get("line")
            log_entry: dict[str, Any] = {"line": line, "command": name}
            if name == "set":
                patch = command.get("patch", {})
                new_config, issues = apply_config_patch(self.engine.config, patch)
                for issue in issues:
                    issue.line = line
                if issues:
                    self.runtime_issues.extend(issues)
                    log_entry["status"] = "rejected"
                else:
                    assert new_config is not None
                    self.engine.set_config(new_config)
                    self.current_sensitivity = None
                    log_entry["status"] = "applied"
                    log_entry["config_hash"] = self.engine.config_hash
            elif name == "run":
                invalid_param = False
                for param_name in ("point_budget", "sample_quota", "snapshot_interval"):
                    if param_name in params:
                        value = params[param_name]
                        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                            self.runtime_issues.append(
                                self._issue(
                                    "E_COMMAND_PARAM",
                                    f"params.{param_name}",
                                    "must be a positive integer",
                                    line,
                                )
                            )
                            invalid_param = True
                if invalid_param:
                    log_entry["status"] = "rejected"
                    self.command_log.append(log_entry)
                    continue
                result = self.engine.run(
                    point_budget=params.get("point_budget"),
                    sample_quota=params.get("sample_quota"),
                    snapshot_interval=params.get("snapshot_interval"),
                    command_cursor=line,
                )
                self.run_results.append(result)
                sensitivity, issues = self._get_sensitivity(
                    self._default_sensitivity_options(), line=line
                )
                self.runtime_issues.extend(issues)
                self._write_report()
                log_entry.update(
                    {
                        "status": "completed" if not result.pending_samples else "pending",
                        "completed_samples": len(result.completed_samples),
                        "pending_samples": len(result.pending_samples),
                        "consumed_points": result.consumed_points,
                        "resume_cursor": result.resume_cursor,
                    }
                )
            elif name == "sensitivity":
                self.runtime_issues.extend(self._ensure_run(line))
                options, issues = self._normalize_sensitivity_options(params, line)
                self.runtime_issues.extend(issues)
                sensitivity, issues = self._get_sensitivity(options, line=line)
                self.runtime_issues.extend(issues)
                if sensitivity is not None:
                    self.current_sensitivity = sensitivity
                    self._write_report()
                log_entry["status"] = "completed" if sensitivity is not None else "failed"
                if sensitivity is not None:
                    log_entry["variant_count"] = sensitivity["variant_count"]
                    log_entry["flip_count"] = sensitivity["flip_attribution"]["total_flips"]
            elif name == "report":
                self.runtime_issues.extend(self._ensure_run(line))
                if self.current_sensitivity is None:
                    sensitivity, issues = self._get_sensitivity(
                        self._default_sensitivity_options(), line=line
                    )
                    self.runtime_issues.extend(issues)
                    self.current_sensitivity = sensitivity
                self._write_report()
                log_entry["status"] = "written"
            elif name == "status":
                last = self.engine.last_run
                log_entry["status"] = "ok"
                log_entry["engine_status"] = {
                    "resume_state": self.engine.resume_state,
                    "config_hash": self.engine.config_hash,
                    "completed_samples": len(last.completed_samples) if last else 0,
                    "pending_samples": len(last.pending_samples) if last else 0,
                    "snapshot_id": self.engine.loaded_snapshot.snapshot_id,
                }
            self.command_log.append(log_entry)

        if self.engine.last_run is not None:
            if self.current_sensitivity is None:
                sensitivity, issues = self._get_sensitivity(
                    self._default_sensitivity_options()
                )
                self.runtime_issues.extend(issues)
                self.current_sensitivity = sensitivity
            self._write_report()
        all_issues = [*self.command_issues, *self.runtime_issues]
        pending = bool(self.engine.last_run and self.engine.last_run.pending_samples)
        exit_code = 3 if pending else 2 if all_issues else 0
        return ExecutionResult(
            exit_code=exit_code,
            config=self.engine.config,
            engine=self.engine,
            reports=self.report_paths,
            report=self.current_report,
            run_results=self.run_results,
            runtime_issues=self.runtime_issues,
            command_log=self.command_log,
        )


def execution_summary(result: ExecutionResult) -> dict[str, Any]:
    last = result.engine.last_run
    report = result.report or {}
    return {
        "exit_code": result.exit_code,
        "scheme": result.config.scheme,
        "scheme_display": result.config.scheme_display,
        "config_hash": result.engine.config_hash,
        "resume_state": result.engine.resume_state,
        "snapshot_id": last.snapshot_id if last else None,
        "completed_samples": len(last.completed_samples) if last else 0,
        "pending_samples": len(last.pending_samples) if last else 0,
        "resume_cursor": last.resume_cursor if last else None,
        "points_consumed_this_run": last.consumed_points if last else 0,
        "report_id": report.get("report_id"),
        "reports": result.reports,
        "runtime_issue_count": len(result.runtime_issues),
        "command_log": result.command_log,
    }
