"""Per-series deterministic pipeline with content-fingerprint caching."""

from __future__ import annotations

import time

import numpy as np

from .model import run_backtest, run_gradient_descent, fit_ols
from .state import RunState
from .synthetic import SeriesData, fragment_id_of
from .transforms import build_feature_matrix, preprocess
from .util import (
    content_fingerprint,
    derived_key,
    matrix_summary,
    native,
    rounded_float,
)


def series_fingerprint(series: SeriesData) -> str:
    return content_fingerprint(
        {"series_id": series.series_id, "values": series.values}
    )


def block_indices(length: int, block_size: int) -> list[tuple[int, int]]:
    blocks = []
    for start in range(0, length, block_size):
        blocks.append((start, min(length, start + block_size)))
    return blocks


class SeriesProcessor:
    """Runs the four stages for one series and records reuse/recompute lists."""

    def __init__(self, state: RunState, params: dict, tier_params: dict, block_size: int):
        self.state = state
        self.params = params
        # Concurrency and snapshot cadence only control execution, never the
        # mathematical artifacts, so they must not participate in cache keys.
        artifact_params = {
            key: value
            for key, value in params.items()
            if key not in ("concurrency", "snapshot_interval")
        }
        self.params_key = content_fingerprint(artifact_params)
        self.tier_params = tier_params
        self.block_size = block_size

    def reset_audit(self) -> None:
        self.reused: list[str] = []
        self.recomputed: list[str] = []
        self.stage_seconds: dict[str, float] = {
            "preprocess": 0.0,
            "features": 0.0,
            "models": 0.0,
            "backtest": 0.0,
        }

    # ------------------------------------------------------------------
    # Stage 1: preprocessing with fragment-level (block-level) caching
    # ------------------------------------------------------------------
    def run_preprocess(self, series: SeriesData) -> dict:
        self.reset_audit()
        started = time.perf_counter()
        x = np.asarray(series.values, dtype=np.float64)
        length = x.shape[0]
        full = preprocess(x, self.params)

        segment_results: list[dict] = []
        reused_fragments: list[str] = []
        rebuilt_fragments: list[str] = []
        for block_start, block_end in block_indices(length, self.block_size):
            fragment_id = fragment_id_of(block_start, self.block_size)
            block_payload = {
                "fragment_id": fragment_id,
                "block_start": block_start,
                "block_end": block_end,
                "trend": [],
                "seasonal": [],
                "adjusted": [],
                "differenced": [],
                "preprocess_deps": [],
                "diff_deps": [],
                "values_fingerprint": [],
            }
            block_reused = 0
            for idx in range(block_start, block_end):
                # Per-point key on a fixed grid: appending points never changes
                # the key of any previously computed index.
                point_key = derived_key(
                    [
                        "preprocess-point",
                        self.params_key,
                        idx,
                        # The full raw dependency window: any in-range edit
                        # changes the key and forces an exact recompute.
                        [
                            float(v)
                            for v in x[
                                full["diff_deps"][idx][0] : full["diff_deps"][idx][1] + 1
                            ]
                        ],
                    ]
                )
                cached = self.state.load(
                    self.state.preprocess_dir, series.series_id, point_key
                )
                if cached is not None:
                    point = cached
                    block_reused += 1
                    self.reused.append(f"preprocess:{fragment_id}:idx-{idx}")
                else:
                    point = {
                        "index": idx,
                        "fragment_id": fragment_id,
                        # Round before caching: the in-memory value and the
                        # JSON-persisted value then are the exact same float,
                        # eliminating full-vs-incremental last-digit drift.
                        "trend": rounded_float(full["trend"][idx]),
                        "seasonal": rounded_float(full["seasonal"][idx]),
                        "adjusted": rounded_float(full["adjusted"][idx]),
                        "differenced": rounded_float(full["differenced"][idx]),
                        "preprocess_deps": list(full["preprocess_deps"][idx]),
                        "diff_deps": list(full["diff_deps"][idx]),
                    }
                    self.state.store(
                        self.state.preprocess_dir,
                        series.series_id,
                        point_key,
                        point,
                    )
                    self.recomputed.append(f"preprocess:{fragment_id}:idx-{idx}")
                block_payload["trend"].append(point["trend"])
                block_payload["seasonal"].append(point["seasonal"])
                block_payload["adjusted"].append(point["adjusted"])
                block_payload["differenced"].append(point["differenced"])
                block_payload["preprocess_deps"].append(point["preprocess_deps"])
                block_payload["diff_deps"].append(point["diff_deps"])

            # Content identity of the fragment, independent of key naming.
            block_payload["values_fingerprint"] = content_fingerprint(
                [float(v) for v in x[block_start:block_end]]
            )
            segment_results.append(block_payload)
            if block_reused == block_end - block_start:
                reused_fragments.append(fragment_id)
            elif block_reused == 0:
                rebuilt_fragments.append(fragment_id)
            else:
                rebuilt_fragments.append(fragment_id)

        ordered = sorted(segment_results, key=lambda p: p["block_start"])
        # Reconstruct every downstream array from the persisted (canonical)
        # point values. A cache hit and a fresh compute then traverse the exact
        # same bytes, which is what makes full vs incremental fingerprints
        # identical regardless of thread scheduling.
        canonical = {
            key: np.asarray(
                [v for part in ordered for v in part[key]], dtype=np.float64
            )
            for key in ("trend", "seasonal", "adjusted", "differenced")
        }
        preprocess_deps = [
            pair for part in ordered for pair in part["preprocess_deps"]
        ]
        diff_deps = [pair for part in ordered for pair in part["diff_deps"]]
        differenced = canonical["differenced"]
        pre_fingerprint = content_fingerprint(
            {
                "differenced": native(differenced),
                "params": self.params_key,
            }
        )
        canonical_raw = {
            "trend": canonical["trend"],
            "seasonal": canonical["seasonal"],
            "adjusted": canonical["adjusted"],
            "differenced": canonical["differenced"],
            "preprocess_deps": preprocess_deps,
            "diff_deps": diff_deps,
        }
        return {
            "raw": canonical_raw,
            "fingerprint": pre_fingerprint,
            "fragments": [p["fragment_id"] for p in ordered],
            "reused_fragments": reused_fragments,
            "rebuilt_fragments": rebuilt_fragments,
            "elapsed": time.perf_counter() - started,
        }

    # ------------------------------------------------------------------
    # Stage 2: feature matrix
    # ------------------------------------------------------------------
    def run_features(self, series: SeriesData, pre_bundle: dict) -> dict:
        started = time.perf_counter()
        feat = build_feature_matrix(pre_bundle["raw"], self.params)
        feature_key = derived_key(
            ["feature-matrix", pre_bundle["fingerprint"], self.params_key]
        )
        cached = self.state.load(self.state.feature_dir, series.series_id, feature_key)
        if cached is not None:
            self.reused.append("features:matrix")
            cached["_key"] = feature_key
            cached["_live"] = feat
            self.stage_seconds["features"] += time.perf_counter() - started
            return cached

        payload = {
            "cache_key": feature_key,
            "feature_names": feat["feature_names"],
            "matrix_summary": matrix_summary(feat["matrix"]),
            "row_indices": native(feat["row_indices"]),
            "row_deps": native(feat["row_deps"]),
            "target_deps": native(feat["target_deps"]),
            "fingerprint": content_fingerprint(
                {
                    "matrix": native(feat["matrix"]),
                    "targets": native(feat["targets"]),
                    "names": feat["feature_names"],
                }
            ),
        }
        self.state.store(self.state.feature_dir, series.series_id, feature_key, payload)
        self.recomputed.append("features:matrix")
        payload["_key"] = feature_key
        payload["_live"] = feat
        self.stage_seconds["features"] += time.perf_counter() - started
        return payload

    # ------------------------------------------------------------------
    # Stage 3: per-window model fits
    # ------------------------------------------------------------------
    def run_models(self, series: SeriesData, pre_bundle: dict, feature_artifact: dict) -> dict:
        started = time.perf_counter()
        feat = feature_artifact["_live"]
        min_train = self.params["rolling_window"]
        model_entries: list[dict] = []
        for position in range(min_train, len(feat["row_indices"])):
            target_idx = feat["row_indices"][position]
            if target_idx + self.params["forecast_steps"] > len(series.values):
                break
            x_train = feat["matrix"][:position]
            y_train = feat["targets"][:position]
            data_key = content_fingerprint(
                {
                    "x": native(x_train),
                    "y": native(y_train),
                    "tier": self.tier_params,
                }
            )
            fold_key = derived_key(["model-fold", data_key])
            cached = self.state.load(self.state.model_dir, series.series_id, fold_key)
            if cached is not None:
                self.reused.append(f"model:{cached['fold_id']}")
                model_entries.append(cached)
                continue

            fit = fit_ols(x_train, y_train)
            trace = run_gradient_descent(x_train, y_train, self.tier_params)
            fold_id = f"fold-{position - min_train:05d}"
            payload = {
                "cache_key": fold_key,
                "fold_id": fold_id,
                "target_index": int(target_idx),
                "train_rows": int(position),
                "beta": [rounded_float(v) for v in fit["beta"].tolist()],
                "gd_loss_first": rounded_float(trace[0]["loss"]),
                "gd_loss_last": rounded_float(trace[-1]["loss"]),
                "gd_trace": [
                    {
                        "epoch": item["epoch"],
                        "loss": rounded_float(item["loss"]),
                        "params": [rounded_float(v) for v in item["params"].tolist()],
                    }
                    for item in trace
                ],
                "dependency": (
                    [
                        min(p[0] for p in feat["row_deps"][:position]),
                        max(p[1] for p in feat["row_deps"][:position]),
                    ]
                ),
            }
            self.state.store(self.state.model_dir, series.series_id, fold_key, payload)
            self.recomputed.append(f"model:{fold_id}")
            model_entries.append(payload)

        self.stage_seconds["models"] += time.perf_counter() - started
        return {"entries": model_entries, "min_train": min_train}

    # ------------------------------------------------------------------
    # Stage 4: rolling backtest
    # ------------------------------------------------------------------
    def run_backtest(
        self,
        series: SeriesData,
        pre_bundle: dict,
        feature_artifact: dict,
        model_bundle: dict,
    ) -> dict:
        started = time.perf_counter()
        feat = feature_artifact["_live"]
        x_raw = np.asarray(series.values, dtype=np.float64)
        horizon = self.params["forecast_steps"]
        fold_records: list[dict] = []
        all_actual: list[float] = []
        all_pred: list[float] = []
        for model_entry in model_bundle["entries"]:
            position = model_entry["train_rows"]
            target_idx = model_entry["target_index"]
            fold_key = derived_key(
                [
                    "backtest-fold",
                    model_entry["cache_key"],
                    feature_artifact["fingerprint"],
                    horizon,
                ]
            )
            cached = self.state.load(
                self.state.backtest_dir, series.series_id, fold_key
            )
            if cached is not None:
                self.reused.append(f"backtest:{cached['fold_id']}")
                fold_records.append(cached)
                all_actual.extend(cached["actuals"])
                all_pred.extend(cached["predictions"])
                continue

            # Reuse the persisted (canonical) beta rather than fitting a second
            # time, so recompute and cache-hit paths consume identical numbers.
            from .transforms import standardize_fit

            fit = {
                "beta": np.asarray(
                    [rounded_float(v) for v in model_entry["beta"]], dtype=np.float64
                ),
                "scaler": standardize_fit(feat["matrix"][:position]),
            }
            from .model import _forecast_raw

            predictions = _forecast_raw(
                pre_bundle["raw"],
                feat,
                fit,
                target_idx,
                horizon,
                x_raw,
                self.params,
            )
            actuals = [
                float(x_raw[target_idx + step]) for step in range(horizon)
            ]
            from .model import error_metrics

            payload = {
                "cache_key": fold_key,
                "fold_id": model_entry["fold_id"],
                "target_index": int(target_idx),
                "predictions": [rounded_float(v) for v in predictions],
                "actuals": [rounded_float(v) for v in actuals],
                "metrics": {
                    key: rounded_float(value)
                    for key, value in error_metrics(
                        np.asarray(actuals), np.asarray(predictions)
                    ).items()
                },
                "dependency": model_entry["dependency"],
            }
            self.state.store(
                self.state.backtest_dir, series.series_id, fold_key, payload
            )
            self.recomputed.append(f"backtest:{payload['fold_id']}")
            fold_records.append(payload)
            all_actual.extend(payload["actuals"])
            all_pred.extend(payload["predictions"])

        from .model import error_metrics

        aggregate = (
            {
                key: rounded_float(value)
                for key, value in error_metrics(
                    np.asarray(all_actual), np.asarray(all_pred)
                ).items()
            }
            if fold_records
            else {"mae": 0.0, "rmse": 0.0, "mape": 0.0}
        )
        self.stage_seconds["backtest"] += time.perf_counter() - started
        return {
            "folds": sorted(fold_records, key=lambda f: f["fold_id"]),
            "aggregate_metrics": native(aggregate),
        }

    # ------------------------------------------------------------------
    # Full per-series orchestration
    # ------------------------------------------------------------------
    def process(self, series: SeriesData) -> dict:
        pre_bundle = self.run_preprocess(series)
        self.stage_seconds["preprocess"] += pre_bundle["elapsed"]
        feature_artifact = self.run_features(series, pre_bundle)
        model_bundle = self.run_models(series, pre_bundle, feature_artifact)
        backtest = self.run_backtest(
            series, pre_bundle, feature_artifact, model_bundle
        )

        coefficient_summary = [
            {
                "fold_id": entry["fold_id"],
                "target_index": entry["target_index"],
                "beta": native(entry["beta"]),
                "gd_loss_first": native(entry["gd_loss_first"]),
                "gd_loss_last": native(entry["gd_loss_last"]),
            }
            for entry in model_bundle["entries"]
        ]
        return {
            "series_id": series.series_id,
            "data_fingerprint": series_fingerprint(series),
            "preprocess": {
                "fingerprint": pre_bundle["fingerprint"],
                "fragments": pre_bundle["fragments"],
                "differenced_summary": matrix_summary(
                    pre_bundle["raw"]["differenced"]
                ),
            },
            "features": {
                "cache_key": feature_artifact["cache_key"],
                "feature_names": feature_artifact["feature_names"],
                "matrix_summary": feature_artifact["matrix_summary"],
                "row_count": len(feature_artifact["row_indices"]),
            },
            "models": coefficient_summary,
            "backtest": backtest,
            "audit": {
                "reused": sorted(self.reused),
                "recomputed": sorted(self.recomputed),
                "preprocess_reused_fragments": pre_bundle["reused_fragments"],
                "preprocess_rebuilt_fragments": pre_bundle["rebuilt_fragments"],
            },
            "stage_seconds": native(self.stage_seconds),
        }
