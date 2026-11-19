"""多阶段增量渲染引擎。

核心职责
--------
* 依据配置构造阶段 DAG 与每阶段缓存键（上游内容指纹 + 参数指纹）。
* 命中持久化产物则直接复用且不重新计算，未命中阶段纳入稳定排序的重算集合。
* 线程池并发执行可并行阶段（去噪链与逐帧解码），最大并发数受
  ``max_workers`` 限制；所有清单按稳定排序键输出，与单线程逐字段一致。
* 单阶段异常隔离到 failure 清单并继续其余可执行阶段。
* 输出含耗时、复用/重算统计、注入命中编号的 JSON 诊断报告。
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .config import param_fingerprint_inputs
from .errors import (
    E_STAGE_FAILED,
    WorkbenchError,
    fail_record,
)
from .hashing import cache_key, fingerprint
from .persistence import SnapshotStore
from .stages import (
    denoise_step,
    StageSpec,
    apply_patch,
    build_stage_specs,
    decode_frame,
    encode_init_image,
    encode_prompt,
    init_latent,
    replaced_denoise_steps,
)


@dataclass(frozen=True)
class StageNode:
    spec: StageSpec
    deps: tuple[str, ...]
    param_keys: tuple[str, ...]


class RenderEngine:
    def __init__(
        self,
        config: Mapping[str, Any],
        store: SnapshotStore,
        injections: list[Mapping[str, Any]] | None = None,
        signal_gate: Any | None = None,
    ) -> None:
        self.cfg = dict(config)
        self.store = store
        self.injections = list(injections or [])
        self.signal_gate = signal_gate
        self.steps = int(self.cfg["steps"])
        self.frames = int(self.cfg["frames"])
        self.mode = self.cfg["mode"]
        self.replace_steps = int(self.cfg.get("replace_steps", 0))
        self.replaced = replaced_denoise_steps(self.mode, self.replace_steps, self.steps)
        self.specs = build_stage_specs(
            self.steps, self.frames, self.mode, self.replace_steps
        )
        self.spec_by_name = {spec.name: spec for spec in self.specs}
        self.nodes = self._build_graph()
        self.param_fps = self._parameter_fingerprints()
        self.config_hash = fingerprint(param_fingerprint_inputs(self.cfg))
        self._io_lock = threading.Lock()
        self._commit_lock = threading.Lock()

    # ---- 阶段 DAG 与参数指纹 -----------------------------------------------
    def _build_graph(self) -> dict[str, StageNode]:
        nodes: dict[str, StageNode] = {}
        nodes["encode"] = StageNode(self.spec_by_name["encode"], (), ("prompt",))
        if self.mode == "i2v":
            nodes["image_encode"] = StageNode(
                self.spec_by_name["image_encode"], (), ("init_image",)
            )
        else:
            nodes["latent_init"] = StageNode(
                self.spec_by_name["latent_init"],
                ("encode",),
                ("seed", "resolution"),
            )
        patch_steps = {int(item["step"]) for item in self.cfg.get("patches", [])}
        for index in range(self.steps):
            name = f"denoise_{index:03d}"
            if index == 0:
                deps = (
                    ["image_encode"]
                    if self.mode == "i2v" and 0 in self.replaced
                    else ["encode", "latent_init"]
                )
            else:
                deps = ["encode", f"denoise_{index - 1:03d}"]
            # 单步数学只依赖该步噪声表条目、引导与种子；steps 变化通过
            # 噪声表长度与 DAG 结构生效，不直接进入单步缓存键。
            param_keys = ["seed", "guidance", "noise_table"]
            if index in patch_steps:
                param_keys.append("patches")
            if self.mode == "i2v":
                param_keys.append("replace_steps")
            nodes[name] = StageNode(
                self.spec_by_name[name], tuple(deps), tuple(sorted(set(param_keys)))
            )
        last_denoise = f"denoise_{self.steps - 1:03d}"
        for index in range(self.frames):
            name = f"decode_{index:03d}"
            nodes[name] = StageNode(
                self.spec_by_name[name],
                (last_denoise,),
                ("seed", "frames", "resolution"),
            )
        return nodes

    def _parameter_fingerprints(self) -> dict[str, str]:
        cfg = self.cfg
        patch_by_step = {int(item["step"]): item for item in cfg.get("patches", [])}
        views = {
            "prompt": fingerprint(cfg["prompt"]),
            "seed": fingerprint(cfg["seed"]),
            "steps": fingerprint(cfg["steps"]),
            "guidance": fingerprint(cfg["guidance"]),
            "frames": fingerprint(cfg["frames"]),
            "resolution": fingerprint(
                [cfg["resolution"], cfg["height"], cfg["width"]]
            ),
            "noise_table": fingerprint(
                [int(cfg["noise_table"].shape[0]), cfg["noise_table"]]
            ),
            "init_image": fingerprint(cfg["init_image"]),
            "init_image_path": fingerprint(cfg.get("init_image_path")),
            "replace_steps": fingerprint(cfg.get("replace_steps", 0)),
            "patches": fingerprint(
                [
                    {
                        "step": step,
                        "top": entry["patch"]["top"],
                        "left": entry["patch"]["left"],
                        "values": np.asarray(entry["patch"]["values"], dtype=np.float64),
                    }
                    for step, entry in sorted(patch_by_step.items())
                ]
            ),
        }
        return views

    def stage_cache_key(
        self, stage_name: str, dependency_fps: Mapping[str, str]
    ) -> str:
        node = self.nodes[stage_name]
        param_payload = {key: self.param_fps[key] for key in node.param_keys}
        return cache_key(stage_name, fingerprint(param_payload), dependency_fps)

    # ---- 主运行入口：DAG 线程池调度 ----------------------------------------
    def run(self, *, serial: bool = False) -> dict[str, Any]:
        started = time.perf_counter()
        previous_manifest = self.store.load_committed_manifest()
        global_version = int((previous_manifest or {}).get("global_version", 0)) + 1
        self.store.open_staging(
            global_version=global_version,
            config_hash=self.config_hash,
            stage_specs=self.specs,
        )

        reused: list[str] = []
        recomputed: list[str] = []
        failures: list[dict[str, Any]] = []
        timings: dict[str, float] = {}
        injection_hits: list[dict[str, Any]] = []
        content_fps: dict[str, str] = {}
        failed_stages: set[str] = set()
        remaining = set(self.nodes)
        completed: set[str] = set()
        pending_failures: list[dict[str, Any]] = []
        max_workers = 1 if serial else max(1, int(self.cfg["max_workers"]))
        pool = ThreadPoolExecutor(max_workers=max_workers)
        snapshot_dir: str | None = None

        def ready_tasks() -> list[str]:
            tasks = []
            for name in sorted(remaining, key=self._sort_key_of):
                node = self.nodes[name]
                if all(dep in completed for dep in node.deps):
                    tasks.append(name)
            return tasks

        try:
            while remaining:
                wave: list[tuple[str, str, dict, Any]] = []
                for name in ready_tasks():
                    node = self.nodes[name]
                    spec = node.spec
                    if any(dep in failed_stages for dep in node.deps):
                        failed_stages.add(name)
                        remaining.discard(name)
                        completed.add(name)
                        pending_failures.append(
                            fail_record(
                                E_STAGE_FAILED,
                                "上游阶段失败，本阶段跳过",
                                stage=name,
                                step=spec.step,
                                frame=spec.frame,
                                exception_type="UpstreamSkipped",
                            )
                        )
                        continue
                    dep_fps = {dep: content_fps[dep] for dep in node.deps}
                    key = self.stage_cache_key(name, dep_fps)
                    cached = self.store.load_staged_artifact(name)
                    if cached is not None and cached.get("cache_key") == key:
                        with self._io_lock:
                            content_fps[name] = cached["content_fingerprint"]
                            reused.append(name)
                        remaining.discard(name)
                        completed.add(name)
                        continue
                    recomputed.append(name)
                    remaining.discard(name)
                    hit = self._matching_injection(name)
                    if hit is not None:
                        injection_hits.append(
                            {"id": hit.get("id"), "stage": name,
                             "type": hit.get("type")}
                        )
                    wave.append((name, key, dep_fps, hit))

                if not wave:
                    # 全部剩余任务都被上游失败隔离。
                    continue

                futures = {
                    name: pool.submit(self._task_wrapper, name, key, dep_fps, hit)
                    for name, key, dep_fps, hit in wave
                }
                # 波次屏障：等待本波全部任务结束，确保提交期间无工作线程写盘。
                wave_results = []
                for name, future in futures.items():
                    wave_results.append((name, future.result()))
                # 结果按稳定排序键落盘与提交，线程调度顺序不影响产物顺序。
                wave_results.sort(key=lambda pair: self._sort_key_of(pair[0]))
                for finished_name, result in wave_results:
                    completed.add(finished_name)
                    if result["ok"]:
                        content_fps[finished_name] = result["content_fingerprint"]
                        timings[finished_name] = result["elapsed"]
                        with self._commit_lock:
                            self.store.commit_stage()
                    else:
                        failed_stages.add(finished_name)
                        timings[finished_name] = result["elapsed"]
                        pending_failures.append(result["failure"])
                    if self.signal_gate is not None:
                        self.signal_gate.after_stage(finished_name, self.store)
            failures.extend(pending_failures)
            snapshot_dir = self.store.commit()
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            self.store.discard_staging()
            raise
        else:
            pool.shutdown(wait=True)

        report = {
            "status": "failed" if failures else "ok",
            "serial": serial,
            "max_workers": max_workers,
            "global_version": global_version,
            "config_hash": self.config_hash,
            "snapshot_dir": snapshot_dir,
            "mode": self.mode,
            "reused": sorted(reused, key=self._sort_key_of),
            "recomputed": sorted(recomputed, key=self._sort_key_of),
            "failures": sorted(failures, key=self._failure_sort_key),
            "timings": {
                name: timings[name]
                for name in sorted(timings, key=self._sort_key_of)
            },
            "injection_hits": sorted(
                injection_hits, key=lambda item: self._sort_key_of(item["stage"])
            ),
            "stats": {
                "stage_total": len(self.specs),
                "reused_count": len(reused),
                "recomputed_count": len(recomputed),
                "failure_count": len(failures),
                "elapsed_seconds": round(time.perf_counter() - started, 6),
            },
        }
        return report

    def _task_wrapper(
        self,
        stage_name: str,
        key: str,
        dep_fps: Mapping[str, str],
        injection: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        spec = self.nodes[stage_name].spec
        node = self.nodes[stage_name]
        started = time.perf_counter()
        if injection is not None and injection.get("type") == "hard_kill":
            # 先让本阶段正常计算并完整落盘，再由 signal_gate 在阶段边界
            # 执行不可捕获的 os._exit，模拟 SIGKILL。
            envelope, array = self._compute(spec, None)
            envelope.update(
                {
                    "cache_key": key,
                    "dependency_fingerprints": {
                        dep: dep_fps[dep] for dep in sorted(dep_fps)
                    },
                    "param_fingerprints": {
                        name: self.param_fps[name] for name in node.param_keys
                    },
                }
            )
            with self._io_lock:
                self.store.stage_artifact(spec, envelope, array)
            with self._commit_lock:
                self.store.commit_stage()
            if self.signal_gate is not None:
                self.signal_gate.after_stage(stage_name, self.store)
            return {
                "ok": True,
                "content_fingerprint": "hard-kill-no-return",
                "elapsed": round(time.perf_counter() - started, 6),
            }
        try:
            envelope, array = self._compute(spec, injection)
            envelope.update(
                {
                    "cache_key": key,
                    "dependency_fingerprints": {
                        dep: dep_fps[dep] for dep in sorted(dep_fps)
                    },
                    "param_fingerprints": {
                        name: self.param_fps[name] for name in node.param_keys
                    },
                }
            )
            with self._io_lock:
                record = self.store.stage_artifact(spec, envelope, array)
            return {
                "ok": True,
                "content_fingerprint": record["content_fingerprint"],
                "elapsed": round(time.perf_counter() - started, 6),
            }
        except WorkbenchError as error:
            return {
                "ok": False,
                "elapsed": round(time.perf_counter() - started, 6),
                "failure": error.to_record(exception_type=type(error).__name__),
            }
        except Exception as error:
            return {
                "ok": False,
                "elapsed": round(time.perf_counter() - started, 6),
                "failure": fail_record(
                    E_STAGE_FAILED,
                    str(error),
                    stage=stage_name,
                    step=spec.step,
                    frame=spec.frame,
                    exception_type=type(error).__name__,
                    injection_id=(injection or {}).get("id"),
                ),
            }

    def _sort_key_of(self, stage_name: str) -> tuple[str, int, int]:
        return self.spec_by_name[stage_name].sort_key

    @staticmethod
    def _failure_sort_key(record: Mapping[str, Any]) -> tuple[str, int, int]:
        stage = record.get("stage") or ""
        return (
            stage,
            record.get("step") if record.get("step") is not None else -1,
            record.get("frame") if record.get("frame") is not None else -1,
        )

    def _matching_injection(self, stage_name: str) -> Mapping[str, Any] | None:
        for injection in self.injections:
            if injection.get("stage") == stage_name:
                return injection
        return None

    # ---- 阶段计算 -----------------------------------------------------------
    def _compute(
        self, spec: StageSpec, injection: Mapping[str, Any] | None
    ) -> tuple[dict[str, Any], np.ndarray | None]:
        self._maybe_raise_injection(injection)
        cfg = self.cfg
        if spec.kind == "encode":
            result = encode_prompt(cfg["prompt"], cfg["seed"])
            envelope = {
                "tokens": result["tokens"],
                "in_vocab": result["in_vocab"],
                "out_of_vocab": result["out_of_vocab"],
                "summary": _vector_summary(result["embedding"]),
            }
            return envelope, result["embedding"]

        if spec.kind == "image_encode":
            result = encode_init_image(cfg["init_image"], cfg["seed"])
            return {}, result["image_latent"]

        if spec.kind == "latent_init":
            prompt_artifact = self._must_load("encode")
            result = init_latent(
                prompt_artifact["content_fingerprint"],
                cfg["seed"],
                cfg["height"],
                cfg["width"],
            )
            return {}, result["latent"]

        if spec.kind == "denoise":
            return self._compute_denoise(spec)

        if spec.kind == "decode":
            return self._compute_decode(spec)

        raise WorkbenchError(E_STAGE_FAILED, f"未知阶段类型: {spec.kind}", stage=spec.name)

    def _compute_denoise(self, spec: StageSpec) -> tuple[dict[str, Any], np.ndarray]:
        cfg = self.cfg
        index = spec.step
        assert index is not None
        prompt_artifact = self._must_load("encode")
        prompt_embedding = prompt_artifact["array"]
        patch_by_step = {int(item["step"]): item for item in cfg.get("patches", [])}

        if index in self.replaced:
            image_artifact = self._must_load("image_encode")
            latent = np.array(image_artifact["array"], copy=True)
            origin = "image_latent_replace"
        else:
            if index == 0:
                previous_artifact = self._must_load("latent_init")
            else:
                previous_artifact = self._must_load(f"denoise_{index - 1:03d}")
            latent = denoise_step(
                previous_artifact["array"],
                prompt_embedding,
                index,
                cfg["steps"],
                cfg["guidance"],
                cfg["noise_table"][index],
                cfg["seed"],
            )
            origin = "denoise"

        patch_applied = False
        if index in patch_by_step:
            entry = patch_by_step[index]
            latent = apply_patch(latent, entry["patch"], stage=spec.name, step=index)
            patch_applied = True

        return (
            {
                "step": index,
                "origin": origin,
                "replaced": index in self.replaced,
                "patch_applied": patch_applied,
            },
            latent,
        )

    def _compute_decode(self, spec: StageSpec) -> tuple[dict[str, Any], np.ndarray]:
        cfg = self.cfg
        frame_index = spec.frame
        assert frame_index is not None
        last_denoise = self._must_load(f"denoise_{cfg['steps'] - 1:03d}")
        frame = decode_frame(
            last_denoise["array"],
            frame_index,
            cfg["frames"],
            cfg["seed"],
            cfg["height"],
            cfg["width"],
        )
        return {"frame": frame_index}, frame

    def _must_load(self, stage_name: str) -> dict[str, Any]:
        with self._io_lock:
            artifact = self.store.load_staged_artifact(stage_name)
        if artifact is None:
            raise WorkbenchError(
                E_STAGE_FAILED,
                f"依赖阶段 {stage_name} 的产物缺失",
                stage=stage_name,
            )
        return artifact

    @staticmethod
    def _maybe_raise_injection(injection: Mapping[str, Any] | None) -> None:
        if injection is None:
            return
        injection_type = injection.get("type")
        if injection_type == "stage_exception":
            raise WorkbenchError(
                injection.get("code", E_STAGE_FAILED),
                injection.get("message", "注入的单阶段异常"),
                stage=injection.get("stage"),
                injection_id=injection.get("id"),
            )
        if injection_type == "runtime_exception":
            raise RuntimeError(injection.get("message", "注入的运行时异常"))

    # ---- 独立并发解码（用于并发/串行一致性自检） ----------------------------
    def run_concurrent_decode(self) -> list[dict[str, Any]]:
        last_name = f"denoise_{self.steps - 1:03d}"
        last_artifact = self._must_load(last_name)
        max_workers = min(max(1, int(self.cfg["max_workers"])), max(1, self.frames))

        def worker(frame_index: int) -> tuple[int, np.ndarray]:
            frame = decode_frame(
                last_artifact["array"],
                frame_index,
                self.frames,
                self.cfg["seed"],
                self.cfg["height"],
                self.cfg["width"],
            )
            return frame_index, frame

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(worker, range(self.frames)))
        results.sort(key=lambda item: item[0])
        return [{"frame": index, "pixels": frame} for index, frame in results]


def _vector_summary(array: np.ndarray) -> dict[str, Any]:
    cast = array.astype(np.float64)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(cast.min()),
        "max": float(cast.max()),
        "mean": float(cast.mean()),
    }
