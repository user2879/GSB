"""端到端演示与验证：合成数据、全量/增量一致性、快照恢复、注入业务码。"""

import json
import os
import shutil

import numpy as np

from .config import PipelineConfig
from .injection import InjectionRegistry
from .pipeline import DiarizationPipeline
from .synth import generate_dataset, write_wav, build_timeline, mix_timeline


def _load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, sort_keys=True)


def _load_inputs(input_dir):
    config = PipelineConfig.from_dict(_load_json(os.path.join(input_dir, "config.json")))
    truth = _load_json(os.path.join(input_dir, "truth.json"))
    injections = _load_json(os.path.join(input_dir, "injections.json"))
    return config, truth, injections


def _run_once(input_dir, out_dir, allow_edit=False, fresh=False):
    config, truth, injections = _load_inputs(input_dir)
    registry = InjectionRegistry(injections)
    pipeline = DiarizationPipeline(input_dir, out_dir, config, truth, registry,
                                   allow_edit=allow_edit, fresh=fresh)
    return pipeline.run()


def _comparable(report):
    """剥离运行态字段（缓存动作、快照版本），只比较逐字段业务结果。"""
    results = {}
    for audio, item in report["results"].items():
        results[audio] = {
            "segments": item["segments"],
            "boundary_records": item["boundary_records"],
            "clusters": item["clusters"],
            "adjudications": item["adjudications"],
            "speaker_intervals": item["speaker_intervals"],
            "verification": item["verification"],
        }
    return {
        "audio_order": report["audio_order"],
        "results": results,
        "failures": report["failures"],
    }


def _replace_audio(input_dir, truth, name, seed):
    rng = np.random.default_rng(seed)
    index = int(name.split("_")[1].split(".")[0])
    n_speakers = 2 + (index % 2)
    segments = build_timeline(rng, n_speakers, 4000)
    samples = mix_timeline(segments, 4000, rng)
    write_wav(os.path.join(input_dir, name), samples)
    from .synth import build_truth
    truth[name] = build_truth(segments, n_speakers)
    _write_json(os.path.join(input_dir, "truth.json"), truth)


def _run_rule_isolated(input_dir, workspace, rule):
    """在独立输出目录用指定重叠规则运行，其余配置与当前输入完全一致。"""
    config, truth, injections = _load_inputs(input_dir)
    import dataclasses
    isolated = dataclasses.replace(config, overlap_rule=rule)
    registry = InjectionRegistry(injections)
    out_dir = os.path.join(workspace, f"rule_isolated_{rule}")
    pipeline = DiarizationPipeline(input_dir, out_dir, isolated, truth, registry,
                                   allow_edit=False, fresh=True)
    return pipeline.run()


def run_full_demo(workspace):
    input_dir = os.path.join(workspace, "input")
    out_dir = os.path.join(workspace, "output")
    report_dir = os.path.join(workspace, "reports")
    if os.path.exists(workspace):
        shutil.rmtree(workspace)
    os.makedirs(workspace, exist_ok=True)
    generate_dataset(input_dir)

    # 1) 全量运行
    full = _run_once(input_dir, out_dir, fresh=True)
    _write_json(os.path.join(report_dir, "01_full_run.json"), full)

    # 2) 无改动续跑：应全部命中缓存复用
    resume = _run_once(input_dir, out_dir)
    _write_json(os.path.join(report_dir, "02_resume_reuse.json"), resume)

    # 3) 编辑序列下的增量运行
    config, truth, injections = _load_inputs(input_dir)
    edit_dir = os.path.join(workspace, "edit_output")
    shutil.copytree(out_dir, edit_dir)

    # 3a 替换单段音频
    target = "mix_006.wav"  # 无注入的干净音频，适合替换
    _replace_audio(input_dir, truth, target, seed=999)
    edit_replace = _run_once(input_dir, edit_dir, allow_edit=True)
    _write_json(os.path.join(report_dir, "03_edit_replace.json"), edit_replace)

    # 3b 调整能量门限
    config.energy_threshold = 0.03
    config.save(os.path.join(input_dir, "config.json"))
    edit_energy = _run_once(input_dir, edit_dir, allow_edit=True)
    _write_json(os.path.join(report_dir, "04_edit_energy.json"), edit_energy)

    # 3c 调整聚类阈值
    config.cluster_distance_threshold = 0.6
    config.save(os.path.join(input_dir, "config.json"))
    edit_cluster = _run_once(input_dir, edit_dir, allow_edit=True)
    _write_json(os.path.join(report_dir, "05_edit_cluster.json"), edit_cluster)

    # 3d 重叠归属规则变更（同时保留一份与 05 仅规则不同的对照，便于直接比对）
    config.overlap_rule = "max_margin"
    config.save(os.path.join(input_dir, "config.json"))
    edit_rule = _run_once(input_dir, edit_dir, allow_edit=True)
    _write_json(os.path.join(report_dir, "06_edit_rule.json"), edit_rule)
    rule_only_nearest = _run_rule_isolated(input_dir, workspace, "nearest")
    rule_only_margin = _run_rule_isolated(input_dir, workspace, "max_margin")
    _write_json(os.path.join(report_dir, "06b_rule_nearest.json"), rule_only_nearest)
    _write_json(os.path.join(report_dir, "06c_rule_max_margin.json"), rule_only_margin)

    # 3e 量化档位切换（恢复默认 nearest 规则，保证最终基准使用高正确率裁决）
    config.overlap_rule = "nearest"
    config.frac_bits = 8
    config.save(os.path.join(input_dir, "config.json"))
    edit_gear = _run_once(input_dir, edit_dir, allow_edit=True)
    _write_json(os.path.join(report_dir, "07_edit_gear.json"), edit_gear)

    # 3f 新增一段音频
    extra_name = "mix_007.wav"
    rng = np.random.default_rng(777)
    extra_segments = build_timeline(rng, 3, 4000)
    write_wav(os.path.join(input_dir, extra_name),
              mix_timeline(extra_segments, 4000, rng))
    from .synth import build_truth
    truth[extra_name] = build_truth(extra_segments, 3)
    _write_json(os.path.join(input_dir, "truth.json"), truth)
    edit_add = _run_once(input_dir, edit_dir, allow_edit=True)
    _write_json(os.path.join(report_dir, "08_edit_add.json"), edit_add)

    # 4) 从空白工作区按最终状态一次性全量运行（基准）
    canonical_dir = os.path.join(workspace, "canonical_output")
    canonical = _run_once(input_dir, canonical_dir, fresh=True)
    _write_json(os.path.join(report_dir, "09_canonical_full.json"), canonical)

    rule_diffs = []
    for audio in rule_only_margin["audio_order"]:
        near = {a["frame_index"]: a["assigned_cluster"]
                for a in rule_only_nearest["results"][audio]["adjudications"]}
        margin = {a["frame_index"]: a["assigned_cluster"]
                  for a in rule_only_margin["results"][audio]["adjudications"]}
        rule_diffs.append({
            "audio": audio,
            "changed_frames": sum(1 for k in near if near[k] != margin.get(k)),
        })
    consistency = {
        "incremental_equals_canonical":
            _comparable(edit_add) == _comparable(canonical),
        "rule_switch_changed_frames": rule_diffs,
        "recompute_counts": {
            "resume": resume["stats"]["compute"],
            "after_replace": edit_replace["stats"]["compute"],
            "after_energy": edit_energy["stats"]["compute"],
            "after_cluster": edit_cluster["stats"]["compute"],
            "after_rule": edit_rule["stats"]["compute"],
            "after_gear": edit_gear["stats"]["compute"],
            "after_add": edit_add["stats"]["compute"],
        },
        "reuse_counts": {
            "resume": resume["stats"]["reuse"],
            "after_replace": edit_replace["stats"]["reuse"],
            "after_energy": edit_energy["stats"]["reuse"],
            "after_cluster": edit_cluster["stats"]["reuse"],
            "after_rule": edit_rule["stats"]["reuse"],
            "after_gear": edit_gear["stats"]["reuse"],
            "after_add": edit_add["stats"]["reuse"],
        },
        "injection_failures": canonical["failures"],
    }
    _write_json(os.path.join(report_dir, "10_consistency_summary.json"), consistency)
    return consistency, canonical


def main():
    import sys
    workspace = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.getcwd(), "demo_workspace")
    consistency, canonical = run_full_demo(workspace)
    print(json.dumps(consistency, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
