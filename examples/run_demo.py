"""完整示例：编辑序列下增量运行 vs 空白工作区全量运行的逐字段对比。

编辑序列覆盖：词条新增、词频调整、词条删除、概率表替换、单句文本替换。
"""

from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from segpos import ComponentConfig, Dictionary, ProbabilityTable, IncrementalPipeline  # noqa: E402
from segpos.pipeline import load_sentences  # noqa: E402

TAGS = ["n", "v", "a", "d"]


def build_pipeline(config_path, dict_path, table_path, snapshot_dir,
                   sentences_path, table_version=1):
    config = ComponentConfig.load(config_path)
    dictionary = Dictionary.load(dict_path, allowed_pos=set(TAGS))
    table = ProbabilityTable.load(table_path, version=table_version)
    return IncrementalPipeline(
        config, dictionary, table, snapshot_dir, sentences_path
    )


def write_sentences(path: Path, sentences: dict) -> None:
    lines = [f"{sid}\t{text}" for sid, text in sorted(sentences.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def perturb_table(table_data: dict) -> dict:
    """构造内容不同、结构合法的第二版概率表。"""

    data = copy.deepcopy(table_data)
    for head, mapping in data["trans_logprob"].items():
        tags = list(mapping)
        values = [mapping[tag] for tag in tags]
        shifted = values[1:] + values[:1]
        data["trans_logprob"][head] = dict(zip(tags, shifted))
    return data


def comparable_results(pipe: IncrementalPipeline) -> dict:
    results = pipe.results_by_id()
    return {
        sid: {
            "tokens": [
                {
                    "start": token["start"],
                    "end": token["end"],
                    "text": token["text"],
                    "match_basis": token["match_basis"],
                    "hit": token["hit"],
                    "boundary_confidence": token["boundary_confidence"],
                    "pos": token["pos"],
                    "local_path_score": token["local_path_score"],
                }
                for token in payload["tokens"]
            ],
            "pos_sequence": payload["pos_sequence"],
            "sentence_logprob": payload["sentence_logprob"],
            "smoothing_hits": payload["smoothing_hits"],
            "low_confidence_words": payload["low_confidence_words"],
        }
        for sid, payload in sorted(results.items())
    }


def comparable_failures(pipe: IncrementalPipeline) -> list:
    return sorted(
        [
            {
                "sentence_id": item["sentence_id"],
                "stage": item["stage"],
                "code": item["code"],
                "exception_type": item["exception_type"],
            }
            for item in pipe.failures
        ],
        key=lambda item: (item["sentence_id"], item["stage"]),
    )


def main() -> int:
    sample_dir = ROOT / "examples" / "sample_data"
    output_dir = ROOT / "examples" / "output" / "demo"
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = sample_dir / "config.json"
    dict_path = sample_dir / "dictionary.txt"
    table_path = sample_dir / "prob_table.json"
    base_table = json.loads(table_path.read_text(encoding="utf-8"))

    # ---------- 增量工作区：从基线开始，逐步施加编辑 ----------
    inc_snap = output_dir / "inc_snapshot"
    inc_sentences = output_dir / "inc_sentences.txt"
    shutil.copyfile(sample_dir / "sentences.txt", inc_sentences)
    pipe = build_pipeline(
        config_path, dict_path, table_path, inc_snap, inc_sentences
    )
    baseline = pipe.run(output_dir / "report_baseline.json")
    print("[1/6] 基线全量完成:",
          f"成功 {baseline.report['success_sentences']} 句,",
          f"失败 {baseline.report['failed_sentences']} 句")

    invalidated_1 = pipe.add_word("训练模型", 66, "n")
    print("[2/6] 新增词条 '训练模型'，失效句子:", invalidated_1)
    run_1 = pipe.run(output_dir / "report_edit1.json")
    print("      复用", run_1.reused, "重算", run_1.recomputed)

    invalidated_2 = pipe.update_freq("学习", 200)
    print("[3/6] 调整词频 '学习'->200，失效句子:", invalidated_2)
    run_2 = pipe.run(output_dir / "report_edit2.json")
    print("      复用", run_2.reused, "重算", run_2.recomputed)

    invalidated_3 = pipe.remove_word("野外")
    print("[4/6] 删除词条 '野外'，失效句子:", invalidated_3)
    run_3 = pipe.run(output_dir / "report_edit3.json")
    print("      复用", run_3.reused, "重算", run_3.recomputed)

    table_v2 = perturb_table(base_table)
    table_v2_path = output_dir / "prob_table_v2.json"
    table_v2_path.write_text(
        json.dumps(table_v2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    invalidated_4 = pipe.replace_prob_table(
        ProbabilityTable(table_v2, version=2)
    )
    print("[5/6] 替换概率表 v2，失效句子数:", len(invalidated_4))
    run_4 = pipe.run(output_dir / "report_edit4.json")
    print("      复用", run_4.reused, "重算", run_4.recomputed)

    inputs = load_sentences(inc_sentences)
    final_sentences = dict(inputs.sentences)
    final_sentences["S006"] = "它们在山谷里缓慢行走"
    write_sentences(inc_sentences, final_sentences)
    invalidated_5 = pipe.replace_sentence_text("S006", final_sentences["S006"])
    print("[6/6] 替换单句文本 S006，失效:", invalidated_5)
    run_5 = pipe.run(output_dir / "report_edit5.json")
    print("      复用", run_5.reused, "重算", run_5.recomputed)

    # ---------- 全量对照工作区：相同最终输入，从空白一次性运行 ----------
    full_dict_path = output_dir / "dictionary_final.txt"
    final_entries = [
        f"{entry.word} {entry.freq} {entry.pos}"
        for entry in pipe.dictionary.to_entries()
    ]
    full_dict_path.write_text("\n".join(final_entries) + "\n", encoding="utf-8")
    full_sentences = output_dir / "sentences_final.txt"
    write_sentences(full_sentences, final_sentences)

    full_pipe = build_pipeline(
        config_path,
        full_dict_path,
        table_v2_path,
        output_dir / "full_snapshot",
        full_sentences,
        table_version=2,
    )
    full_outcome = full_pipe.run(output_dir / "report_full.json")

    inc_results = comparable_results(pipe)
    full_results = comparable_results(full_pipe)
    inc_failures = comparable_failures(pipe)
    full_failures = comparable_failures(full_pipe)

    match = (
        inc_results == full_results
        and inc_failures == full_failures
        and list(inc_results) == list(full_results)
    )
    print("\n=== 逐字段一致性 ===")
    print("句子顺序一致:", list(inc_results) == list(full_results))
    print("分词/词性/得分逐字段一致:", inc_results == full_results)
    print("失败清单顺序与内容一致:", inc_failures == full_failures)
    print("最终判定:", "PASS" if match else "FAIL")

    (output_dir / "incremental_results.json").write_text(
        json.dumps(inc_results, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "full_results.json").write_text(
        json.dumps(full_results, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if not match:
        return 1
    print("\n最终全量报告:", json.dumps(full_outcome.report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
