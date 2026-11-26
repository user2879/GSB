# IntentLab

IntentLab 是一个本地可重复复核的搜索查询理解诊断组件。它在单一 Python 进程中完成受控词表样本生成、规则修订合成、定点触发匹配、意图打分、相邻查询合并、槽位对齐、float64 误差归因、内容指纹增量复用、预算调度与阶段化快照恢复。

项目不访问网络、不使用容器、不依赖 GPU。运行时依赖 Python 3.11、NumPy 和 Tkinter（Ubuntu 包 `python3-tk`）。

## 快速开始

```bash
cd /home/zsl/feedback/uidfc5909d3-default-404
python3 scripts/generate_samples.py
python3 -m unittest discover -s tests -v
python3 -m intentlab validate
python3 -m intentlab run
python3 -m intentlab run          # 第二次运行命中持久化缓存
python3 -m intentlab recover
python3 -m intentlab gui --report reports/latest.json
```

产物位置：

- 输入：`input/config.json`、`input/vocabulary.json`、`input/queries.json`、`input/rules.json`、`input/edits.json`
- 已提交快照：`snapshots/generations/<generation>/`，可加载代由 `snapshots/current.json` 唯一指定
- 报告：`reports/latest.json`、`reports/latest.md`，以及按运行代命名的副本

## 已实现能力

- 查询、词表、槽位、修订树、阈值、定点枚举和编辑命令流逐项校验，按业务码拒绝非法部分并保留有效部分。
- 修订链逐级合成有效规则视图，记录触发词最终生效修订、覆盖来源和完整 provenance；支持祖先回滚与兄弟分支非冲突合并。
- 字符跨度最长优先、同长词项字典序、再按起始位置升序的贪心匹配，输出被吞并候选的抑制清单。
- 定点整数打分严格按意图标识和触发词条目序累加，支持阈值边界、量化半步与固定 tie 规则。
- 相邻且意图一致的上下文查询合并，槽位跨查询映射到统一 term 偏移空间，冲突按固定优先级裁决并留痕。
- 固定点、float64 无量化参考、高一位宽参考三路对照，逐步归因触发匹配、意图打分、边界决断和槽位对齐误差。
- 查询级、触发词条目级和片段级内容指纹缓存；单查询替换或单触发词权重变化只重算真实依赖者。
- 全局匹配点数预算、来源通道配额、单批重算上限三重约束；待办持久化且下一批优先处理。
- 四阶段产物写入同一临时代，校验运行中输入文件和当前代后整体提交；强制终止后只恢复 `current.json` 指向的代。
- 报告包含每修订失效计数、边界意图数、复用命中数、业务码计数、待办、恢复诊断和逐查询账本。

## 快速验收结果

本机实际执行结果：

```text
python3 -m unittest discover -s tests -v
Ran 24 tests in 11.036s
OK
```

使用固定种子 `20261127` 的 24 条查询、8 个会话和修订树 `r00..r07`：

```text
首次运行: processed=24/24 pending=0 segments=14 boundary=1 reuse=0 codes={}
第二次运行: processed=24/24 pending=0 segments=14 boundary=1 reuse=62 codes={}
复用拆分: match=24 score=24 merge=14
```

样本同时覆盖：

- `q018` 得分与阈值精确相等，标记为边界意图。
- `q016..q018` 合并片段中 `wide_fragment` 与 `poi` 的全局偏移冲突被裁决并记录。
- 替换 `q010` 后仅 `q010` 的匹配和打分失效，其他 23 条查询结论不变。
- 调整 `t_nearby_coffee` 权重后 24 条匹配全部复用，仅引用该触发词的 `q004/q016/q018` 重算打分。
- 仅把定点从 `12/16` 调整为 `16/32` 时，24 条匹配仍全部复用，24 条打分按配置键失效。
- 1 线程和 8 线程的全部结论、得分、缓存键与片段逐字段一致。
- `rules`、`score` 阶段被 SIGKILL 后 `current.json` 均保持不变，重启后复用已提交代。

更详细的设计、公式、数据契约和测试映射见 `docs/TECHNICAL.md`。

## 样本格式

查询片段示例：

```json
{
  "query_id": "q001",
  "terms": ["北京", "天气", "一下", "提醒"],
  "session_id": "s01",
  "turn": 0,
  "source": "source-a",
  "ground_truth_intent": "weather",
  "candidates": ["weather", "nearby", "news"],
  "slots": [
    {"name": "city", "start": 0, "end": 1},
    {"name": "weather", "start": 1, "end": 2}
  ],
  "priority": 0
}
```

修订片段示例：

```json
{
  "revision_id": "r06",
  "parent_id": "r04",
  "seq": 6,
  "label": "news-weight-and-delete-old-music",
  "ops": [
    {"op": "update", "trigger_id": "t_news_stock", "weight": 1200},
    {"op": "delete", "trigger_id": "t_music_play"}
  ]
}
```

编辑命令示例：

```json
{"op": "replace_query", "query": {"query_id": "q010", "terms": ["请", "播放", "周杰伦"], "...": "..."}}
{"op": "add_revision", "revision": {"revision_id": "r07", "parent_id": "r06", "...": "..."}}
{"op": "set_active_revisions", "revision_ids": ["r07"]}
{"op": "set_fixed_point", "fraction_bits": 16, "quantum": 32}
{"op": "set_threshold", "intent_id": "weather", "value": 140}
```

## 关键确定性规则

- 触发匹配字符跨度越长越先；同长按触发词项元组字典序；再按 term 起始位置升序。相同情况下用触发词 ID 和条目序兜底。
- 定点打分为 `weight * hit_count * 2^fraction_bits // term_count`，按意图 ID、再按触发词条目序累加。
- 量化采用 half-up：`floor((raw + floor(quantum/2)) / quantum) * quantum`。
- 临界带为 `abs(score - threshold) <= floor(quantum/2)`；默认 `accept_on_tie`。
- 槽位冲突优先级固定为：较早查询、较长跨度、查询 ID 字典序、槽位名。
- 调度顺序：上一批待办优先，其次按优先级、会话、轮次、查询 ID；并发完成顺序不影响输出顺序。
- 只有 `current.json` 指向且校验通过的完整代可被加载；临时目录和未引用代只出现在恢复诊断中。

## 测试

```bash
python3 -m unittest discover -s tests -v
```

测试不访问网络，全部使用 `tempfile` 和固定样本。覆盖校验、规则合成、匹配、打分、边界、槽位、增量、并发、预算待办、快照强杀恢复、报告完整性和 CLI。
