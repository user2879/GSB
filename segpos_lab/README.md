# segpos_lab：中文分词与词性标注增量交互组件

纯本地、无容器、无外部服务依赖。Python 3.11 + NumPy 1.26.4。

## 快速开始

```bash
cd segpos_lab
python3.11 -m virtualenv .venv          # 系统 python 缺 ensurepip 时用 virtualenv
. .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 1. 固定种子生成样本（含非法注入场景 samples/invalid）
python scripts/generate_samples.py --out samples/base --seed 20261123

# 2. 三种运行模式
python -m segpos.runner full        --sample-dir samples/base --snapshot-dir run/snap-full --report run/report-full.json
python -m segpos.runner interactive --sample-dir samples/base --snapshot-dir run/snap-inc  --report run/report-inc.json
python -m segpos.runner restore     --sample-dir samples/base --snapshot-dir run/snap-inc  --report run/report-restore.json

# 3. 验收脚本
bash scripts/acceptance.sh
```

## 输入文件

| 文件 | 格式 |
| --- | --- |
| `charset.txt` | 受控汉字与标点（无空白字符参与校验） |
| `sentences.txt` | 每行 `句子ID<TAB>原始文本` |
| `lexicon.txt` | 每行 `词 词频 词性`，重复词形按词频累加 |
| `prob_table.json` | `tags / start_logp / transition_logp / emission_logp`，对数概率；`smoothing` 为正概率 ε |
| `config.json` | 并发、首响预算、快照间隔、阈值、定点位宽、量化档位、平滑、OOV 规则 |
| `commands.json` | `add_entry / remove_entry / adjust_freq / replace_prob_table / replace_sentence` |

概率表替换支持 `{"table": "@prob_table_v2.json"}` 的本地文件引用。

## 关键语义

- 非法条目不中断运行，按 `E_SENTENCE_INVALID / E_LEXICON_INVALID /
  E_PROB_TABLE_INVALID / E_CONFIG_INVALID / E_COMMAND_INVALID` 进入失败清单。
- 首响预算内无法完成时返回 `E_INTERACT_BUDGET_PENDING`，待办句子保留且不污染
  最终结果；排空后结果与无预算全量运行逐字段一致。
- 快照采用版本目录 + `current` 原子符号链接；强制终止后只从完整快照恢复。

## 测试

```bash
python -m unittest discover -s tests -v
```

28 个用例覆盖加载校验、FMM 决断、OOV 规则边界差异、维特比并列与确定性、
增量/全量等价（含不同命令切分块）、精确失效、预算挂起、并发一致性、
快照恢复与强杀恢复。
