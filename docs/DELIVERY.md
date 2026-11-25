# NER 复核首版交付说明

本目录是在空项目上首次落地的本地、确定性中文命名实体识别复核系统。系统不依赖外部网络服务；样例、参数包、命令流和注入清单均由固定整数种子在本地生成。

## 1. 环境与实际准备结果

- 解释器：`Python 3.11.0rc1`，命令 `python3 --version` 可复现。
- 项目虚拟环境：`.venv/`。
- 本机 `python3 -m venv` 因缺少 `ensurepip/python3.11-venv` 不可用；已使用用户态 `virtualenv` 创建等价 `.venv`：
  - `virtualenv --clear .venv`
  - `. .venv/bin/activate`
  - `pip install numpy==1.26.4 pytest`
- 已验证：`numpy 1.26.4`、`pytest 9.2.0`。
- Tkinter Python 包存在，但当前 WSL 镜像缺少 `libtk8.6.so`，且 `sudo -n` 需要密码，因此未能在本会话启动 GUI。CLI、测试和验收不依赖 Tk。修复系统依赖：

```bash
sudo apt-get update
sudo apt-get install -y python3-tk
```

## 2. 快速运行

所有命令均在项目根目录执行：

```bash
. .venv/bin/activate

# 生成确定性语料、词表、参数包、命令流和注入清单
python -m ner_review.cli seed-demo --out examples/generated --samples 24

# 加载校验
python -m ner_review.cli validate-corpus --file examples/generated/corpus.jsonl
python -m ner_review.cli validate-package --file examples/generated/model_params.json

# 增量执行并逐条命令提交快照
python -m ner_review.cli run \
  --data examples/generated \
  --workspace state/demo \
  --report reports/demo_run.json

# 六类隔离异常注入
python -m ner_review.cli inject \
  --data examples/generated \
  --out reports/injections

# 交互预算内陈旧摘要 -> 后台完成后权威替换
python -m ner_review.cli budget-demo \
  --budget-ms 5 \
  --work-ms 120

# 一键验收：增量/full parity、快照、预算、SIGKILL、注入
python scripts/acceptance.py

# 完整测试
pytest -q
```

可选 Tkinter 界面：

```bash
python -m ner_review.cli gui --data examples/generated --workspace state/gui
```

## 3. 输入格式

语料为 JSONL，每行一个对象。解析器同时接受空格序列或数组：

```json
{
  "id": "S0001",
  "char_sequence": "北 京 大 学",
  "label_sequence": "B-LOC E-LOC B-ORG E-ORG"
}
```

标签集合固定为 `O` 与四类实体 `LOC/ORG/PER/TIME` 的 `B/I/E/S` 形式，标签编码按 `ner_review/labels.py` 中的排序稳定分配。

参数包包含：

- `model_version`
- `feature_dim`
- `bucket_count`
- `label_encoding`
- `hash_table[bucket][feature]`
- `emission_weights[label][feature][bucket]`
- `transition_weights.start/end/matrix`

## 4. 固定业务码

| 业务码 | 含义 |
|---|---|
| `NER-E1001` | JSONL/JSON 行无法解析 |
| `NER-E1002` | 样本标识重复 |
| `NER-E1003` | 字序列与标签序列长度不一致 |
| `NER-E1004` | 未知标签 |
| `NER-E1005` | BIESO 片段起止、连续性或类型非法 |
| `NER-E1007` | 词表条目非法 |
| `NER-E2001` | 参数包缺失、截断或 JSON 非法 |
| `NER-E2002` | `feature_dim` 非正整数 |
| `NER-E2003` | `bucket_count` 非正整数 |
| `NER-E2004` | 特征维度/桶/权重形状越界 |
| `NER-E2005` | 权重包含 NaN/Infinity |
| `NER-E3001` | 未知或畸形命令 |
| `NER-E3002` | 预算内未完成最新代次，返回陈旧摘要 |
| `NER-E3003` | 尚无任何已完成摘要 |
| `NER-E4001` | 快照版本/model/config 哈希不兼容 |
| `NER-E4002` | 快照哈希或阶段文件损坏 |
| `NER-E4003` | 无可用快照，执行重建 |
| `NER-W4101..W4105` | BIESO 修复规则与依据 |

故障进入对应 `failures` 清单，解析器继续处理后续样本；注入套件会为每个故障创建独立临时目录。

## 5. 确定性处理链

1. `ner_review/validation.py`：语料、词表、参数包、命令流和注入清单校验。
2. `ner_review/features.py`：
   - 当前字、前后字、相对位置分档、词表命中长度五类特征；
   - BLAKE2b 稳定哈希到桶；
   - 按 `(feature_index, bucket, family_index)` 排序；
   - 哈希值和权重分别定点量化，整数乘积再按固定舍入规则缩回；
   - 逐字记录有效小数位、量化步长、量化整数、舍入方向码和累加顺序。
3. `ner_review/decoder.py`：
   - 权重直接作为对数域 potential；
   - 位置升序、目标标签编码升序、前驱标签编码升序显式循环；
   - 不使用 NumPy/Python 数组默认归约；
   - 并列时选择最小编码，并输出 `tie_positions` 与前驱并列记录。
4. `ner_review/entities.py`：
   - `I` 前无 `B`、孤立 `B`、`E` 无 `B`、类型跳跃等固定修复；
   - 每个修复包含位置、前后标签、规则码、依据和受影响位置；
   - 汇总标签、修复标签和词表精确匹配候选；
   - 多类型按“类型编码升序、置信度降序”决断，其余类型逐条进入抑制清单。

## 6. 内容指纹与精确失效

四个阶段为 `features -> emission -> viterbi -> merge`。缓存键由规范 JSON 的 SHA-256 派生：

- features：字序列指纹、被引用词表条目的内容指纹、结构参数指纹；
- emission：features payload 指纹、哈希表/发射权重指纹、量化配置指纹；
- viterbi：emission payload 指纹、转移权重指纹、模型版本；
- merge：viterbi payload 指纹、词表依赖指纹。

因此：

- 替换样本只失效该样本的四个阶段；
- 增删改词表时，通过旧引用索引以及新旧文本出现位置求影响样本，未出现条目的样本直接命中；
- 改发射权重或量化位宽只从 emission 起重算，features 复用；
- 改转移权重只从 viterbi 起重算，features/emission 复用；
- 删除样本会清除其活动索引和缓存。

历史 `reused/recomputed/invalidated` 事件与具体命令历史相关：空白全量必然全部 miss，增量运行必然可能有 hit，二者的历史事件列表不可能逐字相同，否则精确复用就没有发生。系统另外输出仅由最终内容派生的 `execution_manifest.blank_recompute_plan`、`cache_key_map` 和 `live_cache_keys`；这些字段以及逐样本标签、总分、实体、抑制清单在增量和空白全量之间逐字段一致。

## 7. 交互预算与并发

`ner_review/coordinator.py` 使用单写者后台线程、队列和不可变已发布摘要：

- 命令进入后等待配置的 `interaction_budget_ms`；
- 预算内完成则返回 `fresh` 权威摘要；
- 预算超时立即返回最近摘要，并带 `NER-E3002`、`stale_generation`、`target_generation`；
- 后台完成后用同一引擎/账本发布 `fresh` 摘要和快照，陈旧结论不会被冒充为最新；
- 测试用门闩事件稳定复现 stale -> fresh，不依赖机器速度。

## 8. 快照与强杀恢复

`ner_review/snapshot.py` 将以下内容写入独立暂存目录：

- `01_validation.json`
- `02_features.json`
- `03_emission.json`
- `04_viterbi.json`
- `05_merge.json`
- `state.json`：语料、词表、包、命令游标、代次、待办队列、计数和最近账本
- `manifest.json`：快照格式版本、单调快照版本、`model_version`、`config_hash`、各阶段文件 SHA-256

提交流程是：写暂存目录 → fsync → `os.replace` 为正式快照目录 → 原子替换 `current.json` 指针。运行时不做逐文件覆盖。启动时：

- 哈希和版本匹配：`resume`；
- 格式/model/config 版本不兼容：`NER-E4001` 并重建；
- 指针或阶段文件哈希损坏：`NER-E4002` 并重建；
- 无快照：`NER-E4003` 后重建。

`tests/test_process_recovery.py` 会启动真实子进程，在 `SIGKILL`（退出码 `-9`）前只在内存应用命令、不提交快照；随后新进程从旧快照恢复并重放命令流，最终结论与空白全量一致。

## 9. 实际验证结果

最近一次完整测试：

```text
17 passed in 11.00s
```

结果文件：

- `reports/pytest-output.txt`
- `reports/acceptance_report.json`
- `reports/acceptance_report.md`
- `reports/injections/injection_report.json`
- `reports/budget_demo.json`

一键验收覆盖：

- 10 个本地样本、40 个活动阶段缓存键；
- 增量命令序列 vs 相同最终状态空白全量：`parity=true`，双方权威指纹相同；
- 快照往返 parity：`true`；
- 错误 model 版本探测码：`NER-E4001`；
- 预算 stale 码：`NER-E3002`，随后 fresh；
- 真实 SIGKILL 返回码 `-9`，恢复进程返回 0，恢复后 parity：`true`；
- 6/6 异常注入命中预期业务码：`NER-E1002/E1003/E1004/E2001/E2004/E2005`。

## 10. 后续可加强项

- 在可安装 `python3-tk/libtk8.6` 的桌面 WSL 中实际打开 GUI；
- 若语料规模增大，可把阶段 envelope 换为 SQLite/二进制内容寻址存储；
- 可进一步细化发射权重依赖到单个 `(label, feature, bucket)` 坐标，目前权重变更按其影响的全部样本 emission 阶段精确失效。
