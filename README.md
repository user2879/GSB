# 本地确定性中文短文本近重复簇账本

这是一个可集成、可持久化、可增量重算的近重复探测组件。它面向本地 JSONL 中文短文本库，使用确定性归一化、分窗 MinHash、带符号随机投影 LSH、定点相似度与稳定簇账本，完成近似近重复分组。

核心代码在 `near_dup/`，固定种子样例在 `samples/`，可重复端到端演示在 `scripts/run_demo.py` 或 `python -m near_dup.cli demo`，实际验证明细在 `reports/diagnostic_report.json` 与 `reports/verification.md`。

## 快速开始

```bash
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt

python scripts/generate_samples.py --out samples
python -m near_dup.cli demo --samples samples --run-dir runs/demo --report-dir reports
python -m unittest discover -s tests -v
```

若基础镜像缺少 `ensurepip`，可先安装系统 `python3-venv`，或使用 `python -m venv --without-pip .venv` 后通过官方 `get-pip.py` 引导。当前验证环境为 Python `3.11.0rc1`、NumPy `2.4.6`。

NumPy 版本会写入诊断报告；核心哈希、定点相似度与持久化结构使用纯 Python 整数运算，避免平台浮点和线程调度引入差异。

## 目录

```text
near_dup/
  config.py          YAML 配置校验与派生指纹
  text_store.py      JSONL 文本逐条校验
  plan.py            编辑计划、故障注入清单校验与批次应用
  features.py        确定性归一化、置换、等长分窗和补位
  hashing.py         分窗 MinHash、SRP 分桶、定点相似度
  engine.py          增量缓存、预算调度、聚类账本、并发与阶段提交
  snapshot.py        整代目录提交、LATEST 原子替换与恢复校验
  demo.py            完整可重复验证与诊断报告生成
  cli.py             命令行入口
samples/             固定种子输入、编辑计划、注入清单和反例
scripts/             样本生成与演示入口
tests/               20 项标准库 unittest
reports/             已实际运行生成的 JSON/Markdown 验证报告
```

`runs/` 是演示生成的持久化快照目录，可在运行时重建。

## 输入格式

文本库 `samples/texts.jsonl` 每行一条：

```json
{"text_id":"T001","terms":["甲","甲","甲"],"source":"forum","version":1}
```

校验规则：`text_id` 唯一且非空；`terms` 非空，逐项归一化后必须落在配置词表内；`source` 必须属于 `source_labels`；`version` 为非负整数且按文件顺序单调不减。非法行不会被丢弃，而是进入带 `E_TEXT_INVALID` 的失败清单。

配置 `samples/config.yaml` 使用依赖无关的 YAML 子集，支持扁平映射、行内列表和缩进块列表。关键字段：

```yaml
window_count: 4
signature_dimension: 32
similarity_threshold: 0.75
bucket_bits: 10
fixed_decimal_places: 4
quantization_levels: 1000
signature_batch_budget: 2048
max_workers: 4
snapshot_commit_interval: 3
source_labels: [forum, wiki, news]
vocabulary: [甲, 乙, 丙, 丁]
```

校验码为 `E_CONFIG_INVALID`。签名维度必须为正偶数；相似度阈值必须为 `[0,1]` 内的有限值；分桶位数不得超过签名维度；定点位宽、量化档位、预算、并发度和提交间隔均须为合法正整数。

编辑计划 `samples/edit_plan.json` 按 `batches` 顺序执行，支持：

- `append`：追加文本。
- `replace`：以相同 `text_id` 替换文本。
- `delete`：删除文本。
- `set_config`：修改可编辑配置字段。

结构错误、未知操作或缺失目标输出 `E_PLAN_INVALID`；操作内文本内容错误输出 `E_TEXT_INVALID`。编辑计划在 `samples/invalid_plan.json` 中另有反例。

故障注入清单 `samples/injections.json` 独立于正常输入，覆盖四类故障，注入编号会写入诊断报告：

- `zero_signature_dimension` → `E_CONFIG_INVALID`
- `non_finite_similarity` → `E_CONFIG_INVALID`
- `bucket_bits_out_of_range` → `E_CONFIG_INVALID`
- `cluster_id_conflict` → `E_CLUSTER_ID_CONFLICT`

## 计算语义

1. **归一化与分窗**：逐词项执行 NFKC 全半角折叠、`casefold` 大小写折叠和空白合并。以文本 ID 与签名算法指纹派生 SHA-256 计数器随机源，在局部执行 Fisher-Yates 置换，不读取进程全局随机状态。序列按配置窗数切成等长窗口；最后不足部分使用包含窗口与位置的固定占位符补齐，并持久化补位位置。
2. **分窗 MinHash**：总签名维度 `D` 按固定顺序分摊给 `W` 个窗口，前 `D % W` 个窗口多一个哈希通道。每个通道在对应窗口内取固定哈希族的最小值，最终签名长度严格等于 `D`。若配置出现 `W > D`，后续窗口没有哈希通道，但仍保留确定性窗口产物。
3. **定点相似度**：签名一致分量数除以 `D`，先转换到 `10**fixed_decimal_places` 定点整数，再按 `quantization_levels` 量化为离散档位。阈值使用同一量化规则。相似度恰等于阈值时视为命中并记入边界清单；多个候选相同分值时按字典序最小的簇标识/代表文本决断。
4. **LSH 与簇账本**：每个分桶位使用由配置派生的 ±1 随机投影向量，投影符号组成桶键。候选文本只与同桶当前代表文本比较。簇标识始终取成员中字典序最小的文本 ID；合并记录两个旧簇、成员、相似度、边界标记、任务排序键与合并原因。
5. **删除语义**：删除文本会从当前文本缓存、签名、桶、待办队列和所有簇中清除；两成员簇删除一条后自动输出单元素簇标记。

## 增量缓存与预算

- 归一化键由文本内容指纹、签名算法指纹和窗数派生。
- 签名键由归一化结果指纹、签名算法指纹、签名维度、定点位宽与量化档位派生。
- 分桶键由签名键、分桶投影指纹和桶位数派生。因此只改分桶位数时签名可直接复用。
- 簇指纹由成员文本缓存键的有序列表派生，桶组指纹还包含聚类参数与成员键。
- 只改相似度阈值时，签名和分桶全部复用，仅簇账本重算。
- 替换或删除单条文本时，仅该文本签名失效；相关桶和簇重算，其他桶组直接复用持久化产物。

签名预算按 `文本数 × signature_dimension` 计。失效文本先按上一批待办顺序，再按新失效文本的文本 ID 字典序排队；预算不足的剩余文本保持稳定待办顺序，并输出 `E_SIGNATURE_BUDGET_PENDING`。预算账本记录文本 ID、签名缓存键、任务排序键、分量成本、状态、本批已消耗与累计消耗。

## 并发与快照

签名计算和桶内相似度比较使用 `ThreadPoolExecutor`，线程数严格来自 `max_workers`。任务携带由文本 ID 与阶段名派生的稳定排序键；并发与串行最终状态逐字段一致。

快照按代组织：

```text
snapshot/
  generations/g00000005/{manifest.json,normalization.json,signatures.json,buckets.json,ledger.json}
  LATEST
  tmp/
```

所有阶段先写入临时目录，再整体重命名为不可变代目录；最后通过同文件系统的 `os.replace` 原子更新 `LATEST`。清单记录全局版本号、配置哈希、完成文本、待办、预算、各阶段版本和阶段哈希。恢复时可用：

```python
engine.recover_manifest(
    expected_version=5,
    expected_config_hash="...",
)
```

版本或配置哈希不一致会抛出 `E_SNAPSHOT_VERSION_MISMATCH`。演示会真实制造一次签名阶段中断，保留部分已提交快照，再从该快照恢复并与全量重建比较。

## 集成 API

```python
from near_dup import NearDuplicateEngine, load_config, load_text_jsonl

config, failures = load_config("samples/config.yaml")
records, text_failures = load_text_jsonl("samples/texts.jsonl", config)

engine = NearDuplicateEngine("runs/my-ledger")
result = engine.sync(records, config, external_failures=text_failures)

assert result["clusters"] == sorted(result["clusters"], key=lambda item: item["cluster_id"])
projection = NearDuplicateEngine.state_projection(result)
```

`sync` 可反复调用；传入最新完整文本集合和最新配置即可。组件通过内容与阶段缓存键自动识别追加、替换、删除和配置变化。

## 已验证结果

最近一次实际运行：

- `python -m unittest discover -s tests -v`：20 项测试全部通过。
- 增量编辑序列与从空白全量重建：核心状态投影逐字段一致，SHA-256 为 `2a2687ef3a9b9e196480fc844778231113d835a5698be9777d374cab2b4b4c89`（报告生成时值）。
- 并发与串行：逐字段一致。
- 签名阶段中断后的恢复结果：与全量重建逐字段一致。
- 四类故障注入：全部命中预期业务码。
- 预算演示：首批只完成 `T001`，第二批优先完成 `T002`，待办顺序保持稳定。

核心一致性投影覆盖逐文本签名、分桶归属、簇账本成员顺序、边界文本、合并日志、失败清单顺序与待办。`invalidated/reused` 执行审计记录的是“相对上一次持久化状态实际做了什么”，因此增量运行与从空白运行天然不同；其正确性通过缓存键、阶段版本、真实重算计数和最终状态一致性分别验证。

机器可读诊断见 `reports/diagnostic_report.json`，人读摘要见 `reports/verification.md`。
