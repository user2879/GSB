# 设备日志分桶蓄水池诊断组件

本项目在纯本地、无容器环境中实现设备日志的确定性分桶抽样、两级容量裁决、三版本证据交换、内容指纹增量缓存、阶段化快照恢复、边界诊断与异常注入隔离。实现仅依赖 Python 标准库；测试环境另外安装 `numpy` 与 `pytest`。

## 环境与快速开始

已在 Linux（WSL Ubuntu）上按以下方式验证：

```bash
cd /home/zsl/feedback/uidfc5909d3-default-332

# 当前机器缺少 ensurepip 时使用 virtualenv；若系统支持可直接 python3 -m venv .venv
python3 -m virtualenv -p /usr/bin/python3.11 .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install numpy pytest
pip install -e .

python tools/generate_sample.py
pytest -q
python -m log_sampler verify \
  --config samples/config.json \
  --batches samples/batches.json \
  --injections samples/injections.json \
  --commands samples/commands.json \
  --state artifacts/verification --fresh
```

`verify` 会生成增量运行、空白全量重放、强杀恢复、预算/容量探针和三版本证据互转产物，并把结论写入 `artifacts/verification/verification.json`。实际结果见 [REPORT.md](REPORT.md)。

## 目录

```text
log_sampler/
  validation.py     设备/批次/配置校验与 E_SAMPLE_FORMAT 隔离
  reservoir.py      桶指纹、256 位随机源、确定性蓄水池历史
  truncation.py     桶内容量 + 全局预算两级裁决与截断审计
  evidence.py       p1/p2/p3 编解码、校验与旁路元数据升级还原
  cache.py          桶/最终样本/序列化产物缓存键与失效判定
  snapshot.py       staging -> version 的原子提交与清单
  pipeline.py       并发调度、阶段执行、账本与恢复续跑
  diagnostics.py    随机判据、截断权重边界与翻转来源分类
  state.py          编辑命令、会话持久化与命令幂等
  demo.py           一键验收与强杀子进程测试
  cli.py            run/replay/convert/compare/validate-evidence/verify
samples/            固定种子生成的配置、批次、故障注入和命令序列
tests/              pytest 单元与端到端验收
tools/              确定性样本生成器
```

## 输入

`samples/config.json` 的关键结构：

```json
{
  "seed": 20261126,
  "active_profile": "balanced",
  "evidence_format": "p3",
  "max_workers": 4,
  "devices": [{"device_id": "edge-a", "severities": ["info", "warning", "critical"]}],
  "profiles": {
    "balanced": {
      "global_budget": 8,
      "bucket_capacity": 3,
      "quantization": {"decimals": 3, "tier": "fine"}
    }
  },
  "allowed_quantization": {"decimals": [2, 3, 4], "tiers": ["coarse", "fine"]},
  "bucket_overrides": {"edge-a|critical": {"capacity": 2, "priority": 30}}
}
```

批次文件是 `{"batches": [{"arrival_seq": 1, "records": [...]}]}`。每条记录包含整数 `timestamp`、`device_id`、`severity`、`template_id`、有限非负 `metric` 和全局唯一 `log_id`。批次按文件中的流顺序校验，`arrival_seq` 必须严格递增；坏记录只进入失败清单，不中断其余记录。

故障注入支持：

```text
non_finite_metric        非有限计量值
unknown_bucket           未声明的设备/严重级别组合
batch_order_regression   批次到达序号回退
missing_random_digest    记录携带的随机摘要缺失/非法
```

每项注入必须有独立编号；该编号会进入 `failure.injection_id`、`validation.injections_applied` 和验收报告。

## 确定性抽样语义

1. 桶标识为 `device_id|severity`。记录进入所属桶，并按 `(arrival_seq, record_index, log_id)` 升序形成 1 基到达序位 `i`。
2. 每桶容量为 `k`。前 `k` 条记录按序填入槽位 `1..k`；其后若入样，则替换槽位 `((i - 1) mod k) + 1`，并记录被替换日志标识。
3. 随机源由桶指纹、固定整数种子、日志内容指纹和到达序位派生；裁决使用 256 位整数精确比较 `random * i < k * 2^256`，不依赖平台浮点。
4. 桶指纹包含桶容量、优先级、采样档位、全局预算、量化配置与种子。因此切换桶容量或全局预算会改变随机源、桶历史、槽位与被替换引用。
5. 桶内先保证容量上限，再对全部当前槽位候选做全局截断：优先级数值高者优先；同优先级按入池到达序位升序；序位仍相同按 `log_id` 字典序。排序完全确定。
6. `truncation.audit` 逐条记录给出 `truncation_layer = bucket|global|null` 和裁决依据；`final_samples` 按最终全局名次输出。

## 证据格式

| 版本 | 载体 | 必含字段 | 旁路补充 |
|---|---|---|---|
| p1 | 78 列定宽 ASCII 行 + CRC32 | 桶标识、槽位、24 位整数权重、0/1 入样标记 | 序位、随机摘要、被替换引用、计数快照、日志标识等 |
| p2 | 排序键紧凑 JSON | p1 字段 + 到达序位 + 随机源摘要 | 被替换引用、计数快照、日志标识、记录指纹等 |
| p3 | 带版本头的二进制 + SHA-256 | p2 字段 + 被替换引用 + 桶内计数快照 + 日志标识等 | 无 |

降级时，目标版本无法表达的字段按行索引写入 `*.metadata.json`。升级时解析器先读取旁路并合并，不重新计算随机裁决。可直接使用：

```bash
python -m log_sampler convert --from p3 --to p1 \
  --input artifacts/verification/incremental/versions/v00000009/evidence.p3.bin \
  --output /tmp/evidence.p1.txt \
  --metadata-out /tmp/evidence.p1.metadata.json

python -m log_sampler convert --from p1 --to p3 \
  --input /tmp/evidence.p1.txt --metadata /tmp/evidence.p1.metadata.json \
  --output /tmp/evidence-roundtrip.p3.bin

python -m log_sampler validate-evidence --version p3 \
  --input /tmp/evidence-roundtrip.p3.bin
```

## 增量缓存与账本

缓存键分为三层：

- 桶历史：桶标识、记录内容指纹集合、桶参数指纹、随机源摘要、采样档位与预算档位；
- 最终样本：全部桶历史键、全局预算、优先级和量化配置；
- 序列化产物：最终样本键、证据版本和 schema。

替换单条记录或调整单桶容量时，只有该桶历史键变化，其他桶读取持久化产物；最终样本和序列化产物因下游键变化而精确失效。`ledger.json` 输出：

- 每桶 `cache_hit/cache_miss`、原因、内容与参数指纹；
- 每条记录 `reuse_hit/invalidated/removed` 原因；
- 最终样本来源桶、槽位和全局名次；
- 最终样本与 p1/p2/p3 序列化产物的独立缓存状态；
- 由桶标识与最早到达序位派生的稳定任务排序键。

## 阶段快照与恢复

流水线分为 `validation -> buckets -> truncation -> serialization` 四阶段。产物与待办桶队列、已完成批次游标、已消耗预算先写入 `staging/current`；完成后整个目录重命名为 `versions/vNNNNNNNN`，再原子替换根 `manifest.json`。清单包含全局版本号、配置哈希、输入哈希、随机源摘要、四阶段产物版本和提交状态。

`--crash-after` 可注入真实进程终止：

```bash
python -m log_sampler run --state /tmp/crash-state --crash-after bucket:2
# 退出码 137；再次运行同一命令即依据 staging 清单续跑
python -m log_sampler run --state /tmp/crash-state
```

重启时若配置哈希与输入哈希匹配，则续跑已完成桶；否则放弃旧 staging 并重建。恢复后的逻辑结果与未终止运行逐字段一致。

## 边界诊断

`diagnostics` 输出：

- 入样裁决落在随机判据半步内：高精度参考裁决、定点裁决、差异方向；
- 全局截断相邻候选权重差落在量化档位半步内：原始权重、fine/coarse 权重、假设裁决与真实名次裁决；
- 两档位结论翻转时分源输出业务码：
  - `E_D_FLIP_ORDINAL`：序位/阈值计算导致翻转；
  - `E_D_FLIP_RANDOM_SOURCE`：随机源定点化导致翻转；
  - `E_D_FLIP_MIXED`：两类因素同时参与；
  - `E_D_FLIP_TRUNCATION_TIE`：截断相邻权重在量化档位间翻转。

本地样本最终口径同时产出 `E_D_FLIP_RANDOM_SOURCE` 与 `E_D_FLIP_TRUNCATION_TIE`。

## 常用命令

```bash
# 基线增量会话
python -m log_sampler run --state artifacts/demo \
  --config samples/config.json --batches samples/batches.json \
  --injections samples/injections.json

# 应用命令序列（命令带内容指纹，重复执行幂等）
python -m log_sampler run --state artifacts/demo --commands samples/commands.json

# 从有效会话做空白缓存全量重放并逐字段比较
python -m log_sampler replay --source-state artifacts/demo \
  --state artifacts/replay --fresh

# 任意两个逻辑结果比较
python -m log_sampler compare \
  --left artifacts/demo/latest_result.json \
  --right artifacts/replay/latest_result.json
```

支持命令：`append_batch`、`replace_batch`、`delete_log`、`set_bucket_capacity`、`set_bucket_priority`、`set_profile`、`set_global_budget`、`set_quantization`、`set_evidence_format`。

## 测试

```bash
pytest -q
```

测试覆盖：输入越界与失败隔离、批次回退、确定性蓄水池、容量/预算真实变化、两级截断与并列决断、p1/p2/p3 完整互转与损坏检测、缓存单桶失效、线程上限、命令幂等、诊断翻转分类、强杀续跑与空白重放逐字段一致。

