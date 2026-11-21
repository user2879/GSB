# LayerPack：增量模型量化与部署包组件

LayerPack 是一个可嵌入的本地领域组件，在逐层权重包、校准集和压缩配置之上执行：

- 对称/非对称定点量化与反量化；
- 分组量化、逐通道缩放、全精度层白名单；
- 校准集误差评估与软标签蒸馏校正；
- 基于内容指纹的增量失效与持久化复用；
- 单批点数预算下的稳定并发调度；
- 四阶段原子快照、崩溃恢复与诊断报告；
- 八类交互编辑命令，以及增量/全量逐字段一致性验收。

项目不需要容器、外部服务或网络。所有样本由固定种子在本地生成。

## 快速开始

环境要求：Python 3.10+，`numpy`、`PyYAML`、`pytest`。

```bash
python3 -m pytest -q
python3 -m layerpack validate \
  --weights samples/weights.jsonl \
  --calibration samples/calibration.json \
  --config samples/config.yaml --pretty
```

执行一次完整压缩：

```bash
python3 -m layerpack run \
  --weights samples/weights.jsonl \
  --calibration samples/calibration.json \
  --config samples/config.yaml \
  --state-dir .layerpack/demo --pretty
```

应用八类命令并直接比较“增量最终状态”与“空白目录全量重建”：

```bash
python3 -m layerpack commands \
  --commands examples/commands-all-eight.jsonl \
  --weights samples/weights.jsonl \
  --calibration samples/calibration.json \
  --config samples/config.yaml \
  --state-dir .layerpack/eight-command-demo \
  --reset --latency-budget-ms 0 --full-compare
```

命令最终会输出：

- `interim` 帧：带 `E_INTERACT_BUDGET_PENDING`，`persisted=false`；
- `final` 帧：带完整结果与诊断，`persisted=true`；
- `full_comparison.equal=true` 时，增量结果与空白全量重建逐字段一致。

## 项目结构

```text
layerpack/
  config.py       配置加载、归一化与校验
  loader.py       JSONL/JSON/YAML 加载与业务码
  quantize.py     对称/非对称定点量化和误差统计
  calibration.py  校准误差与软标签校正
  fingerprint.py  规范化 JSON、数组指纹、固定顺序数值运算
  cache.py        量化/反量化/校准/清单内容寻址缓存
  scheduler.py    单批点数预算与线程池稳定调度
  snapshot.py     阶段临时目录、原子替换、恢复点
  deployment.py   部署清单与原子部署包导出
  engine.py       四阶段引擎与诊断
  session.py      八类命令和会话状态
  interaction.py  交互延迟预算与 interim/final 帧
  cli.py          命令行入口
samples/          固定种子生成的确定性样本
examples/         命令序列与注入配置
tests/            本地验收测试
scripts/          样本重生成脚本
docs/             架构、验收和格式说明
```

## 输入格式

### 权重包 JSONL

每行一个层对象。数组按 C 顺序展开，也接受与 `shape` 完全一致的嵌套数组。

```json
{"name":"embed","shape":[6,8],"weights":[0.1,-0.2],"content_version":1}
```

校验规则：

- 层名非空且唯一；
- 形状为非空正整数列表，元素数必须匹配；
- 所有元素有限；
- 文件顺序上的 `content_version` 单调不减。

### 校准集 JSON

```json
{
  "samples": [
    {
      "id": "cal-00",
      "input": [0.1, 0.2],
      "reference_output": [0.3],
      "layers": ["all"]
    }
  ]
}
```

每层的输出定义为：权重最后一维与 `input` 做固定顺序点积；对二维及以上权重，逐行展平输出。`layers` 为空或包含 `"all"` 时适用于所有层。

### 压缩部署配置 YAML

```yaml
schema_version: 1
quantization:
  default_bits: 8
  scheme: symmetric          # symmetric | asymmetric
  group_size: 4
  group_axis: -1
  per_channel_scaling: true
  scale_significant_digits: 7
layers:
  encoder.attn.k:
    bits: 4
    group_size: 4
full_precision_layers: ["encoder.norm"]
deployment:
  package_format_version: "1.0.0"
  artifact_name: "demo-transformer-slice"
runtime:
  interaction_latency_budget_ms: 20
  concurrency: 4
  point_budget_per_batch: 65536
  snapshot_commit_interval: 1
  stage_delay_ms: 0
distillation:
  temperature: 2.0
  soft_label_weight: 0.35
  correction_gain: 1.0
```

允许位宽为 `2, 4, 8, 16`。`group_size` 必须整除 `group_axis` 指向的维度。`stage_delay_ms` 是测试友好字段，用于稳定制造“交互预算内尚未完成”的窗口；生产配置可保持 0。

## 输出与状态目录

一次运行的核心结果包含：

- `layers[]`：逐层量化参数、量化码摘要、反量化摘要、失真指标；
- `calibration[]`：逐样本校正前/后偏差、校正量、软标签关闭标记；
- `error_report`：逐层误差表与总量；
- `deployment_manifest`：部署格式、逐层摘要、总量、全局清单指纹；
- `failure_list`：按阶段、层名、业务码稳定排序的失败清单。

诊断位于 `reports/diagnostics.json`，包含：

- 已完成层、受影响层、失效/复用清单与计数；
- 逐层误差、退化组来源、软标签关闭层；
- 待算队列、预算扣减序列、批次与最大并发；
- 快照版本、阶段版本、恢复点、配置/校准/权重哈希；
- 全部失败业务码和明细。

部署包位于 `deploy/package-<manifest_sha256前20位>/`，包含 `manifest.json`、逐层量化码和参数，`deploy/current.json` 原子指向当前包。

## 业务码

| 业务码 | 含义 |
| --- | --- |
| `E_CONFIG_INVALID` | YAML、枚举、分组、白名单、并发/预算字段非法 |
| `E_WEIGHT_SHAPE_MISMATCH` | JSONL、层名重复、形状/元素数/版本单调性非法 |
| `E_NONFINITE_WEIGHT` | 权重包含 NaN/Inf |
| `E_CALIB_MISSING` | 校准文件、样本、输入维度或引用层缺失 |
| `E_INTERACT_BUDGET_PENDING` | 交互预算内只返回最近完成摘要 |
| `E_POINT_BUDGET_PENDING` | 单批点数预算不足，存在待算队列 |

## 交互命令

命令为 JSON/JSONL，支持：

```json
{"type":"set_bits","layer":"head","bits":8}
{"type":"set_scheme","scheme":"asymmetric"}
{"type":"set_group_size","layer":"encoder.attn.k","group_size":8}
{"type":"set_per_channel","enabled":false}
{"type":"set_temperature","value":1.5}
{"type":"set_soft_label_weight","value":0.2}
{"type":"replace_layer_weights","layer":"head","shape":[4,8],"weights":[...]}
{"type":"replace_calibration_sample","sample":{...}}
```

全局命令省略 `layer` 时作用于全部层。权重替换省略 `content_version` 时分配最大版本号 + 1。会话状态只在最终帧产生后写入 `session/state-<hash>/`，中间帧不会提交命令。

## 确定性约定

- 浮点运算统一使用 `float64`，缩放因子按十进制有效位向零截断；
- 量化取整固定为 nearest-ties-to-even；
- 组内统计、点积、校正和误差累计均按原形 C 顺序逐项执行；
- 所有层、样本、失败、复用和预算事件均按稳定键排序；
- 线程只改变执行先后，不改变结果、顺序或预算扣减序列；
- 任何未失效产物从内容寻址缓存直接加载。

更完整的设计与验收说明见 `docs/architecture.md` 和 `docs/acceptance.md`。
