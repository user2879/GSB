# leveldiag — 连续域约束求解与数值误差传播诊断实验环境

面向关卡设计的确定性生成链：YAML 规格 → 固定种子连续域候选布局 → 双精度量化为
网格图块 → 带误差区间的约束判定 → 可独立复算的误差传播账本。规格与种子一致时，
两次运行逐字节一致。

## 环境准备（Windows）

```bat
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Linux/macOS：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

依赖固定在 `requirements.txt`：`numpy==2.4.6`、`PyYAML==6.0.3`，仅需 Python ≥ 3.11，
无容器、无外部数据附件，样本全部本地合成。

## 一条命令演示

```bash
.venv/bin/python -m leveldiag.demo demo --out demo_out
```

生成四类样本并完成全部自检：

| 样本 | 用途 | 预期账本现象 |
| --- | --- | --- |
| `normal` | 正常规格，权重和精确为 1 | 7 条全部确定 PASS |
| `weight_critical` | 权重偏差恰好等于声明容忍（双精度临界值） | 权重归一化条目 UNCERTAIN，边界报告留痕 |
| `boundary` | 连续密度 0.8125 量化到 0.875，阈值取 0.875 | 1 条 UNCERTAIN，严格模式抛 `E_BOUNDARY_AMBIGUOUS` |
| `overflow` | 资源密度上限 1e-6 | 密度条目确定 FAIL |

同一条命令还会：每份样本生成两次做逐字节比对、运行独立复算器、演示六个业务错误码、
执行阈值变更与量化步长变更的增量闭包，汇总写入 `demo_out/demo_report.json`。

## 单独生成与复算

```bash
.venv/bin/python -m leveldiag.demo generate examples/sample_spec.yaml --out out/lv
.venv/bin/python -m leveldiag.demo recompute out/lv
# 严格边界模式（存在 UNCERTAIN 时退出码 2）
.venv/bin/python -m leveldiag.demo generate examples/boundary_spec.yaml \
    --out out/boundary --strict-boundary
```

## 测试

```bash
.venv/bin/python -m unittest discover -s tests -v
```

25 个用例覆盖规格校验与字段路径、连续域/工件确定性、量化分片与步长隔离、
UNCERTAIN 边界口径、复算 STALE/MISMATCH、增量闭包与全部错误码。

## 目录结构

```
leveldiag/
  errors.py        六个业务错误码
  util.py          规范化 JSON/sha256/统一比较口径 classify
  spec.py          YAML 解析、双精度校验、spec_hash
  seeding.py       PCG64 子种子派生树
  continuous.py    连续域高程/密度场、地形抽样、资源点
  quantize.py      量化规则、逐图块 delta、误差上界、区域分片
  grid.py          离散网格与 grid_hash
  constraints.py   七类约束判定与误差区间
  ledger.py        误差传播账本与边界报告
  generator.py     流水线与工件落盘
  recompute.py     独立复算器（只消费工件）
  incremental.py   叶子 diff、声明闭包、增量重生成
  samples.py       四类确定性样本
  demo.py          CLI（demo/generate/recompute）
tests/             unittest 测试
examples/          示例 YAML 规格
docs/DESIGN.md     数值口径、误差传播、闭包与错误码技术说明
```

工件输出（每个样本目录）：`spec.canonical.json`、`continuous/`（逐区域分片 +
seed_tree + 归一化留痕）、`mapping/`（逐区域量化分片 + 规则与误差上界）、
`grid.json`、`ledger.json`、`boundary_report.json`、`manifest.json`（全文件哈希）。

## 实际验证结果

在 Python 3.11 + NumPy 2.4.6 + PyYAML 6.0.3 下：

* `unittest`：25/25 通过；
* 四类样本两次运行全部 `identical=True`（逐字节一致）；
* 复算器对全部样本 `mismatch_count=0`；
* 六个错误码 `E_SPEC_INVALID / E_QUANT_BOUNDS / E_LEDGER_STALE /
  E_LEDGER_MISMATCH / E_BOUNDARY_AMBIGUOUS / E_CLOSURE_MISMATCH` 均按预期触发；
* 阈值变更仅重写账本三件套 + 规格文件，网格/连续域/映射分片哈希不变；
  步长变更重写 mapping/grid/账本，连续域分片哈希不变。
