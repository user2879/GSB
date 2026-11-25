# vscan：本地列式块文件的确定性向量化扫描

这是一个从零实现的可验收版本，覆盖列存编解码、批级计划、NumPy 谓词过滤、
定点聚合、收益归因、受限并发调度、增量失效重算与 SQLite 账本恢复。
实现不使用外部服务、容器或未授权网络目标。

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pytest

# 生成确定性列存、执行、全量重建校验并写报告
python -m vscan verify \
  --spec examples/spec.json \
  --data examples/runtime/data \
  --ledger examples/runtime/ledger.sqlite \
  --output examples/runtime/report.json

# 仅查看计划、裁剪与下推依据
python -m vscan explain --spec examples/spec.json

# 只建列存文件
python -m vscan build --spec examples/spec.json --data /tmp/vscan-data
```

本次 WSL 镜像的系统 Python 缺少 `ensurepip`，实际等价流程是
`python3 -m venv --without-pip .venv` 后用官方 `get-pip.py` 引导，再安装同一组依赖；
普通 Ubuntu 环境可直接使用上面的标准命令。

样本声明在 `examples/spec.json`。生成物已在 `examples/runtime/` 中留档，包括
`data/manifest.json`、每个段每列的 `.col` 文件、SQLite 账本和完整 JSON 报告。

## 代码入口

- `vscan.spec`：JSON 强校验、规范化与 `E_SPEC` 定位。
- `vscan.codec`：原始、字典、游程三种二进制编码及选择性行解码。
- `vscan.store`：固定种子生成、原子写入、列摘要/段摘要/快照摘要。
- `vscan.planner`：批边界、列裁剪、下推谓词和计划摘要。
- `vscan.executor`：NumPy 掩码过滤、分组与定点聚合。
- `vscan.attribution`：整数收益归因与双口径容差决策。
- `vscan.scheduler`：真实线程池、并发上限与解码内存准入。
- `vscan.session`：编辑失效传播、增量重算和全量重建校验。
- `vscan.ledger`：SQLite 原子批提交、整体替换、备份和恢复审计。
- `vscan.cli`：`build`、`plan`、`explain`、`run`、`verify`。

`python -m vscan` 与安装后的 `vscan` 命令等价。

## 会话式增量编辑

```python
from vscan.session import Session
from vscan.spec import load_spec

with Session(load_spec("examples/spec.json"), "examples/runtime/data", "examples/runtime/ledger.sqlite") as s:
    report = s.run()
    report = s.edit_column_value("s_east", "amount", 17, 12345)
    report = s.edit_projection("q_product", ["product", "amount"])
    report = s.edit_predicate("q_region", {"op": "gt", "column": "amount", "value": 500})
    report = s.set_batch_rows(8)
    assert s.verify_against_full_rebuild()["consistent"]
```

单值编辑只失效包含该行的批；未引用该列的查询和未包含该行的批复用已有摘要。
投影、谓词或批行数编辑只失效目标查询的全部批，批行数编辑会重建全部批边界。

## 已执行验证

环境：Linux/WSL Ubuntu，Python `3.11.0rc1`，NumPy `2.4.6`，pytest `9.2.0`。

```text
$ .venv/bin/pytest -q
46 passed
```

样本验收结果：

```text
queries=2, batches=16
max_observed_inflight=3
max_observed_decoded_bytes=1152 <= budget 1200
full_rebuild_verification.consistent=true
segments_checked=2, columns_checked=10
batches_checked=16, queries_checked=2
```

样本归因：

```text
q_product: bytes 5310 -> 1827, saved 3483
           (column_pruning=3186, predicate_pushdown=297, batch_partition=0)
           candidates 118 -> 85, comparisons 118 -> 85, ppm 口径受 2ppm 容差影响
q_region:  bytes 5310 -> 2655, saved 2655
           (column_pruning=2124, predicate_pushdown=531, batch_partition=0)
           candidates 118 -> 59, physical comparisons 236 -> effective 118
```

账本测试还验证了：同快照重开不替换；快照变化时整体替换并生成 `0444` 只读备份；
强制子进程在未提交批后 `os._exit(17)`，同快照重开仍保留完整提交记录，批指纹不匹配
的记录被标为 `unknown`，恢复重算会与旧提交记录逐项比较。

详细设计、精确字节格式、归因公式与恢复规则见
[`docs/design.md`](docs/design.md)。

## 测试范围

- 三种编码的确定性往返、字典升序、游程定义、选择性行解码。
- 加载顺序无关的列/段/快照摘要与非空计数。
- 计划摘要对谓词顺序、投影顺序和批行数不变；批边界无重叠无遗漏。
- 向量化过滤与独立逐行朴素扫描结果一致。
- 空值跳过、空值计数、整数溢出饱和、均值向零截断。
- 两套归因口径的整数差值和稳定选择，全程无浮点。
- 线程池真实并行、并发峰值、内存峰值、等待起止和单批至多一次。
- 单值/投影/谓词/批行数编辑后的精确失效集合与 E_DIVERGE 定位。
- SQLite 原子提交、内容摘要、只读备份、未提交批恢复和逐项重放比较。
- CLI 样本端到端执行与全量重建校验。
