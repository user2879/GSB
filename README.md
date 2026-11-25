# fxpipe — 定点整数嵌入/聚合算子链

本地、确定性、无 GPU/无容器的 LLM 推理前嵌入与聚合算子链引擎。详见
[`DESIGN.md`](DESIGN.md)。

## 环境准备

```bash
python3 -m venv .venv            # 若系统缺 ensurepip，可用 virtualenv -p python3.11 .venv
.venv/bin/pip install "numpy>=2.0"
```

## 生成样本并运行

```bash
.venv/bin/python main.py generate workspace/samples
S=workspace/samples

# 校验加载层（e2）
.venv/bin/python main.py validate \
  --config $S/config.json --tokens $S/tokens.json --segments $S/segments.json \
  --graph $S/graph.json --e2-file $S/chain.e2.json --bypass $S/bypass.json

# 全量运行（e2），产出快照与诊断报告
.venv/bin/python main.py run \
  --config $S/config.json --tokens $S/tokens.json --segments $S/segments.json \
  --graph $S/graph.json --e2-file $S/chain.e2.json --bypass $S/bypass.json \
  --snapshot-root workspace/snapshots/run --report workspace/run/report.json

# 带编辑命令流的增量运行（复用同一快照根目录）
.venv/bin/python main.py run \
  --config $S/config.json --tokens $S/tokens.json --segments $S/segments.json \
  --graph $S/graph.json --e2-file $S/chain.e2.json --bypass $S/bypass.json \
  --edits $S/edits.json \
  --snapshot-root workspace/snapshots/run --report workspace/run/inc.json

# 一键端到端验收（双版本互操作 + 增量==全量 + 失效/复用账本）
.venv/bin/python main.py acceptance workspace
```

## 测试

```bash
.venv/bin/python -m unittest discover -s tests
```

## 业务码

| 码 | 含义 |
| --- | --- |
| `E_SCHEMA` | 字段/枚举/图结构校验失败，跳过该条目且不中断其余条目 |
| `E_FIELD_TRUNCATED` | e1 定宽字段宽度不足或含非法字符 |
| `E_MISSING_FIELD` | 必需量化字段缺失（以显式缺失标记补齐） |
| `E_DUPLICATE` / `E_CYCLE` / `E_REF` | 链序号重复 / 算子图有环 / 输入引用不存在 |
| `E_OPERATOR_QUOTA_EXCEEDED` | 单算子点数超配额，本批中止该算子 |
| `E_POINT_BUDGET_PENDING` | 批预算不足，任务保留待办队列 |
