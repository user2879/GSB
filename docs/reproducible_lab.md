# 可复现实验步骤（Windows / Linux 本地）

## 1. 环境准备

```bash
python --version            # 需要 3.10+
pip install numpy
python -c "import sqlite3, hashlib, threading, numpy; print('deps ok')"
```

## 2. 生成确定性倾斜样本

```bash
python -m hashlot.samplegen --out samples/skew_sample.json
```

## 3. 固定哈希种子运行

```bash
set PYTHONHASHSEED=42            # Windows CMD
# export PYTHONHASHSEED=42       # Linux/macOS
python -m hashlot.cli --sample samples/skew_sample.json --out-dir out/lab --virtual-clock run
```

## 4. 关键输出文件

- `out/lab/mapping_summary.json`：分组键→分区映射摘要与分区行数分布。
- `out/lab/watermark.json`：每连接承载键数、峰值水位、等待事件、借用连接。
- `out/lab/report.json`：定点聚合结果、误差上界、再分发分配、结果摘要。
- `out/lab/ledger.sqlite`：批次账本与会话快照。
- `out/lab/events.jsonl`：结构化日志。

## 5. 恢复与重放

```bash
# 批次 5 提交后强制终止（退出码 7）
python -m hashlot.cli --sample samples/skew_sample.json --out-dir out/lab --virtual-clock --crash-after-batch 5 run
# 依据完整批次重放校验，再从下一游标续跑
python -m hashlot.cli --sample samples/skew_sample.json --out-dir out/lab --virtual-clock recover
python -m hashlot.cli --sample samples/skew_sample.json --out-dir out/lab --virtual-clock replay
```

## 6. 异常注入

```bash
python -m hashlot.cli ... --inject-borrow-cycle run       # E_BORROW
python -m hashlot.cli ... --inject-early-deadline run     # E_DEADLINE
python -m hashlot.cli ... --inject-digest-mismatch run    # 先污染账本
python -m hashlot.cli ... --inject-digest-mismatch replay # E_DIGEST
```

## 7. 全量测试

```bash
PYTHONHASHSEED=42 python -m unittest discover -s tests
```
