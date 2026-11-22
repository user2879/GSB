# Windows 本地运行手册

## 1. 环境准备

```powershell
py --version          # 需 3.10 及以上；同时确认 python --version 可用
py -m pip install "numpy>=1.26,<1.27"
```

不引入任何其他第三方库；SQLite 随 Python 标准库提供。节点子进程通过
`sys.executable -u -m configreg.node` 拉起，Windows 下带
`CREATE_NEW_PROCESS_GROUP` 标志，回环地址固定 `127.0.0.1`、端口来自注册表。

## 2. 生成输入

```powershell
python scripts\generate_inputs.py
python scripts\generate_dispersion_inputs.py
```

产出 `data\registry.json`、`data\baseline.json`、`data\candidate.json`
以及离散场景 `data\dispersion\*`。

## 3. 分步命令

```powershell
# 需求 1：强校验
python -m configreg.cli validate

# 需求 2、3：差异分类、影响链、定点误差界
python -m configreg.cli diff --precision 4

# 需求 4-7：完整采集、预算判定、重试、账本与快照
python -m configreg.cli run --quiet --ledger artifacts\ledger.sqlite3 `
  --batch-size 2 --socket-timeout-ms 800 --max-attempts 5 `
  --max-elapsed-delta-ms 50 --max-mismatch-nodes 0

# 需求 8：重放校验
python -m configreg.cli replay --ledger artifacts\ledger.sqlite3 `
  --socket-timeout-ms 800 --max-attempts 5

# 需求 8：强制终止演练（提交 1 批后终止，退出码 3，无退出报告）
python -m configreg.cli run --ledger artifacts\crash.sqlite3 --crash-after-batches 1

# 重启恢复：未提交项标记未知并重跑
python -m configreg.cli run --ledger artifacts\crash.sqlite3

# 需求 7：改精度重开 -> 整体替换（观察 replacement_events）
python -m configreg.cli run --precision 6 --ledger artifacts\ledger.sqlite3
```

## 4. 一键验收

```powershell
python scripts\demo.py
```

结果落在 `artifacts\demo\`：`00_summary.json` 为总览，`01..09` 为各场景
原始结论，`ledger.sqlite3` 为账本，`logs\` 为节点子进程日志。

## 5. 运行测试

```powershell
python -m unittest discover -s tests -v
```

## 6. 错误码

| 错误码 | 含义 | 触发位置 |
| --- | --- | --- |
| `E_MANIFEST` | 清单/注册表校验失败，整体拒绝并列出违规字段 | 加载阶段 |
| `E_REGRESSION` | 超预算回归，含差异数值与预算余量 | 判定阶段 |
| `E_REPLAY` | 重放与账本首个不一致，定位 item/node/field | 恢复阶段 |
| `E_TERMINATED` | 强制终止演练，不生成退出报告（CLI 退出码 3） | 崩溃注入 |

## 7. 查看 SQLite 账本

```powershell
python -c "import sqlite3; c=sqlite3.connect('artifacts/ledger.sqlite3'); print(c.execute('SELECT name FROM sqlite_master WHERE type=''table''').fetchall())"
```

表：`meta`、`batches`、`diff_records`、`item_verdicts`、`observations`、
`scheduler_records`、`snapshot`、`replacement_events`。
