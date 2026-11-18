# 本地存储配额与限流配置编辑器

这是一个只依赖 Python 标准库的本地应用。它读取本地 JSON 配置和 NDJSON 访问事件，完成配置格式升级、派生配额解析、整数定点配额/令牌桶判定，并把配置版本、逐事件判定、误差界、用量和恢复进度写入本地 SQLite。Tkinter 界面只在本机 Windows 上启动，不访问网络、端口或真实存储服务。

## 快速开始

Windows 官方 Python 3.11+ 安装包默认包含 `tkinter` 与 `sqlite3`。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python .\scripts\generate_samples.py --output .\data
python -m quota_app upgrade --config-dir .\data\configs --events .\data\events\access-events.ndjson --ledger .\data\ledger.sqlite3 --log .\data\logs\quota-app.jsonl
python -m quota_app run --config-dir .\data\configs --events .\data\events\access-events.ndjson --ledger .\data\ledger.sqlite3 --log .\data\logs\quota-app.jsonl --batch-size 3
python -m quota_app gui --config-dir .\data\configs --events .\data\events\access-events.ndjson --ledger .\data\ledger.sqlite3 --log .\data\logs\quota-app.jsonl --preview-budget-ms 50
```

Linux/WSL 上可替换为 `python3`。无 Tkinter 的环境仍可运行完整核心、测试和命令行；GUI 命令会返回 `E_GUI`。

测试不依赖 pytest：

```bash
python -m unittest discover -s tests -t . -v
```

## 本地样本

`scripts/generate_samples.py` 只写本地文件，生成：

- `data/configs/config-v1.json`：v1，只含卷标识和容量。
- `data/configs/config-v2.json`：v2，增加 `ops_per_sec`、`burst_capacity`。
- `data/configs/config-v3.json`：v3，增加 `limit_dimension` 和派生关系。
- `data/configs/upgrade_checkpoint.json`：初始空检查点。
- `data/metadata/derivations.json`：派生声明与预期仲裁结果说明。
- `data/events/access-events.ndjson`：10 条确定性访问事件。

第一次打开配置目录时，v1/v2 会在内存中校验、升级并原子重写为 v3；原文件内容保存在只读 `data/configs/backups/*.bak`。

## 配置格式

当前编码版本为 v3。加载器同时接受整数版本字段 `format_version`、`version` 或 `schema_version`。

```json
{
  "format_version": 3,
  "revision": 3,
  "name": "sample-v3-derivation",
  "volumes": [
    {
      "id": "root",
      "capacity_bytes": 1000,
      "ops_per_sec": 2,
      "burst_capacity": 2,
      "limit_dimension": "operations",
      "derivation": null
    },
    {
      "id": "child",
      "capacity_bytes": 700,
      "ops_per_sec": 5,
      "burst_capacity": 400,
      "limit_dimension": "bytes",
      "derivation": {"upstream_id": "root", "factor": 1, "offset": 100}
    }
  ]
}
```

- `revision` 是配置修订号，必须随编辑单调递增；它不同于格式版本。
- `limit_dimension` 取 `bytes` 或 `operations`，也接受 `write_bytes`、`ops` 等别名并在录入时归一。
- v1 归一后 `ops_per_sec=burst_capacity=0`、维度为 `bytes`，表示不启用限流。
- v3 也接受扁平派生字段 `derived_from`、`factor`、`offset`，但应用总是输出嵌套 `derivation`。
- 所有数值必须是非负整数；布尔值不视为整数。

## 事件格式

每行一个 JSON。规范字段如下，也支持常见别名 `sequence`、`ts_ms`、`ops`、`bytes` 等：

```json
{"operation_count":1,"seq":1,"timestamp_ms":0,"volume_id":"root","write_bytes":100}
```

`seq` 必须严格递增，`timestamp_ms` 必须是非负整数且全流不倒退。事件流在写账本前完整校验，因此 `E_ORDER` 不会留下部分批次。

## 命令

所有命令都使用本地路径：

```text
python -m quota_app validate  --config-dir ... --events ...
python -m quota_app upgrade   --config-dir ... --ledger ... --log ...
python -m quota_app run       --config-dir ... --events ... --ledger ... --log ... --batch-size 3
python -m quota_app recover   --ledger ... --log ...
python -m quota_app preview   --config-dir ... --volume child --capacity-bytes 800 --preview-budget-ms 50 ...
python -m quota_app edit      --config-dir ... --volume child --capacity-bytes 800 --ingest --batch-size 3 ...
python -m quota_app view      --ledger ...
python -m quota_app gui       --config-dir ... --events ... --ledger ... --log ...
```

异常注入使用临时副本，不破坏样本：

```text
python -m quota_app inject corrupt-config       --config-dir data/configs --events data/events/access-events.ndjson --log data/logs/injection.jsonl
python -m quota_app inject missing-checkpoint   --config-dir data/configs --events data/events/access-events.ndjson --log data/logs/injection.jsonl
python -m quota_app inject timestamp-regression --config-dir data/configs --events data/events/access-events.ndjson --log data/logs/injection.jsonl
```

等价的 `run --inject corrupt-config|missing-checkpoint|timestamp-regression` 也可用；也提供 `--inject-corrupt-config`、`--inject-missing-checkpoint`、`--inject-timestamp-regression` 三个独立开关。注入自检成功时退出码为 0，输出中的 `code` 分别固定为 `E_FORMAT`、`E_UPGRADE`、`E_ORDER`。

## 判定规则摘要

- 容量：`累计已放行写入字节 + 本次 write_bytes <= capacity_bytes` 放行；超过则 `OVER_QUOTA`。
- 令牌桶：令牌按 1/1000 单位量整数存储；`增量 = delta_ms * rate_per_sec`，上限为 `burst_capacity * 1000`。
- 顺序：先判断配额，再判断限流；同时越界时返回 `OVER_QUOTA`。拒绝事件不消耗令牌、不增加已放行写入量。
- 边界：容量和令牌均采用“等于可通过”。时间戳只接受整数毫秒；换算采用向下取整到 1/1000 单位，但整数公式无余数。
- 误差：单事件和聚合量化误差上界均为 0，并逐条写入 `judgments`，同时记录在账本计算策略元数据中。
- 派生：`derived = upstream_effective_capacity * factor + offset`；与显式容量冲突时取较小值，即同时满足显式上限和派生上限，并记录仲裁依据。成环或缺失引用返回 `E_DERIVE`。

## 业务码与日志

| 业务码 | 含义 |
| --- | --- |
| `E_FORMAT` | 配置/事件格式、未知版本、缺字段、负数、损坏 JSON |
| `E_UPGRADE` | 旧条目不可读、检查点缺失/损坏、升级无法安全继续 |
| `E_DERIVE` | 派生成环或引用缺失 |
| `E_ORDER` | 事件序号或时间戳倒退 |
| `E_PREVIEW` | 快速预览与权威重放不一致 |
| `E_LATENCY` | 单次预览超过显式整数毫秒预算 |
| `E_REPLAY` | 已提交批次无法按账本输入逐项复现 |
| `E_GUI` | 当前 Python 无 Tkinter 或无法创建窗口 |

每条业务错误写入 JSONL 结构化日志（时间、业务码、操作、卷、字段路径、详情），账本也会保存关键错误到 `structured_events`。详细设计见 `TECHNICAL.md`，实际验证结果见 `VERIFICATION.md`。
