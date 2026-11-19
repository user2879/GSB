# 审计事件关联与取证工作台

这是一个仅使用 Python 3.11 标准库实现的离线工作台。它把 v1 JSONL 与 v2 批量 JSON 审计事件流归一到同一口径，重建 trace 内因果链，按带版本号的规则聚合跨 trace 案件，并以原子检查点、案件快照和 JSON 报告保证乱序、重复、丢失与强制终止场景下的可恢复性。

运行期不访问网络、不依赖外部服务，也不使用 Docker。测试依赖只有 `pytest`；`jsonschema` 列入允许的可选开发依赖，但归一层不依赖它也可运行。

## 目录

```text
audit_forensics/       核心库与 CLI
scripts/               确定性样本生成、同步真值核对脚本
tests/                 pytest 回归测试
sample_data/           可重新生成的确定性 v1/v2 样本、规则与同步清单
```

## 安装

Windows PowerShell（Python 3.11）：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

也可仅安装项目后手工安装测试依赖：

```powershell
python -m pip install -e .
python -m pip install -r requirements.txt
```

Linux/macOS 同理，将虚拟环境激活命令改为 `source .venv/bin/activate`。

## 快速运行

生成固定种子 `20261119` 的本地样本：

```powershell
python scripts/generate_samples.py --output sample_data --force
```

导入混合格式样本。样本故意包含畸形行、畸形元素、冲突、孤儿、循环、跨 trace 父引用、seq 回退与空洞，因此退出码为 `1`（部分成功/存在业务错误），但仍会处理其余合法事件：

```powershell
python -m audit_forensics import `
  --workspace sample_data/workspace `
  --rules sample_data/rules.json `
  --input sample_data/input/v1_primary.jsonl sample_data/input/v1_conflict.jsonl sample_data/input/v1_malformed.jsonl sample_data/input/v2_batch.json `
  --workers 4
```

查询案件、导出报告、校验检查点：

```powershell
python -m audit_forensics query-case --workspace sample_data/workspace
python -m audit_forensics export-report --workspace sample_data/workspace `
  --output sample_data/workspace/report.export.json `
  --graph-output sample_data/workspace/case_graph.export.json
python -m audit_forensics verify-checkpoint --workspace sample_data/workspace
```

核对同步标记真值：

```powershell
python scripts/verify_sync.py --manifest sample_data/sync_manifest.json --report sample_data/workspace/report.json
```

## 数据口径

### 输入格式

- v1：逐行 JSONL，每行一个对象；必需字段为 `event_id`、`trace_id`、`actor`、`object`、`action`、`ts`（整数毫秒）、`seq`；可选 `parent_event_id`。
- v2：JSON 数组；字段口径与 v1 相同，`ts` 可为整数毫秒或 ISO-8601 字符串，可附带 `severity` 与 `source`。

格式识别只读取内容结构：数组根为 v2，JSON 对象/JSONL 为 v1。文件名和扩展名不参与判断。单个畸形行或数组元素会产生带 `source_format`、`source`、`position` 和业务错误码的记录，随后继续处理余下内容。

### 归一记录

跨格式规范化核心固定为八字段，顺序稳定：

```text
event_id, trace_id, actor, object, action, ts, seq, parent_event_id
```

`ts` 一律转为整数毫秒。ISO-8601 的时区偏移会先换算为 UTC；无时区字符串按 UTC 解释。`severity` 与 `source` 保存在事件的扩展属性中，不进入八字段核心，因此同一语义事件从 v1 与 v2 到达时核心记录逐字段一致。重复投递的扩展属性按确定性规则合并：`severity` 取严重度最高值并做稳定排序，`source` 取稳定排序后的首个非空值。

### 去重、冲突与序列诊断

- 事件键为归一后的 `event_id`；首个到达的归一核心作为主记录。
- 核心相同：计为重复投递，保留全部来源与同步标记，不产生冲突。
- 核心的 `actor`、`object`、`action`、`ts`、`seq`（并包括 `trace_id`、`parent_event_id`）任一不同：产生 `E_EVENT_CONFLICT`，记录双方来源格式与完整归一变体。冲突事件被隔离在规则案件之外，避免不确定内容污染案件结论。
- 到达顺序固定为 CLI 输入文件顺序加文件内位置；并发解析不会改变该顺序。
- `E_SEQ_ROLLBACK` 基于去除重复投递后的确定性到达顺序检测；`E_SEQ_GAP` 基于归一后的最终 seq 集合检测。两者与重复投递分类独立。

### 因果链

按 `parent_event_id` 在全部归一事件中解析父引用：

- 父事件不存在：`E_ORPHAN`
- 父引用自环或形成环：`E_CYCLE`
- 父事件存在但属于其他 trace：`E_CROSS_TRACE`

每个 trace 输出稳定的根节点、DFS 顺序、节点深度、父/子关系与 seq/ts。结论只依赖归一事件集合，不依赖到达顺序或格式比例。

## 规则与案件

规则文件示例：

```json
{
  "rules_version": "2026.11.1",
  "rules": [
    {
      "rule_id": "R-VAULT",
      "name": "Vault access",
      "conditions": {
        "actors": ["alice"],
        "objects": ["vault"],
        "severities": ["high", "critical"],
        "time_window": {
          "start": "2026-11-19T00:00:00Z",
          "end": "2026-11-19T23:59:59Z",
          "window_ms": 600000
        }
      },
      "group_by": ["object"]
    }
  ]
}
```

`conditions` 支持 `actors`、`objects`、`actions`、`severities`，以及绝对时间窗 `start_ms`/`end_ms` 或 `start`/`end`。`window_ms` 是可选的会话分段间隔。`group_by` 可从 `actor`、`object`、`action`、`severity`、`trace_id` 中选取。采集器字段 `source` 不参与规则，避免同一事件在不同采集格式间切换时改变案件结论。

每次导入都从完整事件集合全量重算案件，没有增量续接路径。案件键由规则版本、规则 ID、分组值和排序后的成员事件 ID 哈希得到。规则版本变化，或同名版本下规则内容变化，都会记录重建原因并替换全部案件。排序、分组、成员 ID 和窗口切分均使用稳定排序，因此任意合法到达顺序及 v1/v2 混合比例下案件键和成员一致。

## 状态、原子性与恢复

工作状态位于 `<workspace>/.audit/`：

```text
checkpoint.json       权威检查点：归一事件、冲突、文件结论、水位、规则与全量案件
checkpoint.prev.json  最近一次可读的完整旧检查点
case_snapshot.json    派生案件快照
workspace.lock        跨进程非阻塞文件锁
```

检查点与快照均使用同目录临时文件，`fsync` 后 `os.replace` 整体替换。加载端只读取固定最终文件名，并通过规范化 JSON 的 SHA-256 校验载荷完整性；半写临时文件不会被当作有效状态。检查点载荷先于快照提交，快照携带 `generation` 与 `checkpoint_checksum`；若进程在两者之间终止，读取端会识别 `E_SNAPSHOT_STALE` 并从检查点全量重建派生结论。

当前检查点不可读时会回退到 `checkpoint.prev.json`。检查点结构版本、规则版本或规则内容与当前输入不一致时，会从保留的完整事件状态整体重建，不会把旧案件与新规则混合。工作区写操作使用非阻塞文件锁；POSIX 使用 `fcntl.flock`，Windows 使用 `msvcrt.locking`。锁冲突返回 `E_LOCK`。

报告和案件图同样原子写出。报告不含运行时钟时间，字段按固定构造顺序序列化；同一状态重复导出得到完全相同的字节。

### 确定性故障注入（仅测试）

测试通过环境变量在独立子进程中选择确定性边界，不会在生产正常运行时生效：

```text
AUDIT_FORENSICS_TEST_PAUSE_AT=checkpoint|case_snapshot|report|graph
AUDIT_FORENSICS_TEST_PAUSE_SECONDS=30
AUDIT_FORENSICS_TEST_READY_FILE=<父进程等待的标记文件>
AUDIT_FORENSICS_TEST_HOLD_LOCK_SECONDS=2
AUDIT_FORENSICS_TEST_LOCK_READY_FILE=<持锁就绪标记文件>
```

暂停发生在临时文件已写入并 `fsync`、`os.replace` 尚未执行时，父进程可在此精确终止子进程并验证恢复行为。

## CLI

四个逻辑子命令（括号内为兼容别名）：

```text
import (ingest)                 导入多个 v1/v2 文件
query-case (case, query)        查询全部案件或指定案件
export-report (export, report)  导出报告与案件图
verify-checkpoint (verify, checkpoint)  校验检查点、快照与全量重算
```

退出码：

- `0`：全部成功且无业务错误
- `1`：部分成功、存在业务错误、锁冲突、状态不可用或校验失败
- `2`：命令行参数或规则配置错误

所有命令输出可解析 JSON；导入与导出同时把完整业务错误码写入 JSON 报告。

## 主要业务错误码

| 错误码 | 含义 |
|---|---|
| `E_MALFORMED_JSON` / `E_MALFORMED_EVENT` / `E_SCHEMA` / `E_TIMESTAMP` | 畸形 JSON、畸形元素、字段结构或时间错误 |
| `E_ORPHAN` / `E_CYCLE` / `E_CROSS_TRACE` | 缺失父事件、父引用环、跨 trace 父引用 |
| `E_EVENT_CONFLICT` | 同一 event_id 的归一核心内容冲突 |
| `E_SEQ_ROLLBACK` / `E_SEQ_GAP` / `E_SEQ_DUPLICATE` | 到达序回退、序空洞、同 trace 重复 seq |
| `E_LOCK` | 工作区写锁冲突 |
| `E_CHECKPOINT_*` / `E_SNAPSHOT_*` / `E_STATE_SCHEMA_MISMATCH` | 检查点和快照缺失、损坏、过期或结构版本变化 |
| `E_RULES_VERSION_CHANGED` / `E_RULES_CONFIG_CHANGED` | 触发全量案件重建（记录在 `notices`） |

## 样本同步标记

`sample_data/sync_manifest.json` 为每个合法发送端事件显式记录 `source_format`、`source_file`、`position`、`original_ts` 与 `sync_marker`。标记由事件 ID、来源格式、文件名、位置和原始时间表示确定生成。验证端只使用该发送端记录与报告中的归一事件/来源记录进行比较，不读取生成器的预期归一对象。

## 测试

```powershell
python -m pytest -q
```

测试覆盖跨版本归一、内容识别、畸形继续处理、重复与冲突、seq 回退/空洞、三类因果错误、规则版本重建、顺序及格式比例不变、串并行一致、锁冲突、检查点主备恢复、快照过期、两种强制终止边界、报告字节稳定性和同步标记真值核对。

本机实际验证结果见 [VERIFICATION.md](VERIFICATION.md)。
