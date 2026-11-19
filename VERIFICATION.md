# 实际验证记录

验证日期：2026-11-19  
执行环境：Ubuntu 22.04 / WSL2，Python 3.11.0rc1，pytest 9.2.0  
项目依赖：运行期仅 Python 标准库；pytest 仅用于测试。

Windows 专用锁分支使用 `msvcrt.locking`，POSIX 验证使用 `fcntl.flock`。原子替换统一使用 `os.replace`，两条路径均不依赖容器或外部服务。

## 1. 自动化测试

命令：

```text
python -m pytest -q
```

实际结果：

```text
...................                                                      [100%]
19 passed in 6.74s
```

覆盖项：

- v1/v2 内容识别、整数与带时区 ISO-8601 毫秒归一、核心记录逐字段相等。
- 畸形 JSONL 行和 v2 畸形元素带格式/位置继续处理。
- 重复投递、内容冲突、seq 回退、seq 空洞相互独立。
- `E_ORPHAN`、`E_CYCLE`、`E_CROSS_TRACE` 与 v1/v2 混合父链。
- 规则版本变化、同名版本规则内容变化均全量重建。
- 全 v1、全 v2、混合格式、逆序输入、串行与 8 线程解析的案件键/成员一致。
- 跨进程锁冲突返回 `E_LOCK`。
- 当前检查点损坏时回退此前完整检查点；快照 generation/checksum 过期可识别。
- 两个独立子进程强制终止边界：检查点替换前终止、检查点已替换但快照替换前终止。
- 报告重复导出字节完全一致。
- 同步清单 15 条 occurrence 与归一结果逐条一致。

## 2. 确定性样本导入

样本由固定种子 `20261119` 生成：

```text
python scripts/generate_samples.py --output sample_data --force
```

实际输出：

```json
{
  "status": "ok",
  "output": "sample_data",
  "seed": 20261119,
  "occurrences": 15
}
```

导入命令使用 4 个并发解析线程。样本故意包含业务错误，因此整体退出码为 `1`，其余合法事件正常处理：

```text
IMPORT_EXIT=1
```

报告结论：

```json
{
  "status": "partial",
  "files_total": 4,
  "files_ok": 2,
  "events_normalized": 13,
  "events_conflicted": 1,
  "traces": 8,
  "cases": 4,
  "business_error_count": 10,
  "error_codes": [
    "E_CROSS_TRACE",
    "E_CYCLE",
    "E_EVENT_CONFLICT",
    "E_MALFORMED_JSON",
    "E_ORPHAN",
    "E_SCHEMA",
    "E_SEQ_GAP",
    "E_SEQ_ROLLBACK",
    "E_TIMESTAMP"
  ]
}
```

检查点水位：

```json
{
  "generation": 1,
  "unique_events": 13,
  "processed_occurrences": 15,
  "arrival_watermark": 14,
  "max_ts": 1795046412000
}
```

## 3. 跨版本归一与案件关联

同一条 `evt-100-002` 同时由 v1 整数时间和 v2 ISO-8601 时间投递。导入后只有一个归一事件，状态为 `duplicate`，`occurrence_count=2`，核心八字段完全相同；v2 的 `severity=high` 只进入扩展属性并按规则生效。

冲突事件 `evt-700-001` 在两个格式中 `actor` 不同：

```json
{
  "event_id": "evt-700-001",
  "fields": ["actor"],
  "source_formats": ["v1", "v2"]
}
```

冲突事件不进入案件。4 个案件键与成员如下：

```text
CASE-55A36415BE5A5EABA7A4778F  R-VAULT-ACCESS
  evt-100-001, evt-100-002, evt-100-003, evt-600-001, evt-600-002
CASE-C8077311178C2AFEE36B008C  R-DB-OBJECT
  evt-200-001, evt-200-003
CASE-E0F76BE398FDDFBD63008BA4  R-HIGH-SEVERITY
  evt-100-002, evt-100-003, evt-600-001, evt-600-002
CASE-F7666705261BDF6757923819  R-HIGH-SEVERITY
  evt-200-001, evt-200-003
```

自动化测试分别以全 v1、全 v2、混合格式、逆序输入运行，案件签名完全一致。

## 4. 同步标记真值核对

命令：

```text
python scripts/verify_sync.py --manifest sample_data/sync_manifest.json --report sample_data/workspace/report.json
```

实际结果：

```json
{
  "ok": true,
  "checked": 15,
  "errors": []
}
```

核对只使用同步清单中的来源格式、原始时间表示、位置、同步标记，以及报告中的归一事件与 provenance 记录，不使用生成器的预期归一对象。

## 5. 检查点校验与报告

命令：

```text
python -m audit_forensics verify-checkpoint --workspace sample_data/workspace
```

实际结果：

```json
{
  "status": "ok",
  "valid": true,
  "events": 13,
  "cases": 4,
  "checkpoint": {
    "generation": 1,
    "recovered_from_previous": false
  },
  "snapshot": {
    "valid": true,
    "cases": 4
  },
  "errors": []
}
```

样本报告与图文件：

- `sample_data/workspace/report.json`：47,041 bytes。
- `sample_data/workspace/case_graph.json`：9,801 bytes。
- 图包含 17 个节点、17 条边；节点区分事件与案件，边区分 `case_member` 与 `causal_parent`，并携带规则、分组或父引用依据。

## 6. 强制终止与锁验证

测试以独立 Python 子进程运行 CLI，并由父进程在确定性 failpoint 上发送终止信号：

1. 在检查点临时文件写完、`os.replace` 前终止：新的事件未进入状态；后续加载仍为上一完整 generation，临时文件未被作为有效状态加载。
2. 在检查点已整体替换、案件快照替换前终止：加载端识别快照与检查点 generation/checksum 不一致，报告 `E_SNAPSHOT_STALE`；查询从检查点事件全量重建内存案件，下一次导入原子修复快照。
3. 一个子进程持锁时，第二个进程导入返回 `E_LOCK` 和退出码 `1`；持锁进程正常完成。

检查点损坏回退测试中，当前 `checkpoint.json` 被截断后，加载端回退到 `checkpoint.prev.json`，其 generation 为上一条完整提交；过期快照被显式标记，随后可修复。
