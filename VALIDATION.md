# 实际验证记录

验证日期：2026-11-20（Asia/Shanghai）

## 环境

- Python：3.11.0rc1
- NumPy：2.4.6
- SQLite：Python 标准库 `sqlite3`
- Tk：8.6（WSL 显示 `:0`）
- 项目根目录：`/home/zsl/feedback/uidfc5909d3-default-310`
- 验证方式：本地确定性样本；无网络、无端口、无外部服务

本机 WSL 镜像未预装 Tk 绑定，验证时通过本地提取的 Ubuntu `python3-tk` / Tk 8.6 运行时设置 `PYTHONPATH` 和 `LD_LIBRARY_PATH`。应用代码本身只使用标准 `tkinter`，Windows 官方 Python 通常已内置。

## 自动化测试

命令：

```bash
python3 -m unittest discover -s tests -v
```

结果：

```text
Ran 35 tests in 4.178s

OK
```

覆盖点：

- 表达式空白/冗余括号规范化和 AST 摘要一致；
- `if()` 与 `?:` 规范化一致；
- 重复列、重复任务、未知引用、空值位图长度校验；
- UI 非法编辑返回 `E_DATA` 且状态和摘要不变；
- 语义等价编辑不递增参数版本，真实编辑递增并标陈旧；
- 分片首尾相接、确定性摘要、缺口/重叠/越界 `E_RANGE`；
- 空值传播、除零位图、三元未选分支不泄漏异常；
- 逆序完成结果仍按行区间合流，规范字节一致；
- 全局预算峰值和每任务并发峰值不超过上限；
- 运行中修改预算后 running/queued 分片稳定失效；
- SQLite 格式版本与分片账本记录；
- 正常重放一致、篡改后首差异 `E_REPLAY`；
- 快照参数/内容不一致时整体替换；
- 真实子进程 `os._exit(137)` 后恢复与重放；
- CLI 参数解析、日志目录可写和三类注入业务码；
- Tkinter 构造、计划时间线渲染冒烟。

## 正常执行

样本：`samples/data_blocks.json` + `samples/tasks.json`  
全局预算：4；线程：4；任务：3；分片：11。

实际结果：

```json
{
  "status": "committed",
  "parameter_version": 1,
  "content_digest": "ba1513ee1fc735b4a6aebce3f6e1156251de5a88b9b6f841cd24638e964fa68e",
  "plan_digest": "e967b91052655b844acaf6884c1e64c42405873c66499054812d5b39484a272d",
  "overall_digest": "5bfaa993087ca7bdd80e3b54749c1801af764db49afbba7028a67f285654035d",
  "shard_count": 11,
  "peak_resources": 4,
  "task_peak_concurrency": {
    "inventory_margin": 1,
    "revenue_guard": 2,
    "risk_rows": 2
  },
  "task_result_digests": {
    "inventory_margin": "cc087b39f0f0f6aa129a700678fa231cc3d7202f3841f53ae084fd76ee58b1ed",
    "revenue_guard": "5a4afa223499b938751682ca144575ad16804911005910976c0d3117b015f885",
    "risk_rows": "ec857296a2cc7cc3dd0ca5c9fb93085606f36527546845e12935e763c753c9e7"
  },
  "ledger_format_version": 1,
  "ledger_status": "committed",
  "ledger_cumulative_digest": "e1f129ca302ba1ed70c412267164c0ce2014f216a3defaad3becc88e045971fd"
}
```

重复执行两次得到相同的 `overall_digest` 和逐字节相同的任务结果规范 JSON，测试 `test_repeated_execution_has_byte_identical_results` 验证。

## 异常注入

实际命令与结果：

| 注入 | 命令 | 退出码 | 业务码 |
|---|---|---:|---|
| 行区间越界 | `vecflow validate --inject-range` | 2 | `E_RANGE` |
| 空值位图错位 | `vecflow validate --inject-nullmap` | 2 | `E_NULLMAP` |
| 预算耗尽 | `vecflow run --inject-budget --reset-state` | 3 | `E_BUDGET` |

结构化日志 `logs/vecflow.jsonl` 验证计数：

```json
{
  "E_BUDGET": 2,
  "E_NULLMAP": 1,
  "E_RANGE": 1,
  "E_RECOVERY": 1,
  "E_REPLAY": 1
}
```

注入只改变输入 JSON 副本；错误由真实校验器、计划器或调度器产生。

## 强制终止与恢复

命令：

```bash
vecflow run --force-crash-after 2 --reset-state ...
vecflow recover ...
vecflow replay ...
```

实际结果：

```text
被强杀进程退出码：137
恢复前：
  completed = 2
  unknown   = 9
  exit_report_available = false
  exit_report = null
```

未提交的 running/queued 分片全部标记 `unknown`，错误码 `E_RECOVERY`。恢复后重放按账本保存的参数版本执行：

```json
{
  "compared_shards": 2,
  "unknown_rerun_shards": [
    "inventory_margin#0000",
    "inventory_margin#0001",
    "inventory_margin#0002",
    "revenue_guard#0001",
    "revenue_guard#0003",
    "revenue_guard#0004",
    "risk_rows#0000",
    "risk_rows#0001",
    "risk_rows#0002"
  ],
  "result_digest": "5bfaa993087ca7bdd80e3b54749c1801af764db49afbba7028a67f285654035d",
  "exit_report_available_for_interrupted_run": false
}
```

## 一致性重放差异定位

正常批次 11 个分片全部重放一致。随后把首个已提交分片的账本 `result_digest` 篡改为 64 个 0，再执行重放：

```json
{
  "exit_code": 4,
  "code": "E_REPLAY",
  "details": {
    "shard_id": "inventory_margin#0000",
    "field": "result_digest",
    "expected": "0000000000000000000000000000000000000000000000000000000000000000",
    "actual": "2edabe2361037f2e3e4fb74d62d95313ee70f097e2f9a8262584e3602b38474b",
    "basis": "first mismatch between committed ledger record and deterministic replay"
  }
}
```

## GUI 冒烟

命令：

```bash
vecflow gui --smoke --smoke-ms 120
```

结果：

```json
{
  "ok": true,
  "blocks": 2,
  "tasks": 3,
  "timeline_rows": 11,
  "parameter_version": 1,
  "tk_version": 8.6
}
```

界面成功构造数据块摘要、任务摘要、11 行计划时间线、资源曲线画布、结果区和账本状态。

## 机器可读结果

完整验收输出：`artifacts/verification.json`。  
最终状态：`ok=true`。

复现命令：

```bash
python3 tools/generate_samples.py
python3 -m unittest discover -s tests -v
python3 tools/verify.py
```

无 Tk/显示环境时 GUI 测试会自动 skip；Windows 或带 Tk 的桌面环境会运行 GUI 冒烟。
