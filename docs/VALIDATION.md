# 实际验证记录

验证时间：2026-11-23（Asia/Shanghai）。

## 环境

```text
Python      3.11.0 (CPython)
SQLite      3.37.2
NumPy       1.26.4
平台        Linux 反馈工作区（目标运行环境为本地 Windows）
tkinter     ImportError: libtk8.6.so 缺失；核心 CLI 不受影响
```

`python -m pip install -r requirements.txt` 输出 `Requirement already satisfied: numpy==1.26.4`；`pip show numpy` 确认版本为 `1.26.4`。只读 `selfcheck` 返回 `pass_with_warnings`，唯一警告是当前 Linux 容器缺少 Tk 动态库；Windows 安装带 Tcl/Tk 的 Python 后可使用 `--require-tkinter` 严格通过。

## 样本结果

`samples/session.json` + `samples/records.jsonl`：

```text
记录数                 34
v1 / v2 / v3           9 / 9 / 16
全局结论               at_least_once=8, at_most_once=1, exactly_once=1
冲突                   ACK_WITH_UNCOVERED_RANGE=2
                       DUP_IDEMPOTENCY_ACROSS_GROUPS=1
                       SEMANTIC_MARKER_MISMATCH=1
model_digest           33d913b9cf79307a56f6306b6d59ba398c08180c47587f0e7e55b7d230133728
conclusions_digest     bf4f78a1471bfb37f9caa008a856d29097ccd73f332a474cd068192000891dd4
conflicts_digest       bfa791a66fedf41868590a519050804a513303eaca2c095e872a71a24bb71d96
stats_digest           eff2d0b04ba5e71b26da5ab4fa868ac22863ed29dd96d7c5df5f0c63eb59a175
snapshot digest        8cc3b88c78eb3b1ffbf82a244b1e3daaba66f1bc7d83da8d56e46e572d18d5f0
```

主题定点精确一次占比：

```text
archive   1/1   fixed=1000000
orders    4/16  fixed=250000   error_affected=true
payments  3/12  fixed=250000   error_affected=true
audit     1/5   fixed=200000
```

`orders` 与 `payments` 的固定差值 0 落在 `<=2` 合成误差界内；使用字典序稳定决胜，输出顺序为 `archive, orders, payments, audit`。

## 命令验证

1. 主流程：`python -m mwlab run ... --expected samples/expected_summary.json` 成功，四个主题均生成批次。
2. 独立校验：`python -m mwlab verify ...` 返回 `status=ok`、`byte_identical=true`、`conclusion_count=10`、`conclusions_bytes=5936`。
3. 快照：保存 `baseline` 后恢复，`replacement_mode=wholesale_copy_no_merge`；内容摘要前后均为 `8cc3b88c...d18d5f0`。
4. 完整恢复：`recover` 返回 `status=complete`、`replay_ok=true`、`exit_report_available=true`。
5. 强制终止：`run --crash-after-topic 1` 退出码 91；随后 `recover` 返回 `status=incomplete`、仅 `archive` 已提交、`audit/orders/payments` 需重跑、`exit_report_available=false`，未提交消息标记 `unknown_uncommitted_topic`。
6. 重跑恢复：重新完整运行后 `recover` 返回完整、重放一致。
7. 参数演进：同账本改用 `session_evolved.json` 后参数版本 `1 -> 2`，旧四个批次全部 `stale=1`，新视图差异稳定报告 `orders:4` 的 `basis` 变更。
8. 快照整体替换：演进后的账本恢复参数版本 1 的快照，报告 `content_digest` 不一致并整体复制替换，旧账本保存在只读备份。

## 异常业务码

| 命令 | 结果 |
|---|---|
| `run --inject-format` | `E_FORMAT`，退出码 3，`line=1`、`field=record.retry_count`、`value=-1` |
| `run --inject-dupkey` | `E_DUPKEY`，退出码 4，键 `idem-payments-shared` 跨 `analytics/warehouse` |
| `snapshot restore --inject-digest` | `E_SNAPSHOT`，退出码 5，同时报告存储摘要与计算摘要 |
| `verify --inject-divergence` | `E_DIVERGE`，退出码 6，首个差异 `archive:1/verdict`，来源 `semantic_derivation` |
| 错误模型摘要验证 | `E_DIVERGE`，来源 `format_normalization` |
| 篡改账本已提交结论后 `recover` | `E_REPLAY`，退出码 7，首个差异 `archive:1/verdict` |
| `run --inject-format` / 反向区间样本 | `E_FORMAT`，详情均可定位行号和字段 |

结构化日志 `logs/faults.jsonl` 和演示日志 `logs/demo-structured.jsonl` 均包含 `command_error`、`code`、`message`、`details`、`timestamp_utc` 和 PID。

## 自动化测试

```text
python -m unittest discover -s tests -t . -v
Ran 15 tests ... OK
OK
```

测试覆盖：

- v1/v2/v3 归一与来源行号；
- 未知版本、缺失字段、负重投、倒置区间、组版本不匹配；
- 原始行随机乱序后模型摘要、结论与冲突字节一致；
- 消费者组声明顺序反转后结论字节一致；
- 三种语义结论及全局汇总；
- 四类冲突归因与稳定排序；
- 定点比例、单值误差、比较误差与稳定决胜；
- 主流程/独立校验逐字节一致及两种 `E_DIVERGE` 来源；
- 参数版本递增、旧批次陈旧、整体视图差异；
- 格式版本、确认策略和消费者组集合三类变化分别递增参数版本；
- 快照保存、只读旧快照备份、摘要注入、整体恢复；
- 强制终止、未知消息、重跑恢复和 `E_REPLAY` 首差异定位。

## 一键演示

```text
python scripts/demo.py
...
{
  "status": "demo_pass",
  "selfcheck": "pass_with_warnings",
  "model_digest": "33d913b9...133728",
  "conclusions_digest": "bf4f78a1...891dd4",
  "conflicts_digest": "bfa791a6...b71d96",
  "stats_digest": "eff2d0b0...9a175",
  "snapshot_content_digest": "8cc3b88c...d18d5f0",
  "parameter_version_base": 1,
  "parameter_version_evolved": 2,
  "negative_codes": ["E_FORMAT", "E_DUPKEY", "E_SNAPSHOT", "E_DIVERGE", "E_REPLAY"]
}
```
