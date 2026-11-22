# 实际验证结果

验证环境：Python 3.11、NumPy 1.26.4、SQLite3（标准库）。节点以真实独立子进程
绑定 127.0.0.1 临时高位端口，通过回环 TCP JSON Lines 通信；无容器、无线上服务。

## 自动化测试

`python -m pytest tests/` → **30 passed**。

| 测试文件 | 覆盖需求 |
| --- | --- |
| `test_manifest_validation.py` | E_MANIFEST 强校验、单调版本、环检测、非法键/布尔值/未知节点/策略 |
| `test_differ.py` | 五类变更分类、全下游影响链、拓扑序、未变更项不入链、dict 顺序无关 |
| `test_fixedpoint.py` | 向零截断、int64 饱和、整数误差界分量、沿链累计、界内/界外判定 |
| `test_node_and_transport.py` | 真实子进程 TCP、确定性摘要、平台独立哈希、首次故障静默关闭 |
| `test_scheduler.py` | 全局/每节点并发上限实测不超限、排队单调时钟、重试语义不变、E_EXHAUST |
| `test_pipeline_e2e.py` | 回归判定/首差异分组、no_receipt 不计干净、归因 permille、逐字节确定性、快照整体替换（含格式版本）、强制终止 unknown 恢复、E_REPLAY 定位 node.key.field |

## 样本端到端结果（samples/）

两版清单差异（`artifacts/diff_report.json`）：

- 变更项：`adjusted_rate`、`base_rate`、`composite_load`、`discount`、
  `fresh_gate`(新增)、`legacy_toggle`(删除)
- 未变更项：`region_load`、`stable_factor`（不作为影响链根）
- 定点口径：scale=10^6，向零截断，有符号 int64 饱和，纯整数乘加

完整采集会话（`artifacts/run_report.json`，退出码 1=检出回归）：

- 14 个 (节点,配置项) 判定：9 个 `E_REGRESSION`，5 个 `ok`
- 回归均记录两版摘要、逐分组差值、容差与首个差异分组键
- 归因成因：direct_value_change ×4、derived_coefficient_amplification ×2、mixed ×3
- 6 个采集任务全部 attempt 成功；会话内重放校验 `status=ok`
- 连续两次运行的 diff/error_bounds/findings/attributions 逐字节一致（仅采集时钟变化）

强制终止恢复（`artifacts/recover_report.json`）：

- 预置未提交租约 `batch-3-4-orphan` 后恢复，该项标记
  `unknown_rerun_required`，不生成终止时刻退出报告
- 快照识别为 `reused`（版本/摘要一致，未触发替换）
- 恢复后重算差异/影响/误差界并重放判定：`status=ok`，校验 14 findings、9 attributions，
  存储摘要与重放摘要 digest 一致

篡改重放（手工改账本批次 payload 的一个 finding.status）：

- 恢复重放报 `E_REPLAY`，定位首个不一致
  `subject=node-a.adjusted_rate.status`，依据 `ledger='ok' replay='E_REGRESSION'`

快照整体替换（测试内）：声明不同 scale 或不同 ledger format version 时
`action=replaced`，清空 batches/findings/leases 后重建并稳定列出替换项，不做增量合并。

## 复现命令

```bash
pip install -r requirements.txt
python -m pytest tests/
python -m config_regression.cli run \
  --baseline samples/baseline_manifest.json \
  --candidate samples/candidate_manifest.json \
  --registry samples/registry.json --session samples/session.json \
  --database artifacts/session.db --output artifacts/run_report.json
python -m config_regression.cli recover \
  --baseline samples/baseline_manifest.json \
  --candidate samples/candidate_manifest.json \
  --registry samples/registry.json --session samples/session.json \
  --database artifacts/session.db --output artifacts/recover_report.json
```

注：当前 Linux 容器缺少 Tk 系统库（`libtk8.6.so`），故 `cli gui` 返回退出码 3；
Windows 标准 python.org 安装自带 Tkinter，GUI 可直接打开，且 GUI 为可选组件，
不影响数据管线与测试。
