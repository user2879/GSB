# 实际验证记录

验证时间：2026-11-18（Asia/Shanghai）

## 环境

- Python：`3.11.0rc1`，满足 3.11+。
- `json`、`sqlite3`：可用。
- 当前工作目录：可写。
- `numpy`：用户态安装 `2.4.6`，可用；Windows 安装入口在 `requirements.txt`。
- `tkinter`：默认 Linux 解释器未内置。验证时通过 `apt-get download` 下载并解包 Ubuntu Tk 8.6 运行库到 `/tmp`，通过 `PYTHONPATH`/`LD_LIBRARY_PATH` 完成真实窗口验证，未修改系统目录。
- 无 Docker、无外部服务、无网络样本。

默认解释器的 `doctor` 会如实报告 Tk 缺失；附加本地 Tk 运行库后的实际 `doctor` 输出（摘要）：

```json
{
  "core_ready": true,
  "full_ready": true,
  "checks": {
    "python": {"ok": true, "version": "3.11.0rc1"},
    "json": {"ok": true},
    "sqlite3": {"ok": true},
    "numpy": {"ok": true, "version": "2.4.6"},
    "tkinter": {"ok": true},
    "writable_workspace": {"ok": true}
  }
}
```

界面部分采用惰性导入：核心、测试、CLI、视图模型均不因缺少 Tk 而失败。附加 Tk 运行库后，`ui --check --require-tk` 实际创建并销毁完整窗口，结果：

```json
{
  "ok": true,
  "tkinter_available": true,
  "tk_version": "8.6",
  "workbench_ready": true,
  "window_size": [1320, 840],
  "tree_rows": 36,
  "cooling_rows": 7,
  "canvas_items": 29,
  "evidence_chars": 950
}
```

## 自动测试

执行：

```text
python3 -m unittest discover -s tests -v
```

结果：`Ran 38 tests ... OK`。

附加 Tk 8.6 运行库后再次执行同一测试套件：`Ran 38 tests ... OK`；其中 UI 检查实际构造了完整工作台窗口。

覆盖的实际断言包括：

- `E_SPEC` 字段路径：重复工作负载、分钟不连续/重复、负请求、错误数超请求、上下限、冷却、预算、指标枚举、目标范围、分钟网格、布尔伪装整数、JSON 损坏。
- 期望副本整数公式：分母、未夹取值、量化误差 667 毫单位、规则并列取最小 ID。
- 预算分配与输入字典/列表顺序无关，且始终不超过预算。
- 冷却抑制最终保持上轮副本，并回溯触发分钟。
- 全量/增量摘要完全一致：`fd26cacba76eafbe7b976a25d4fa9d1eb83fbdb204ee182724f39cdf3a93095d`。
- 每分钟事务被数据库触发器强制失败时，决策与冷却行均回滚。
- 未最终确认的完整前缀不能生成退出报告，并登记 `COMPLETION_UNCONFIRMED`。
- 声明版本、同版本预算参数、观测记录差异均被 `E_REPLAY` 定位。
- 整体替换清理旧恢复事件。
- 三场景端到端演示输出全部文件。

## 三场景演示

执行：

```text
python3 demo.py artifacts/demo
```

`artifacts/demo/summary.json` 的实际检查全部为 `true`：

```json
{
  "forced_termination_committed_prefix": true,
  "resume_completed_run": true,
  "cooldown_suppression_observed": true,
  "replay_verified_all_minutes": true,
  "budget_replacement_complete": true,
  "budget_changed": true,
  "path_before_full": true,
  "path_after_incremental": true,
  "path_parity": true,
  "version_mismatch_guard": true,
  "ui_viewmodel_check": true
}
```

关键实际值：

- 强制终止退出码：137；可见已提交分钟：4/12；无退出报告。
- 恢复登记：`UNKNOWN_MINUTE_REQUIRES_RERUN`，`minute_index=4`。
- v1 预算 6，自动路径 full，冷却抑制 8 次。
- v1 账本摘要：`1f02956daa132fef723dac4f1f19a30cb9a22d869f7bb3adb41cd08b649e9d13`。
- v2 预算 5，自动路径 incremental，冷却抑制 7 次，预算裁剪 7 次。
- v2 账本摘要：`5ca2988b58f5609e1004df3ac2cc018cd794b2b2bd0abfccedc0493fe6115c82`。
- 旧声明重放新账本：退出码 3，`E_REPLAY`，字段 `declaration_version`。
- 替换前账本保留为 `artifacts/demo/workbench.db.bak`。

## 可审阅产物

- `artifacts/demo/01_cold_start.json`
- `artifacts/demo/02_cooldown_suppression.json`
- `artifacts/demo/03_budget_replacement.json`
- `artifacts/demo/04_path_parity.json`
- `artifacts/demo/05_replay_guard.json`
- `artifacts/demo/06_ui_viewmodel_check.json`
- `artifacts/demo/v1/report.html`
- `artifacts/demo/v1/report.json`
- `artifacts/demo/v1/snapshot.json`
- `artifacts/demo/v2/report.html`
- `artifacts/demo/v2/report.json`
- `artifacts/demo/v2/snapshot.json`
- `artifacts/demo/workbench.db`
- `artifacts/demo/workbench.db.bak`

HTML 报告是自包含静态领域视图，可在没有 Tk 的环境直接审阅预算曲线、逐分钟决策、冷却和恢复事件。Tkinter 工作台在具备 Tk 的 Windows Python 上通过 `run_ui.bat` 启动。
