# 实际验证记录

## 环境

- 执行时间：2026-11-23 07:33（Asia/Shanghai；报告内 UTC 为 2026-11-22T23:33:37Z 至 23:33:53Z）
- 平台：WSL/Linux（Windows 兼容实现，未使用容器）
- Python：3.11.0rc1
- SQLite：3.37.2
- NumPy：1.26.4（已安装但运行时不需要）
- Tkinter：当前 Linux 验证环境缺少 `libtk8.6.so`，图形窗口未实际启动；无头视图模型与稳定渲染顺序已测试。

环境报告：`artifacts/doctor-report.json`。

## 自动化测试

命令：

```powershell
python -m pytest -q
```

结果：`30 passed in 9.89s`。原始输出：`artifacts/pytest-output.txt`。

覆盖点：

- `E_GEOM` 首个字段、块大小集合、精确镜像长度、块号边界、十六进制载荷。
- m1/m2/m3 编解码、归一指纹一致、补齐依据、未知版本、截断、校验和、摘要表、内容摘要、父链。
- 同名快照、断链、COW 位图与摘要同事务、整体替换差异和只读备份。
- 阈值边界、位图/full 逐位一致、块损坏预检、中断重跑、临时标记丢失判定、审计版本。
- 三类注入业务码与修复后逐块复核。
- CLI 退出码、JSON 输出与启动参数注入。

## 完整演示

命令：

```powershell
python -m snapdiag demo --workspace .
```

结果：8 个阶段全部通过，耗时 15.742 秒。报告：`artifacts/demo-report.json`，控制台摘要：`artifacts/demo-output.json`。

### 样本矩阵

| 样本 | 块大小 | 块数 | 卷大小 | 快照 | 执行步骤 |
| --- | ---: | ---: | ---: | ---: | ---: |
| small | 512 | 16 | 8 KiB | 3 | 13 |
| large | 4096 | 1024 | 4 MiB | 3 | 55 |
| high_churn | 1024 | 128 | 128 KiB | 3 | 102 |

### 三版本归一

- m1/m2/m3 归一指纹全部为：
  `d3b1497c311072bcaa8b2b16c37d83c8cdd9ed61edf54790eb2a2178e7318c84`
- `all_equal=true`。
- m1 补齐 19 项、m2 补齐 10 项、m3 补齐 0 项（按示例多个快照累计）。

### 整体替换

- 删除测试稳定列出 `removed=["m2"]`。
- 恢复测试稳定列出 `added=["m2"]`。
- 版本修改测试稳定列出 `replaced=["m0","m1","m2"]`。
- 3 份旧清单备份全部为只读。

### 双路径一致

| 选择 | 已变更/总块 | 比例 | 阈值 | 边界 | 路径 |
| --- | ---: | ---: | ---: | --- | --- |
| 强制 bitmap | 8/16 | 1/2 | 1/2 | true | bitmap |
| 强制 full | 8/16 | 1/2 | 1/2 | true | full |
| auto 边界 | 8/16 | 1/2 | 1/2 | true | full |
| auto 低比例 | 6/16 | 3/8 | 1/2 | false | bitmap |

三个目标相同的结果卷 SHA-256：

`0cc92abe32b70f771bb297b039156210fd45fd85d95d24995ccdb4abb3b82b7e`

恢复块集合均为 `[0,1,2,3,4,5,6,7]`；16 块逐块复核全部通过。

### 中断恢复

- `after_prepare` 中断：卷未替换，重启后标记 `incomplete_unknown`、`requires_rerun=true`；无 `--retry` 回滚报 `E_STATE`；显式重跑成功并逐块复核。
- `after_replace` 中断：卷已是目标摘要，恢复流程完成收尾、逐块复核并标记 `completed`。

### 异常注入

| 注入 | 观测业务码 | 修复/复核 |
| --- | --- | --- |
| 块数据损坏 | `E_INTEGRITY` | 从只读对象备份恢复；128 块全部复核，问题数 0 |
| 清单截断 | `E_FORMAT` | DB 权威清单重写，一致性恢复 |
| 清单版本冲突 | `E_VERSION` | DB 权威清单重写，一致性恢复 |

结构化异常日志位于各样本/演示会话的 `logs/diagnostics.jsonl`。

### 审计版本

- 第一次审计：版本 1，`status=pass`。
- 状态改变后读取旧结论：`E_VERSION`，同时给出旧/新的 `state_version` 与逐快照版本差异。
- 重新审计：版本 2，`status=pass`；版本 1 保留在历史中，未被覆盖。

## 未在本机执行的项

- Tk 图形窗口事件循环未启动，因为验证环境缺少 Tk 动态库；Windows 官方 Python 自带 Tk。界面使用的视图模型、分组顺序、各状态字段和异常路径均已由无头测试覆盖。
