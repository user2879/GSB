# 多集群配置修订分发诊断台

面向“变更风险与回滚”场景的本地专业诊断应用：在 Windows 本机为每个集群启动
确定性模拟服务，维护配置修订链上的结果节点账本，支持界面增量更新、权威重建、
派生依赖失效传播、历史回滚隔离、强制终止恢复与独立重放校验。

仅依赖 Python 标准库（Tkinter、sqlite3、socket、subprocess）与
`numpy==1.26.4`，不使用容器，不访问网络外部目标。

## 环境准备（Windows）

```bat
python --version                 REM 需要 3.11 或 3.12
python -m pip install -r requirements.txt
python -c "import tkinter, sqlite3, numpy; print('ok')"
python scripts\generate_samples.py --out samples
```

运行期数据统一写入工作区 `data/`；样本端口固定为 127.0.0.1:19001-19003，
运行前请确认端口空闲。所有演示与测试在退出时清理子进程。

## 命令行操作

```bat
python -m diagconsole validate                         REM 强校验声明
python -m diagconsole apply                            REM 增量分发并记账
python -m diagconsole rebuild --root 1                 REM 权威整树重建并逐节点比对
python -m diagconsole rollback --target 3              REM 回滚到历史修订并隔离陈旧节点
python -m diagconsole view                             REM 查看活动视图与隔离节点
python -m diagconsole snapshot-save                    REM 持久化会话快照
python -m diagconsole recover                          REM 强制终止后恢复
python -m diagconsole rerun                            REM 重跑 unknown 节点
python -m diagconsole replay                           REM 快照一致性重放
python -m diagconsole inject corrupt-spec              REM -> E_SPEC
python -m diagconsole inject dangling-derives          REM -> E_REF
python -m diagconsole inject snapshot-digest           REM -> E_SNAPSHOT
python -m diagconsole demo --crash                     REM 可重复端到端演示（含强制终止）
run_ui.bat                                             REM 启动 Tkinter 交互界面
```

加 `--local` 可在进程内直接计算（不起服务），便于无头环境快速验证；
默认模式会为每个集群拉起真实子进程服务。`--samples` 与 `--data`
可覆盖默认的 `samples/`、`data/` 目录。

## 交互界面

`run_ui.bat`（或 `python run_ui.py`）打开 Tkinter 诊断台：

- 仅对本批受影响节点做行内增量更新，进度条显示任务/集群/修订；
- 表格展示修订、任务、集群、聚合总量、内容摘要与节点状态；
- 回滚后链外节点标记为 `isolated (read-only)`，统计口径只统计活动根链；
- 无头环境启动会得到明确的 HeadlessError 提示并退回 CLI。

## 八项业务要求落点

1. 强校验：`diagconsole/model.py`，E_SPEC 定位首个违规修订与字段路径。
2. 子进程编排：`diagconsole/orchestrator.py` + `service.py`，
   E_PORT / E_HANDSHAKE / E_REPLY 均带集群标识。
3. 结果账本：`diagconsole/ledger.py`，幂等写入、同事务父引用校验、提交前不可见。
4. 增量与重建分离：`engine.apply_chain`（仅失效节点走服务）与
   `engine.rebuild`（整树重算逐节点比对，不一致 E_PREV）。
5. 失效传播：`engine.analyze_invalidation`，输出失效集合与每步传播依据，
   未失效节点逐字节比对。
6. 回滚隔离：`engine.rollback`，整体替换根/视图/统计，陈旧节点只读列出，
   回滚前后内容摘要可逐字比较。
7. 崩溃恢复：`engine.recover/rerun_unknown` + `snapshot.py`，
   未提交节点标记 unknown 要求重跑，不生成强制终止退出报告。
8. 重放与注入：`engine.replay`（E_REPLAY 定位首个节点与字段），
   `inject.py` 提供 E_SPEC / E_REF / E_SNAPSHOT 三类注入。

## 验收与实测结果

一键演示：

```bat
run_demo.bat
REM 或
python -m diagconsole demo --crash
```

本工作区在 Python 3.11 + numpy 1.26.4 下的实测（`python -m unittest` 与 demo）：

- 单元/集成测试 32 个全部通过（含真实子进程、端口占用、握手超时、回执缺失）。
- `demo --crash` 验收四项全绿：
  - 增量视图与权威重建逐节点一致；
  - 修正案仅重算失效子集，未失效节点聚合与内容摘要逐字节保持；
  - 回滚 r3 后 r4-r6 节点整体隔离、不参与统计；
  - 强制终止后恢复出 unknown 节点并重跑，快照重放全部可复现。
- 三类注入分别得到 E_SPEC、E_REF、E_SNAPSHOT，重放篡改得到 E_REPLAY。

> 注：当前 Linux 沙箱缺少 Tk 系统库（`libtk8.6.so`），界面代码已做无头保护；
> 在安装官方 Python 的 Windows 上 Tkinter 为标准库自带，可直接 `run_ui.bat`。

更多设计细节见 `docs/ARCHITECTURE.md`。
