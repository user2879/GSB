# 验证记录（本地确定性样本）

本文件记录在交付环境中实际执行的验证命令与结果。样本由固定整数种子
`20240517` 经 NumPy `PCG64` 派生（见 `obsflow/sample_data.py`），重复生成
逐字节一致，不访问网络或外部服务。

## 环境

- Python 3.11（要求 >= 3.10），NumPy 1.26（要求 >= 1.24），sqlite3 标准库。
- Tk 运行库在当前 Linux 容器缺失，Tk 界面懒加载并自动降级为无头表格；
  在 Windows 标准 Python 安装中 `tkinter` 可用，直接启动图形界面。

## 自动化测试

```
python -m pytest tests
```

结果：`27 passed`。覆盖：

- 三种历史格式编解码、强校验与归一来源（`tests/test_codec.py`）
- 定点量化、截断方向、饱和规则、误差上界与批次/节奏无关的逐字节确定性
  （`tests/test_aggregation.py`）
- 失效传播与仅失效子集重算、窗口参数变更整网重建（`tests/test_propagation.py`）
- 全局/每观测源双配额真实线程池调度、租约乐观续约与单调时钟真实过期
  （`tests/test_scheduler_lease.py`）
- SQLite 单事务账本、快照摘要、整体替换+只读备份、强制终止恢复、逐窗口
  重放定位、三类异常注入业务码（`tests/test_persistence.py`）
- 包对外接口完整性（`tests/test_package_interface.py`）

## CLI 验收矩阵

```
python -m obsflow.cli generate --output-dir samples/local
python -m obsflow.cli run --workdir artifacts/accept_run
python -m obsflow.cli replay --workdir artifacts/accept_run
python -m obsflow.cli inject truncate --workdir artifacts/accept_inj   # E_INTEGRITY
python -m obsflow.cli inject order --workdir artifacts/accept_inj      # E_ORDER
python -m obsflow.cli inject snapshot --workdir artifacts/accept_snap  # E_SNAPSHOT
python -m obsflow.cli crash --workdir artifacts/accept_crash
python -m obsflow.cli ui --workdir artifacts/accept_ui
```

实测：60 个窗口（3 指标键 × 20），重放逐字段一致；三类注入分别报
`E_INTEGRITY` / `E_ORDER` / `E_SNAPSHOT` 并写入结构化 JSONL 日志；
子进程被 SIGKILL（返回码 -9）后重开账本，60 个窗口全部标记 `unknown`，
重跑后恢复 `ready` 且重放通过。

## Windows 命令提示符步骤

```
py -m venv .venv
.venv\Scripts\activate
python --version
pip install -r requirements.txt
python -m obsflow.cli generate --output-dir samples\local
python -m obsflow.cli run --workdir artifacts\run
python -m obsflow.cli ui --workdir artifacts\run
python -m pytest tests
```

不使用 Docker 或其他容器；所有数据均落在本地目录。
