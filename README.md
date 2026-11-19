# 嵌入式键值存储索引与过期组件

纯 Python + 本地 SQLite 的嵌入式键值引擎：毫秒级过期、前缀二级索引（带条目版本号）、
派生计数视图、确定性轨迹执行、并发安全的索引重建、权威重放校验、会话快照恢复与异常
注入。全程离线，无容器、无外部服务。

## 快速开始

```powershell
# Windows（Python 3.10+，sqlite3/tkinter 随官方安装包提供）
py -m pip install numpy
python data\generate_sample.py
python run_demo.py
python -m unittest discover -s tests -v
```

Linux/macOS/WSL 等价命令：

```bash
python3 -m pip install --user numpy
python3 data/generate_sample.py
python3 run_demo.py
python3 -m unittest discover -s tests -v
```

无显示环境不影响引擎与测试；Tkinter 界面（`python -m kvstore.gui`）为可选项，
所有能力均可通过 `python -m kvstore.cli` 使用。

## 文档

- 设计、数据模型、并发论证与八项要求映射见 [DESIGN.md](DESIGN.md)。
- 端到端实验与确定性基准由 `run_demo.py` 产出到 `results/`。
- 业务错误码定义见 `kvstore/errors.py`。

## 业务错误码

| 码 | 触发条件 |
| --- | --- |
| E_RECORD | 键/前缀/过期戳/写入序号等记录非法，整体拒绝 |
| E_TRACE | 轨迹损坏、序号不连续/重复、时钟回退、PUT 缺字段 |
| E_EXPIRED | 注入逻辑时钟下键已过期 |
| E_INDEX | 派生计数或索引集合与键集合重算不一致 |
| E_VERSION | 索引条目版本回退/非法 |
| E_SNAPSHOT | 快照内容摘要或格式版本不匹配 |
| E_VISIBLE | 快速查询结论与权威重放分歧 |
| E_REPLAY | 快照恢复后重放结论/状态分歧 |
