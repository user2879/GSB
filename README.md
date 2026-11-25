# 多环境部署配置一致性仲裁平台

本地、确定性、无外部服务依赖的跨环境配置仲裁系统。环境服务以子进程方式在
`127.0.0.1` 回环端口提供真实 TCP 仲裁；父进程完成声明校验、定点量化、三类口径
合并、全量/增量自适应、失效传播、配额调度与 SQLite 账本恢复。

## 快速开始

```bash
bash scripts/bootstrap.sh     # 构建 .venv（numpy 1.26.4 + pysqlite3-binary）
bash scripts/run_all.sh       # 预检 + 53 个测试 + 三个验收场景
```

也可单独运行：

```bash
source .venv/bin/activate
python -m arbplatform.cli envcheck   # 环境/端口/SQLite 预检
python -m arbplatform.cli demo       # 多环境仲裁 + 增量保持 + 重放
python -m arbplatform.cli large      # 规模自适应路径选择
python -m arbplatform.cli recovery   # 强制终止恢复 + 快照整体替换
python -m unittest discover -s tests # 测试套件
```

结构化结果写入 `results/*_result.json`，设计与规则见 `docs/DESIGN.md`。

## 输入与样本

* `config/manifest.json`、`config/manifest_large.json`：环境/实例/视图声明。
* `samples/events.ndjson`、`samples/events_large.ndjson`：显式规则 + numpy 种子生成的确定性事件流。

## 业务错误码

| 码 | 含义 |
| --- | --- |
| `E_SPEC` | 声明校验失败，定位首个非法字段 |
| `E_PORT` | 回环端口占用/绑定或连接失败 |
| `E_TIMEOUT` | 真实套接字握手或 IO 超时 |
| `E_REPLY` | 服务回执缺必填字段 |
| `E_REPLAY` | 强制终止后账本重放首个不一致 |
| `E_FIXED` | 定点量化规则违例 |
| `E_STATE` | 非法平台状态迁移 |
