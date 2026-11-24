# 配置权威仲裁与失效诊断

一个可离线运行的 Python 3.11 应用：加载多环境配置声明，以子进程托管回环模拟服务，通过真实套接字并发采集整数定点聚合摘要，完成三口径权威仲裁、误差归因、视图失效传播、SQLite 单事务落账和强制终止后的重放校验。附带 CLI 与可选 Tk 桌面界面。

## 快速开始

```bash
python3.11 -m pip install -r requirements.txt
python3 -m unittest discover -v
python3 -m config_arbiter demo
```

`demo` 会重建示例账本、执行首轮仲裁、模拟 `cn-prod` 切片变更、输出失效传播并完成两次重放。运行前确认样本端口 `46301-46303` 未被占用。

常用命令：

```bash
# 执行全部视图并验证重放
python3 -m config_arbiter run --reset --report artifacts/run-report.json

# 仅按当前声明重新计算并与账本逐键逐字段比对
python3 -m config_arbiter replay

# 打开 Tk 桌面界面（需要 Tk 运行库和图形显示）
python3 -m config_arbiter gui
```

Windows 可双击 `run_demo.bat` 或 `run_gui.bat`。GUI 依赖官方 Python 安装包自带的 Tk。

## 交付内容

- `config_arbiter/`：声明校验、定点运算、子进程服务、套接字客户端、双配额调度、三口径仲裁、失效引擎、SQLite 账本、CLI 与 GUI。
- `samples/sample_spec.json`：三个环境、两个合并视图、三种仲裁口径和并发/超时预算。
- `tests/`：42 项标准库 `unittest`，覆盖真实套接字、错误码、乱序确定性、并发峰值、失效替换、摘要替换、SIGKILL 恢复和 `E_REPLAY`。
- `scripts/run_acceptance.py`：一次运行环境信息、全部测试和验收演示。
- `docs/TECHNICAL.md`：协议、量化公式、仲裁决胜、误差界、失效与账本设计。
- `artifacts/acceptance-output.txt`：最近一次实际验证输出。
- `artifacts/acceptance_report.json`：最近一次验收的结构化结果。

## 样本结果概览

```text
global-release v1
  cpu_limit      weighted_majority  1.234  adopted=cn-prod,us-prod
  memory_limit   interval_midpoint  7.65   adopted=cn-prod,us-prod
  replicas       latest             3      adopted=cn-prod
us-eu-cpu v1
  cpu_limit      interval_midpoint  1.166  adopted=eu-prod,us-prod
  memory_limit   weighted_majority  7.8    adopted=us-prod
```

`cn-prod` 从版本 2 变更为版本 3 后，仅 `global-release` 失效，受影响键为 `cpu_limit`、`replicas`；`us-eu-cpu` 因不引用该环境而保持版本 1。旧结果整体标为陈旧，当前展示只读取新版本。

实际摘要哈希、错误归因、重放字段数和陈旧行数见 `artifacts/acceptance-output.txt`。

## 数据格式

最小结构：

```json
{
  "format_version": 1,
  "session": {
    "scale_exponent": 3,
    "seed": 20261125,
    "sample_count": 48,
    "group_size": 8,
    "noise_bound": 9,
    "global_concurrency": 2,
    "per_environment_concurrency": 1,
    "handshake_timeout_ms": 1500
  },
  "environments": [
    {
      "id": "cn-prod",
      "host": "127.0.0.1",
      "port": 46301,
      "weight": "1.0",
      "slices": [{"version": 2, "values": {"replicas": "3"}}]
    }
  ],
  "views": [
    {
      "id": "global-release",
      "environments": ["cn-prod"],
      "keys": {"replicas": {"mode": "latest", "tolerance": 0}}
    }
  ]
}
```

`mode` 取 `latest`、`weighted_majority`、`interval_midpoint`。容差以会话缩放后的整数单位表示。

## 错误码

- `E_SPEC`：声明非法，`field` 定位首个非法字段。
- `E_PORT`：端口占用、绑定或连接失败。
- `E_TIMEOUT`：服务就绪或套接字握手真实超时。
- `E_FIELD`：回执缺字段或字段不合法。
- `E_REPLAY`：重放不一致，上下文给出首个 `key` 与 `field`。
- `E_TK`：当前环境缺少 Tk 或图形显示。

详细设计、公式和取舍见 `docs/TECHNICAL.md`。
