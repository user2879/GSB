# 配置中心租约分发与变更预算裁决系统

面向 Windows 本地环境的配置中心：以独立子进程运行配置消费节点，基于 127.0.0.1
回环 TCP 的 JSON 行协议通信，围绕配置修订签发带 TTL 与修订序号区间的租约，
在全局下发预算与每节点修订预算下并行裁决，同键冲突按稳定规则串行化，并把
修订、租约、游标、预算、事件与快照持久化到 SQLite，支持强制终止恢复与按记录重放。

## 目录结构

```
configlease/
  errors.py         E_* 结构化错误码
  clock.py          单调毫秒时钟（租约 TTL）
  protocol.py       JSON 行 TCP 协议
  sample.py         NumPy 确定性样本与聚合摘要
  manifest.py       修订清单加载/校验/视图折叠
  storage.py        SQLite 持久化（六表，事务原子提交）
  structured_log.py JSON 行 + SQLite 双写结构化日志
  engine.py         校验规则、双重预算、稳定串行化、快照与重放
  node.py           独立子进程节点（回环 TCP 服务端）
  platform.py       平台：编排/裁决/回滚/恢复/快照
  cli.py            命令行与三类异常注入
tests/              26+ 项 unittest（规则/并发/进程/存储/重放）
scripts/
  generate_data.py  生成本地确定性样本数据
  acceptance.py     五大验收场景端到端驱动（含真实 kill 进程组）
docs/
  ARCHITECTURE.md   架构与设计说明
  WINDOWS_RUNBOOK.md Windows 运行手册
```

## 快速开始

```bash
pip install -r requirements.txt
python scripts/generate_data.py --out runtime/data
python -m configlease serve --workdir runtime/run \
  --manifest runtime/data/manifest.json \
  --nodes runtime/data/nodes.json \
  --requests runtime/data/requests.json
python -m configlease reopen --workdir runtime/run --nodes runtime/data/nodes.json
```

## 验收与测试

```bash
python -m unittest discover -s tests -v
python scripts/acceptance.py --workdir runtime/acceptance --data runtime/data
```

`acceptance.py` 覆盖：租约边界（E_REV/E_KEY/游标不推进/摘要一致）、双重预算
（FIFO 排队、不可绕过、提额放行）、回滚撤销（E_ROLLBACK + 视图整体替换）、
强制终止恢复（真实 SIGKILL 进程组、unknown 租约 E_RECOVER、拒绝退出报告）、
快照版本识别与重放校验（E_VERSION 整体替换、E_REPLAY 字段级比对）。

异常注入：`--inject-manifest`(E_MANIFEST)、`--inject-port`(E_PORT)、
`--inject-version`(E_VERSION)，仅改输入文件或参数，不改源码。

详见 `docs/ARCHITECTURE.md` 与 `docs/WINDOWS_RUNBOOK.md`。
