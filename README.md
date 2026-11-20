# cfgdist — 跨组件平台配置分发与故障恢复系统

面向 Windows 本地环境的配置分发原型：管理若干回环 TCP 模拟节点，支持清单
v1/v2/v3 三种历史格式的强校验与逐条升级、并行分发与回执收敛、状态机幂等提交、
快照整体替换恢复，以及独立的 SQLite 重放一致性校验。

- 设计与业务码、验证结果见 `docs/ARCHITECTURE.md`
- 样本清单：`samples/manifests/`（v1/v2/v3），节点声明：`samples/nodes.json`
- 运行期目录：`run/`（db / snapshots / checkpoints / backups / logs）

## 快速开始

```powershell
pip install -r requirements.txt
python -m pytest -q
python -m cfgdist.cli distribute samples/manifests/manifest_v3.json `
  --nodes samples/nodes.json --workdir run --max-concurrency 2
```

业务码：`E_FORMAT`(10) `E_PORT`(11) `E_VERSION`(12) `E_CONVERGE`(13)
`E_IDEMPOTENT`(14) `E_REPLAY`(15) `E_STATE`(16) `E_PROTO`(17) `E_NODE`(18)。
