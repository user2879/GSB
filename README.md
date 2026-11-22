# colval — 列式向量化执行与格式互操作验证平台

面向本地 Windows 环境（Python 3.10+，依赖仅 numpy + 标准库）的数据处理管线
与结果验证工具。全部验证在本地完成，无需外部服务、账号或网络。

## 功能总览
- b1/b2/b3 三版本列式块编解码与强校验（`E_FORMAT`），SQLite 持久化；
- 低版本块自动升级 + 检查点断点续做，旧块只读备份；
- 升级块与直接 b3 写入块的逐字节一致性校验（`E_DIVERGE`）；
- 表达式解析/规范化/批切分计划编译（`E_PLAN`）；
- 统一精度定点量化（10 的整数幂），显式截断/饱和规则与误差界；
- 全局内存预算 + 每任务并发配额双约束批调度，单调时钟计时；
- 批内向量化求值与确定性合流（任意批大小/顺序结果逐字节一致）；
- 执行账本 + 会话快照版本识别整体替换恢复，重放校验（`E_REPLAY`）。

## 快速开始
```bash
pip install numpy            # 唯一第三方依赖
python -m colval.cli init-sample --base .   # 生成确定性样本（种子 20261123）
python -m colval.cli run         --base .   # 升级 + 并行执行 + 账本 + 快照
python -m colval.cli run --base . --crash-after 7   # 模拟强制终止
python -m colval.cli run         --base .   # 依据账本恢复并重放剩余批
python -m colval.cli verify      --base .   # 升级块 vs 直接 b3 逐字节校验
python -m colval.cli gui         --base .   # Tkinter 界面（Windows 自带 Tk）
python -m pytest tests/ -q                  # 41 项测试
```

## 目录
- `colval/` 平台源码（见 docs/TECHNICAL_NOTES.md 的模块说明）
- `tests/` 41 项单元/端到端测试
- `tasks/expressions.json` 表达式任务清单与批切分参数
- `data/` SQLite 块库 + 升级检查点；`work/` 执行账本；`canonical.json` 基准 b3
- `docs/` 技术说明与实际验证结果
