# 列式存储压缩与扫描系统（Windows 本地）

本地确定性的列式段存储、压缩编码、谓词下推扫描、块级并行调度与会话快照
应用。纯 Python 3.11+，依赖 NumPy；Tkinter 由官方 Python 安装包自带。
不使用容器，不访问网络或真实账号，所有文件位于项目目录内。

## 快速开始

```powershell
py -3.11 -m pip install -r requirements.txt
py -3.11 run_all.py          # 一键验收（样本/测试/三场景/异常码/一致性）
py -3.11 -m colstore.ui      # Tkinter 桌面界面（也可双击 run_gui.bat）
```

Linux/macOS 等价命令：`python3 run_all.py`。

## 命令行

```bash
python -m colstore.cli prepare
python -m colstore.cli scan --query query_and.json --parallelism 4 \
    --memory-budget 65536 --segment-concurrency 2
python -m colstore.cli check --query query_eq.json --parallelism 4
python -m colstore.cli estimate --segment sales --row-group 0 --column region_id
python -m colstore.cli inject --kind payload_truncation   # E_PAYLOAD
python -m colstore.cli inject --kind encoding_mismatch    # E_ENCODING
python -m colstore.cli inject --kind zero_budget          # E_BUDGET
python -m colstore.cli snapshot save --id s1
python -m colstore.cli snapshot open --id s1
python -m colstore.cli snapshot list
```

## 业务码

| 码 | 触发场景 |
|---|---|
| `E_SEGMENT` | 段结构/CRC/行数等整体损坏，整段拒绝加载 |
| `E_PAYLOAD` | 块载荷截断或长度与块头不自洽 |
| `E_ENCODING` | 编码标记与载荷结构不匹配、编码不受支持 |
| `E_QUERY` | 不支持的算子或非法查询 |
| `E_BUDGET` | 零/负预算，或单块内存超过全局预算 |
| `E_DIVERGE` | 跨并行度/跨编码结果或统计不一致 |
| `E_SNAPSHOT` | 快照写一次冲突、只读快照被改写等 |

## 产物路径

- `colstore/sample_data/`：确定性段定义、5 个查询、`fault_params.json`、旧版本快照
- `colstore/data/`：`.colseg` 段文件、`sessions.db`、JSONL 日志、演示总结
- `colstore/data/demo_summary.md` / `demo_summary.json`：三场景验证总结
- `colstore/data/run_all_report.json`：一键验收报告

设计与文件格式、需求映射见 [DESIGN.md](DESIGN.md)。
