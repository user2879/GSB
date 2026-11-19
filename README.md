# 中文分词与词性标注增量组件（segpos）

一个可作为库集成的中文分词 + 词性标注领域组件。核心解决**词典与概率表增量更新时的依赖失效判定**：每次编辑后只重算真正受影响的句子，未失效句子直接复用缓存，且增量结果与“从空白工作区按最终输入全量运行”逐字段一致。

## 环境要求

- Python 3.10 及以上（开发验证使用 3.11）
- NumPy 1.24 及以上（开发验证使用 2.4.6）
- 不依赖 Docker，不访问网络目标；所有样本均为本地确定性数据

### Windows 安装

```powershell
py -3 -m pip install --user "numpy>=1.24"
py -3 -m unittest discover -s tests
```

### Linux/macOS 安装

```bash
python3 -m pip install --user -r requirements.txt
python3 -m unittest discover -s tests
```

## 输入文件（UTF-8）

位于 `examples/sample_data/`，由 `examples/generate_sample_data.py` 确定性生成：

- `sentences.txt`：每行 `句子ID<TAB>文本`，ID 唯一且文本非空；包含两条故意非法行用于失败清单验证。
- `dictionary.txt`：每行 `词 词频 词性`，词频为正整数，词性属于预定义集合 `{n,v,a,d}`。
- `prob_table.json`：`tags`、`start_logprob`、`trans_logprob`、`emit_logprob`、`char_emit_logprob`，全部概率以自然对数存储；缺失发射/转移回退到配置平滑值。
- `config.json`：`workers`、`dict_change_mode`、`snapshot_commit_interval`、`low_confidence_threshold`、`smoothing`。

## 快速开始

```bash
# 生成确定性样本
python3 examples/generate_sample_data.py

# 单次流水线（首次全量，再次运行即续跑复用）
python3 examples/run_pipeline.py \
  --sample-dir examples/sample_data \
  --snapshot-dir examples/output/cli/snapshot \
  --report examples/output/cli/report.json

# 编辑序列增量 vs 空白全量逐字段对比（核心验收脚本）
python3 examples/run_demo.py
```

## 组件结构

```
segpos/
  config.py        增量配置加载与校验（并发数必须为正整数等）
  errors.py        E_SENTENCE_INVALID / E_CONFIG_MISMATCH / E_STAGE_FAILED
  fingerprints.py  SHA-256 内容指纹、缓存键、稳定排序键
  models.py        不可变词典、概率表（NumPy 矩阵）、句子/词数据模型
  segmenter.py     正向最大匹配（同长：词频降序 -> 词形字典序）
  hmm.py           对数域通用维特比（NumPy，含后验边际置信度）
  tagger.py        OOV 字符级 HMM 识别 + 词级词性维特比标注
  cache.py         内容指纹缓存 + 词条倒排索引 + 持久隔离集合
  snapshot.py      临时目录暂存 + os.replace 原子替换 + manifest
  pipeline.py      线程池编排、增量编辑、信号退出、报告、异常隔离
examples/          确定性样本、CLI、增量/全量对比演示
tests/             16 个 unittest 用例
```

## 关键设计

- **缓存键**：`SHA256(文本指纹, 命中词条集合指纹, 概率表版本)`。词典增删/词频调整经倒排索引（新增词额外做文本子串扫描）只失效命中句；概率表替换使全句失效。
- **快照一致性**：每个结果文件先写同目录临时文件再 `os.replace`，最后原子替换 `manifest.json` 作为提交点；强制终止后只依据已完整替换的快照恢复。
- **并发确定性**：线程池并发计算，但结果按 `稳定排序键(句子ID, 阶段名)` 收集落盘，`workers=1/2/4/8` 输出逐字段一致。
- **异常隔离**：分词/OOV/标注三阶段独立捕获，失败句进入持久隔离集合，后续编辑（除非显式替换该句文本）不再产出结果。

详见 `docs/design.md` 与 `docs/verification.md`。
