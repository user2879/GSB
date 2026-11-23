"""跨模态图文检索诊断工作台 (retrieval bench)。

所有模块均围绕"确定性、可审计、可增量复现"设计：
- 固定随机种子与固定浮点累加顺序；
- 阶段产物均带内容/参数/量化档位派生的缓存键；
- 仅依赖标准库、NumPy 与 Pillow。
"""

__all__ = ["errors", "fingerprint", "fixedpoint", "miniyaml"]
