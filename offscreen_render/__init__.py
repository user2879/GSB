"""本地离屏多介质透射渲染引擎。

纯 Python + NumPy 实现，确定性、可离线复现，不依赖任何外部资源。
模块划分：

* :mod:`offscreen_render.scene`       场景解析与输入校验（E_xxx 业务码）
* :mod:`offscreen_render.geometry`    光线-AABB 求交与有序区间链
* :mod:`offscreen_render.precision`   单精度尾数量化
* :mod:`offscreen_render.renderer`    指数透射率、散射、双/单精度误差
* :mod:`offscreen_render.scheduler`   分块并发、内存上限、中断边界
* :mod:`offscreen_render.persistence` 原子持久化、版本识别、归档恢复
* :mod:`offscreen_render.assembly`    行序拼装、PGM 原子写出、报告
* :mod:`offscreen_render.stability`   步长/阈值抖动的稳定性量化
* :mod:`offscreen_render.samplegen`   固定种子确定性样本
* :mod:`offscreen_render.pipeline`    端到端编排
"""

__version__ = "1.0.0"
