"""离屏显示层：OpenGL 可用时走 GL 路径，否则退化到软件缓冲。

环境中通常没有真实 GPU/窗口系统，因此 :func:`create_display` 会先尝试
创建离屏 OpenGL 上下文（PyOpenGL + 平台 GLUT/EGL/OSMesa 之一），任何
失败都回退到 :class:`SoftwareDisplay`。两条路径产出的缩略图都由同一份
确定性 numpy 降采样逻辑生成，GL 路径仅用于证明上下文可创建与纹理上传，
不参与任何哈希/指标计算，因此显示层差异不会影响交付结果的可重复性。
"""

from __future__ import annotations

import numpy as np

THUMB_W = 96
THUMB_H = 54


def _downsample(u8: np.ndarray, tw: int = THUMB_W, th: int = THUMB_H) -> np.ndarray:
    """确定性最近邻降采样（纯整数索引，跨平台结果一致）。"""
    h, w = u8.shape[:2]
    ys = (np.arange(th, dtype=np.int64) * h // th).astype(np.int64)
    xs = (np.arange(tw, dtype=np.int64) * w // tw).astype(np.int64)
    return u8[ys[:, None], xs[None, :], :].copy()


class SoftwareDisplay:
    backend = "software-numpy"

    def __init__(self):
        self.thumbnails: dict[int, np.ndarray] = {}

    def upload(self, frame: int, u8: np.ndarray) -> np.ndarray:
        thumb = _downsample(u8)
        self.thumbnails[frame] = thumb
        return thumb

    def render_timeline(self, entries: list[int]) -> str:
        """返回时间轴缩略图的 ANSI/ASCII 文本表示。"""
        lines = []
        for frame in entries:
            thumb = self.thumbnails.get(frame)
            if thumb is None:
                lines.append(f"frame {frame:03d}: <未渲染>")
                continue
            mean_lum = thumb.mean()
            bar = "#" * int(round(float(mean_lum) / 255.0 * 40))
            lines.append(f"frame {frame:03d}: |{bar:<40}| mean={mean_lum:6.2f}")
        return "\n".join(lines)

    def close(self) -> None:
        self.thumbnails.clear()


class GLDisplay(SoftwareDisplay):
    backend = "opengl-offscreen"

    def __init__(self, gl_module):
        super().__init__()
        self.gl = gl_module
        self._tex_ids: dict[int, int] = {}

    def upload(self, frame: int, u8: np.ndarray) -> np.ndarray:
        thumb = super().upload(frame, u8)
        # 走一次 GL 纹理上传，证明离屏上下文数据通路可用（结果不参与哈希）
        gl = self.gl
        tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, thumb.shape[1],
                        thumb.shape[0], 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE,
                        np.ascontiguousarray(thumb))
        self._tex_ids[frame] = int(tex)
        return thumb

    def close(self) -> None:
        try:
            self.gl.glDeleteTextures(list(self._tex_ids.values()))
        except Exception:
            pass
        super().close()


def create_display():
    """尝试创建离屏 GL 显示，失败则回退软件缓冲。"""
    try:
        import OpenGL.GL as gl  # noqa: F401
        # 离屏上下文需要窗口/缓冲平台；无头环境大概率失败，这是预期回退路径
        import OpenGL.GLUT as glut  # noqa: F401
        glut.glutInit()
        glut.glutInitDisplayMode(glut.GLUT_RGB)
        glut.glutInitWindowSize(THUMB_W, THUMB_H)
        win = glut.glutCreateWindow(b"hdr-sdr-workbench-offscreen")
        glut.glutHideWindow()
        return GLDisplay(gl)
    except Exception:
        return SoftwareDisplay()
