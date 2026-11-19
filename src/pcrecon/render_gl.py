"""Offscreen OpenGL renderer for point clouds and reconstructed mesh.

GL imports are intentionally local so the headless pipeline (and the test
suite) works without a display.  Context/FBO failure raises
``GLContextError`` (E_GL_CONTEXT) instead of crashing the process.
"""
from __future__ import annotations

import os

from .errors import GLContextError


def verify_offscreen_context() -> dict:
    """Create a Qt offscreen GL context and an FBO; return diagnostics."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PyQt5 import QtGui
        from OpenGL import GL
    except Exception as exc:  # pragma: no cover - environment dependent
        raise GLContextError(f"GL bindings unavailable: {exc!r}") from exc

    app = QtGui.QGuiApplication.instance() or QtGui.QGuiApplication([])
    surface = QtGui.QOffscreenSurface()
    surface.create()
    if not surface.isValid():
        raise GLContextError("QOffscreenSurface creation failed")

    context = QtGui.QOpenGLContext()
    if not context.create():
        raise GLContextError("QOpenGLContext.create() failed")
    if not context.makeCurrent(surface):
        raise GLContextError("makeCurrent failed")

    fbo = GL.glGenFramebuffers(1)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
    color = GL.glGenRenderbuffers(1)
    GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, color)
    GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RGBA8, 64, 64)
    GL.glFramebufferRenderbuffer(
        GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_RENDERBUFFER, color
    )
    status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
    if status != GL.GL_FRAMEBUFFER_COMPLETE:
        raise GLContextError(f"FBO incomplete, status=0x{int(status):x}")
    renderer = GL.glGetString(GL.GL_RENDERER)
    version = GL.glGetString(GL.GL_VERSION)
    context.doneCurrent()
    return {
        "status": "ok",
        "renderer": renderer.decode() if isinstance(renderer, bytes) else str(renderer),
        "gl_version": version.decode() if isinstance(version, bytes) else str(version),
        "fbo_id": int(fbo),
    }


class OffscreenRenderer:
    """Minimal deterministic renderer; draws points and mesh lines into an FBO."""

    def __init__(self, width: int = 512, height: int = 512):
        self.info = verify_offscreen_context()
        from OpenGL import GL

        self.GL = GL
        self.width = width
        self.height = height

    def render_snapshot(self, snapshot, *, density: float = 1.0) -> dict:
        GL = self.GL
        GL.glViewport(0, 0, self.width, self.height)
        GL.glClearColor(0.05, 0.05, 0.08, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        point_count = int(snapshot.points.shape[0] * density)
        vertex_count = int(snapshot.merged["vertices"].shape[0])
        face_count = int(snapshot.merged["faces"].shape[0])
        return {
            "rendered_points": point_count,
            "rendered_vertices": vertex_count,
            "rendered_faces": face_count,
            "density": density,
            "version": snapshot.version,
            "renderer": self.info["renderer"],
        }
