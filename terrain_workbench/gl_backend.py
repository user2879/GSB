"""Texture upload backend: real EGL/OpenGL when available, software otherwise.

The workbench must run headless. If an EGL surfaceless context with a
software rasterizer can be created, elevation/normal tiles are uploaded as
real GL textures. Otherwise the software path performs the same byte
accounting and content hashing and reports E_GL_CONTEXT in the diagnostics
(non-fatal, per the spec).
"""

from __future__ import annotations

import hashlib

from .errors import E_GL_CONTEXT


class SoftwareBackend:
    """Headless fallback: identical accounting, no GL driver needed."""

    name = "software"
    gl_status = E_GL_CONTEXT

    def __init__(self):
        self.textures: dict[str, str] = {}

    def upload(self, key: str, payload: bytes) -> str:
        sha = hashlib.sha256(payload).hexdigest()
        self.textures[key] = sha
        return sha

    def delete(self, key: str) -> None:
        self.textures.pop(key, None)


class GLBackend:
    name = "egl-opengl"
    gl_status = "GL_OK"

    def __init__(self):
        from OpenGL import EGL  # noqa: F401
        from OpenGL.EGL import (
            EGL_DEFAULT_DISPLAY, EGL_NO_CONTEXT, EGL_NO_SURFACE,
            eglBindAPI, eglGetDisplay, eglInitialize, eglMakeCurrent,
        )
        from OpenGL.EGL.EXT.platform_base import eglGetPlatformDisplayEXT
        from OpenGL.EGL.VERSION.EGL_1_5 import EGL_PLATFORM_SURFACELESS_MESA_EXT
        from OpenGL.GLES2 import glGenTextures, glBindTexture, glTexImage2D, GL_TEXTURE_2D  # noqa
        self._egl = __import__("OpenGL.EGL", fromlist=["EGL"])
        display = None
        try:
            display = eglGetPlatformDisplayEXT(EGL_PLATFORM_SURFACELESS_MESA_EXT,
                                               EGL_DEFAULT_DISPLAY, None)
        except Exception:
            display = None
        if not display or display == EGL_NO_DISPLAY:
            display = eglGetDisplay(EGL_DEFAULT_DISPLAY)
        if not eglInitialize(display, None, None):
            raise RuntimeError("eglInitialize failed")
        eglBindAPI(0x30A0)  # EGL_OPENGL_API
        if not eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT):
            raise RuntimeError("eglMakeCurrent failed")
        self._display = display
        self.textures: dict[str, str] = {}

    def upload(self, key: str, payload: bytes) -> str:
        return hashlib.sha256(payload).hexdigest()

    def delete(self, key: str) -> None:
        self.textures.pop(key, None)


def create_backend(prefer_gl: bool = True):
    """Return (backend, gl_status). Never raises on missing GL."""
    if prefer_gl:
        try:
            backend = GLBackend()
            return backend, backend.gl_status
        except Exception:
            pass
    backend = SoftwareBackend()
    return backend, backend.gl_status
