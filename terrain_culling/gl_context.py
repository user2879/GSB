"""Offscreen GL context probe.

The deterministic reference pipeline is pure NumPy and needs no GPU. This probe
verifies that an offscreen EGL/GLES context *can* be initialized on the host and
records the result, satisfying the pre-flight check without coupling reproducible
outputs to a particular driver.
"""
from __future__ import annotations

import ctypes


def probe_egl(width: int = 64, height: int = 64) -> dict:
    try:
        from OpenGL import EGL as egl
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"available": False, "backend": "egl", "error": repr(exc)}

    try:
        display = egl.eglGetDisplay(egl.EGL_DEFAULT_DISPLAY)
        if not display:
            return {"available": False, "backend": "egl",
                    "error": f"eglGetDisplay {egl.eglGetError()}"}
        major, minor = ctypes.c_long(), ctypes.c_long()
        if not egl.eglInitialize(display, ctypes.byref(major), ctypes.byref(minor)):
            return {"available": False, "backend": "egl",
                    "error": f"eglInitialize {egl.eglGetError()}"}
        attribs = (egl.EGLint * 13)(
            egl.EGL_SURFACE_TYPE, egl.EGL_PBUFFER_BIT,
            egl.EGL_RED_SIZE, 8, egl.EGL_GREEN_SIZE, 8, egl.EGL_BLUE_SIZE, 8,
            egl.EGL_DEPTH_SIZE, 24,
            egl.EGL_RENDERABLE_TYPE, egl.EGL_OPENGL_ES2_BIT, egl.EGL_NONE)
        count = egl.EGLint()
        configs = (egl.EGLConfig * 4)()
        egl.eglChooseConfig(display, attribs, configs, 4, ctypes.byref(count))
        if count.value == 0:
            return {"available": False, "backend": "egl", "error": "no config"}
        pb_attr = (egl.EGLint * 5)(egl.EGL_WIDTH, width,
                                   egl.EGL_HEIGHT, height, egl.EGL_NONE)
        surface = egl.eglCreatePbufferSurface(display, configs[0], pb_attr)
        ctx_attr = (egl.EGLint * 3)(0x3098, 2, egl.EGL_NONE)
        context = egl.eglCreateContext(display, configs[0],
                                       egl.EGL_NO_CONTEXT, ctx_attr)
        current = egl.eglMakeCurrent(display, surface, surface, context)
        if not current:
            return {"available": False, "backend": "egl",
                    "error": f"makeCurrent {egl.eglGetError()}"}
        result = {
            "available": True,
            "backend": "egl_gles2",
            "egl_version": f"{major.value}.{minor.value}",
            "pbuffer": [width, height],
        }
        egl.eglMakeCurrent(display, egl.EGL_NO_SURFACE, egl.EGL_NO_SURFACE,
                           egl.EGL_NO_CONTEXT)
        egl.eglDestroyContext(display, context)
        egl.eglDestroySurface(display, surface)
        return result
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"available": False, "backend": "egl", "error": repr(exc)}
