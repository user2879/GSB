"""Offscreen graphics-context probe with a deterministic software fallback."""

from __future__ import annotations

import ctypes
import platform
import re
from pathlib import Path
from typing import Any


def _exception_text(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    text = re.sub(r"0x[0-9a-fA-F]+", "<addr>", text)
    return text[:600]


def _ok(report: dict[str, Any]) -> dict[str, Any]:
    report["context_created"] = True
    report["software_fallback"] = False
    report["reason"] = ""
    return report


def _failed(report: dict[str, Any], reason: str) -> dict[str, Any]:
    report["context_created"] = False
    report["software_fallback"] = True
    report["reason"] = reason
    return report


def _probe_egl(report: dict[str, Any]) -> dict[str, Any]:
    try:
        from OpenGL import EGL, GL
        from OpenGL.raw.EGL._types import EGLConfig, EGLint
    except Exception as exc:  # pragma: no cover - platform dependent
        return _failed(report, f"PyOpenGL EGL import failed: {type(exc).__name__}: {_exception_text(exc)}")

    display = None
    context = None
    surface = None
    try:  # pragma: no cover - platform/GPU dependent
        display = EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)
        if display == EGL.EGL_NO_DISPLAY:
            return _failed(report, "eglGetDisplay returned EGL_NO_DISPLAY")
        initialized = EGL.eglInitialize(display, None, None)
        if not initialized:
            return _failed(report, "eglInitialize failed")
        version_text = EGL.eglQueryString(display, EGL.EGL_VERSION)
        attributes = [
            EGL.EGL_SURFACE_TYPE,
            EGL.EGL_PBUFFER_BIT,
            EGL.EGL_RENDERABLE_TYPE,
            EGL.EGL_OPENGL_BIT,
            EGL.EGL_RED_SIZE,
            8,
            EGL.EGL_GREEN_SIZE,
            8,
            EGL.EGL_BLUE_SIZE,
            8,
            EGL.EGL_ALPHA_SIZE,
            8,
            EGL.EGL_NONE,
        ]
        configs = (EGLConfig * 1)()
        config_count = (EGLint * 1)()
        chosen = EGL.eglChooseConfig(display, attributes, configs, 1, config_count)
        if not chosen or config_count[0] < 1:
            return _failed(report, "eglChooseConfig returned no OpenGL pbuffer config")
        config = configs[0]
        context = EGL.eglCreateContext(display, config, EGL.EGL_NO_CONTEXT, [EGL.EGL_NONE])
        if context == EGL.EGL_NO_CONTEXT:
            return _failed(report, "eglCreateContext failed")
        surface = EGL.eglCreatePbufferSurface(
            display,
            config,
            [EGL.EGL_WIDTH, 16, EGL.EGL_HEIGHT, 16, EGL.EGL_NONE],
        )
        if surface == EGL.EGL_NO_SURFACE:
            return _failed(report, "eglCreatePbufferSurface failed")
        if not EGL.eglMakeCurrent(display, surface, surface, context):
            return _failed(report, "eglMakeCurrent failed")
        renderer = GL.glGetString(GL.GL_RENDERER)
        vendor = GL.glGetString(GL.GL_VENDOR)
        version = GL.glGetString(GL.GL_VERSION)
        report.update(
            {
                "api": "EGL/OpenGL",
                "egl_version": version_text.decode("utf-8", "replace") if version_text else "",
                "renderer": renderer.decode("utf-8", "replace") if renderer else "",
                "vendor": vendor.decode("utf-8", "replace") if vendor else "",
                "version": version.decode("utf-8", "replace") if version else "",
                "surface": "pbuffer:16x16",
            }
        )
        return _ok(report)
    except Exception as exc:  # pragma: no cover - platform/GPU dependent
        return _failed(report, f"EGL probe exception: {type(exc).__name__}: {_exception_text(exc)}")
    finally:  # pragma: no cover - platform/GPU dependent
        try:
            if display is not None and context is not None and surface is not None:
                EGL.eglMakeCurrent(display, EGL.EGL_NO_SURFACE, EGL.EGL_NO_SURFACE, EGL.EGL_NO_CONTEXT)
                EGL.eglDestroySurface(display, surface)
                EGL.eglDestroyContext(display, context)
                EGL.eglTerminate(display)
        except Exception:
            pass


def _probe_wgl(report: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover - Windows only
    try:
        from ctypes import wintypes
    except Exception as exc:
        return _failed(report, f"ctypes Windows types unavailable: {exc}")

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
    opengl32 = ctypes.WinDLL("opengl32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    WNDPROC = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)

    class WNDCLASSW(ctypes.Structure):
        _fields_ = [
            ("style", wintypes.UINT),
            ("lpfnWndProc", WNDPROC),
            ("cbClsExtra", ctypes.c_int),
            ("cbWndExtra", ctypes.c_int),
            ("hInstance", wintypes.HINSTANCE),
            ("hIcon", wintypes.HICON),
            ("hCursor", wintypes.HANDLE),
            ("hbrBackground", wintypes.HBRUSH),
            ("lpszMenuName", wintypes.LPCWSTR),
            ("lpszClassName", wintypes.LPCWSTR),
        ]

    class PIXELFORMATDESCRIPTOR(ctypes.Structure):
        _fields_ = [
            ("nSize", wintypes.WORD),
            ("nVersion", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("iPixelType", wintypes.BYTE),
            ("cColorBits", wintypes.BYTE),
            ("cRedBits", wintypes.BYTE),
            ("cRedShift", wintypes.BYTE),
            ("cGreenBits", wintypes.BYTE),
            ("cGreenShift", wintypes.BYTE),
            ("cBlueBits", wintypes.BYTE),
            ("cBlueShift", wintypes.BYTE),
            ("cAlphaBits", wintypes.BYTE),
            ("cAlphaShift", wintypes.BYTE),
            ("cAccumBits", wintypes.BYTE),
            ("cAccumRedBits", wintypes.BYTE),
            ("cAccumGreenBits", wintypes.BYTE),
            ("cAccumBlueBits", wintypes.BYTE),
            ("cAccumAlphaBits", wintypes.BYTE),
            ("cDepthBits", wintypes.BYTE),
            ("cStencilBits", wintypes.BYTE),
            ("cAuxBuffers", wintypes.BYTE),
            ("iLayerType", wintypes.BYTE),
            ("bReserved", wintypes.BYTE),
            ("dwLayerMask", wintypes.DWORD),
            ("dwVisibleMask", wintypes.DWORD),
            ("dwDamageMask", wintypes.DWORD),
        ]

    hwnd = None
    hdc = None
    hglrc = None
    atom = 0
    try:
        def window_proc(hwnd_value, msg, wparam, lparam):
            return user32.DefWindowProcW(hwnd_value, msg, wparam, lparam)

        proc = WNDPROC(window_proc)
        kernel32.GetModuleHandleW.restype = wintypes.HMODULE
        instance = kernel32.GetModuleHandleW(None)
        class_name = "VolumetricEditorOffscreenProbeV1"
        wc = WNDCLASSW()
        wc.style = 0
        wc.lpfnWndProc = proc
        wc.hInstance = instance
        wc.lpszClassName = class_name
        atom = user32.RegisterClassW(ctypes.byref(wc))
        if not atom:
            return _failed(report, f"RegisterClassW failed: {ctypes.get_last_error()}")
        user32.CreateWindowExW.restype = wintypes.HWND
        hwnd = user32.CreateWindowExW(
            0,
            class_name,
            "offscreen-probe",
            0,
            0,
            0,
            1,
            1,
            None,
            None,
            instance,
            None,
        )
        if not hwnd:
            return _failed(report, f"CreateWindowExW failed: {ctypes.get_last_error()}")
        user32.GetDC.restype = wintypes.HDC
        hdc = user32.GetDC(hwnd)
        if not hdc:
            return _failed(report, f"GetDC failed: {ctypes.get_last_error()}")
        pfd = PIXELFORMATDESCRIPTOR()
        pfd.nSize = ctypes.sizeof(PIXELFORMATDESCRIPTOR)
        pfd.nVersion = 1
        pfd.dwFlags = 0x00000004 | 0x00000020 | 0x00000001  # SUPPORT_OPENGL | DRAW_TO_WINDOW | DOUBLEBUFFER
        pfd.iPixelType = 0  # PFD_TYPE_RGBA
        pfd.cColorBits = 32
        pfd.cAlphaBits = 8
        pfd.cDepthBits = 24
        pixel_format = gdi32.ChoosePixelFormat(hdc, ctypes.byref(pfd))
        if not pixel_format:
            return _failed(report, f"ChoosePixelFormat failed: {ctypes.get_last_error()}")
        if not gdi32.SetPixelFormat(hdc, pixel_format, ctypes.byref(pfd)):
            return _failed(report, f"SetPixelFormat failed: {ctypes.get_last_error()}")
        opengl32.wglCreateContext.restype = wintypes.HANDLE
        hglrc = opengl32.wglCreateContext(hdc)
        if not hglrc:
            return _failed(report, f"wglCreateContext failed: {ctypes.get_last_error()}")
        if not opengl32.wglMakeCurrent(hdc, hglrc):
            return _failed(report, f"wglMakeCurrent failed: {ctypes.get_last_error()}")
        opengl32.glGetString.restype = ctypes.c_char_p
        opengl32.glGetString.argtypes = [ctypes.c_uint]
        renderer = opengl32.glGetString(0x1F01) or b""
        vendor = opengl32.glGetString(0x1F00) or b""
        version = opengl32.glGetString(0x1F02) or b""
        report.update(
            {
                "api": "WGL/OpenGL",
                "renderer": renderer.decode("utf-8", "replace"),
                "vendor": vendor.decode("utf-8", "replace"),
                "version": version.decode("utf-8", "replace"),
                "surface": "hidden-window:1x1",
            }
        )
        return _ok(report)
    except Exception as exc:
        return _failed(report, f"WGL probe exception: {type(exc).__name__}: {_exception_text(exc)}")
    finally:
        try:
            if hglrc:
                opengl32.wglMakeCurrent(None, None)
                opengl32.wglDeleteContext(hglrc)
            if hdc and hwnd:
                user32.ReleaseDC(hwnd, hdc)
            if hwnd:
                user32.DestroyWindow(hwnd)
            if atom:
                user32.UnregisterClassW(class_name, instance)
        except Exception:
            pass


def probe_offscreen(backend: str = "auto") -> dict[str, Any]:
    system = platform.system()
    report: dict[str, Any] = {
        "requested_backend": backend,
        "platform": system,
        "python": platform.python_version(),
        "context_created": False,
        "software_fallback": False,
        "render_mode": "software-offscreen-float32",
        "api": "software",
        "renderer": "numpy-deterministic-software",
        "vendor": "local",
        "version": "1.0",
        "surface": "in-memory-tile-framebuffer",
        "reason": "",
    }
    if backend == "software":
        return _failed(report, "software backend selected explicitly")
    if backend not in {"auto", "opengl"}:
        return _failed(report, f"unknown backend selection: {backend}")
    if system == "Windows":
        result = _probe_wgl(report)
    else:
        result = _probe_egl(report)
    if not result["context_created"]:
        result["render_mode"] = "software-offscreen-float32"
        result["api"] = "software"
    return result


def write_backend_log(report: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "event=backend_probe",
        f"platform={report.get('platform', '')}",
        f"requested_backend={report.get('requested_backend', '')}",
        f"context_created={str(report.get('context_created', False)).lower()}",
        f"software_fallback={str(report.get('software_fallback', True)).lower()}",
        f"api={report.get('api', '')}",
        f"renderer={report.get('renderer', '')}",
        f"vendor={report.get('vendor', '')}",
        f"version={report.get('version', '')}",
        f"surface={report.get('surface', '')}",
        f"render_mode={report.get('render_mode', '')}",
        f"reason={report.get('reason', '')}",
    ]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
