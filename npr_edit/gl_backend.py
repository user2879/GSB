"""Optional hidden OpenGL FBO output.

The deterministic software rasteriser is the source of truth.  When a local
offscreen context is available, the BGRA buffer is passed through an invisible
FBO before snapshot publication.  Failure is always reported as E_GL_CONTEXT.
"""

from __future__ import annotations

import ctypes
import os
from typing import Any

from .errors import E_GL_CONTEXT, NPRError


class OffscreenGL:
    def __init__(self, platform_backend: Any) -> None:
        self.platform_backend = platform_backend
        self._initialised = False
        self._gl: Any = None
        self._fbo: int | None = None
        self._texture: int | None = None
        self._width = 0
        self._height = 0

    def _ensure_fbo(self, width: int, height: int) -> None:
        try:
            from OpenGL import GL
        except Exception as exc:  # pragma: no cover - depends on local OpenGL
            raise NPRError(E_GL_CONTEXT, "PyOpenGL could not be imported after context creation") from exc
        self._gl = GL
        if self._initialised and self._width == width and self._height == height:
            return
        if self._fbo is not None:
            GL.glDeleteFramebuffers(1, [self._fbo])
            GL.glDeleteTextures([self._texture])
        texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, width, height, 0, GL.GL_BGRA, GL.GL_UNSIGNED_BYTE, None)
        fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, texture, 0)
        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            raise NPRError(E_GL_CONTEXT, "offscreen framebuffer is incomplete", status=int(status))
        self._fbo = int(fbo)
        self._texture = int(texture)
        self._width = width
        self._height = height
        self._initialised = True

    def roundtrip_bgra(self, image: bytes, width: int, height: int) -> bytes:
        if len(image) != width * height * 4:
            raise NPRError(E_GL_CONTEXT, "BGRA buffer does not match the OpenGL viewport")
        try:
            self._ensure_fbo(width, height)
            GL = self._gl
            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
            GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
            source = (ctypes.c_ubyte * len(image)).from_buffer_copy(image)
            GL.glTexSubImage2D(
                GL.GL_TEXTURE_2D,
                0,
                0,
                0,
                width,
                height,
                GL.GL_BGRA,
                GL.GL_UNSIGNED_BYTE,
                ctypes.cast(source, ctypes.c_void_p),
            )
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)
            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
            GL.glFinish()
            destination = (ctypes.c_ubyte * len(image))()
            GL.glReadPixels(0, 0, width, height, GL.GL_BGRA, GL.GL_UNSIGNED_BYTE, ctypes.cast(destination, ctypes.c_void_p))
            result = bytes(destination)
        except NPRError:
            raise
        except Exception as exc:  # pragma: no cover - depends on local OpenGL
            raise NPRError(E_GL_CONTEXT, "offscreen FBO roundtrip failed") from exc
        if len(result) != len(image):
            raise NPRError(E_GL_CONTEXT, "offscreen FBO returned a short BGRA buffer")
        return result

    def close(self) -> None:
        if not self._initialised:
            return
        try:  # pragma: no cover - depends on local OpenGL
            self._gl.glDeleteFramebuffers(1, [self._fbo])
            self._gl.glDeleteTextures([self._texture])
        finally:
            self._initialised = False


class _GLFWPlatform:
    def __init__(self) -> None:
        import glfw  # type: ignore

        if not glfw.init():
            raise NPRError(E_GL_CONTEXT, "GLFW initialisation failed")
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        self._glfw = glfw
        self.window = glfw.create_window(1, 1, "npr-hidden", None, None)
        if not self.window:
            glfw.terminate()
            raise NPRError(E_GL_CONTEXT, "GLFW could not create a hidden OpenGL window")
        glfw.make_context_current(self.window)

    def close(self) -> None:
        self._glfw.make_context_current(None)
        self._glfw.destroy_window(self.window)
        self._glfw.terminate()


class _WGLPlatform:  # pragma: no cover - Windows-only path
    PFD_DRAW_TO_WINDOW = 0x00000004
    PFD_SUPPORT_OPENGL = 0x00000020
    PFD_DOUBLEBUFFER = 0x00000001
    PFD_TYPE_RGBA = 0
    WGL_CONTEXT_MAJOR_VERSION_ARB = 0x2091
    WGL_CONTEXT_MINOR_VERSION_ARB = 0x2092
    WGL_CONTEXT_PROFILE_MASK_ARB = 0x2094
    WGL_CONTEXT_CORE_PROFILE_BIT_ARB = 0x00000001

    class PIXELFORMATDESCRIPTOR(ctypes.Structure):
        _fields_ = [
            ("nSize", ctypes.c_ushort),
            ("nVersion", ctypes.c_ushort),
            ("dwFlags", ctypes.c_uint32),
            ("iPixelType", ctypes.c_ubyte),
            ("cColorBits", ctypes.c_ubyte),
            ("cRedBits", ctypes.c_ubyte),
            ("cRedShift", ctypes.c_ubyte),
            ("cGreenBits", ctypes.c_ubyte),
            ("cGreenShift", ctypes.c_ubyte),
            ("cBlueBits", ctypes.c_ubyte),
            ("cBlueShift", ctypes.c_ubyte),
            ("cAlphaBits", ctypes.c_ubyte),
            ("cAlphaShift", ctypes.c_ubyte),
            ("cAccumBits", ctypes.c_ubyte),
            ("cAccumRedBits", ctypes.c_ubyte),
            ("cAccumGreenBits", ctypes.c_ubyte),
            ("cAccumBlueBits", ctypes.c_ubyte),
            ("cAccumAlphaBits", ctypes.c_ubyte),
            ("cDepthBits", ctypes.c_ubyte),
            ("cStencilBits", ctypes.c_ubyte),
            ("cAuxBuffers", ctypes.c_ubyte),
            ("iLayerType", ctypes.c_ubyte),
            ("bReserved", ctypes.c_ubyte),
            ("dwLayerMask", ctypes.c_uint32),
            ("dwVisibleMask", ctypes.c_uint32),
            ("dwDamageMask", ctypes.c_uint32),
        ]

    def __init__(self) -> None:
        if os.name != "nt":
            raise NPRError(E_GL_CONTEXT, "Windows WGL backend is unavailable on this platform")
        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32
        opengl32 = ctypes.windll.opengl32
        self._user32 = user32
        self._gdi32 = gdi32
        self._opengl32 = opengl32
        user32.CreateWindowExW.restype = ctypes.c_void_p
        user32.CreateWindowExW.argtypes = [
            ctypes.c_uint32,
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        user32.GetDC.restype = ctypes.c_void_p
        user32.GetDC.argtypes = [ctypes.c_void_p]
        user32.ReleaseDC.restype = ctypes.c_int
        user32.ReleaseDC.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        user32.DestroyWindow.restype = ctypes.c_bool
        user32.DestroyWindow.argtypes = [ctypes.c_void_p]
        gdi32.ChoosePixelFormat.restype = ctypes.c_int
        gdi32.ChoosePixelFormat.argtypes = [ctypes.c_void_p, ctypes.POINTER(self.PIXELFORMATDESCRIPTOR)]
        gdi32.SetPixelFormat.restype = ctypes.c_bool
        gdi32.SetPixelFormat.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(self.PIXELFORMATDESCRIPTOR)]
        opengl32.wglCreateContext.restype = ctypes.c_void_p
        opengl32.wglCreateContext.argtypes = [ctypes.c_void_p]
        opengl32.wglMakeCurrent.restype = ctypes.c_bool
        opengl32.wglMakeCurrent.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        opengl32.wglDeleteContext.restype = ctypes.c_bool
        opengl32.wglDeleteContext.argtypes = [ctypes.c_void_p]
        opengl32.wglGetProcAddress.restype = ctypes.c_void_p
        opengl32.wglGetProcAddress.argtypes = [ctypes.c_char_p]
        hwnd = user32.CreateWindowExW(0, "STATIC", "npr-hidden", 0, 0, 0, 1, 1, None, None, None, None)
        if not hwnd:
            raise NPRError(E_GL_CONTEXT, "WGL hidden window creation failed")
        hdc = user32.GetDC(hwnd)
        descriptor = self.PIXELFORMATDESCRIPTOR()
        descriptor.nSize = ctypes.sizeof(descriptor)
        descriptor.nVersion = 1
        descriptor.dwFlags = self.PFD_DRAW_TO_WINDOW | self.PFD_SUPPORT_OPENGL | self.PFD_DOUBLEBUFFER
        descriptor.iPixelType = self.PFD_TYPE_RGBA
        descriptor.cColorBits = 32
        descriptor.cAlphaBits = 8
        descriptor.cDepthBits = 24
        pixel_format = gdi32.ChoosePixelFormat(hdc, ctypes.byref(descriptor))
        if not pixel_format or not gdi32.SetPixelFormat(hdc, pixel_format, ctypes.byref(descriptor)):
            raise NPRError(E_GL_CONTEXT, "WGL pixel format selection failed")
        dummy = opengl32.wglCreateContext(hdc)
        if not dummy or not opengl32.wglMakeCurrent(hdc, dummy):
            raise NPRError(E_GL_CONTEXT, "WGL context creation failed")
        try:
            proc = opengl32.wglGetProcAddress(b"wglCreateContextAttribsARB")
            if not proc:
                raise NPRError(E_GL_CONTEXT, "WGL_ARB_create_context is not available")
            create_attribs = ctypes.WINFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int))(
                proc
            )
            attribs = (ctypes.c_int * 7)(
                self.WGL_CONTEXT_MAJOR_VERSION_ARB,
                3,
                self.WGL_CONTEXT_MINOR_VERSION_ARB,
                3,
                self.WGL_CONTEXT_PROFILE_MASK_ARB,
                self.WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
                0,
            )
            context = create_attribs(hdc, None, attribs)
            if not context or not opengl32.wglMakeCurrent(hdc, context):
                raise NPRError(E_GL_CONTEXT, "WGL 3.3 core context creation failed")
        finally:
            opengl32.wglMakeCurrent(None, None)
            opengl32.wglDeleteContext(dummy)
        self.hwnd = hwnd
        self.hdc = hdc
        self.context = context

    def close(self) -> None:  # pragma: no cover - Windows-only path
        self._opengl32.wglMakeCurrent(None, None)
        self._opengl32.wglDeleteContext(self.context)
        self._user32.ReleaseDC(self.hwnd, self.hdc)
        self._user32.DestroyWindow(self.hwnd)


def open_offscreen_gl() -> OffscreenGL:
    """Create an invisible context or raise ``E_GL_CONTEXT``.

    GLFW is used when installed.  On Windows a small WGL bootstrap is used as a
    fallback, matching the dependency list in the task statement.
    """

    if os.environ.get("NPR_DISABLE_GL", "").lower() in {"1", "true", "yes"}:
        raise NPRError(E_GL_CONTEXT, "offscreen OpenGL was disabled by NPR_DISABLE_GL")
    platform: Any = None
    errors: list[str] = []
    try:
        platform = _GLFWPlatform()
    except Exception as exc:
        errors.append(f"glfw:{type(exc).__name__}:{exc}")
    if platform is None and os.name == "nt":
        try:
            platform = _WGLPlatform()
        except Exception as exc:
            errors.append(f"wgl:{type(exc).__name__}:{exc}")
    if platform is None:
        raise NPRError(E_GL_CONTEXT, "no local offscreen OpenGL context could be created", attempts=errors)
    backend = OffscreenGL(platform)
    try:
        # Force FBO creation immediately so probe failures are reported here.
        backend._ensure_fbo(1, 1)
    except Exception:
        platform.close()
        raise
    return backend


def probe_offscreen_gl() -> dict[str, Any]:
    backend = open_offscreen_gl()
    try:
        return {"available": True, "backend": type(backend.platform_backend).__name__}
    finally:
        backend.close()
