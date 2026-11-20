"""Invisible FBO rendering with a CPU/mesh-only fallback on unsupported hosts."""

from __future__ import annotations

import ctypes
import hashlib
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import ErrorCode, StudioError
from .camera import CameraPose, look_at, perspective
from .snapshots import SceneSnapshot, SnapshotStore


@dataclass(frozen=True)
class RenderResult:
    ok: bool
    code: str | None
    message: str
    context: dict[str, Any]
    width: int
    height: int
    pixels_rgba: bytes | None = None
    image_sha256: str | None = None
    snapshot_version: int | None = None
    snapshot_content_sha256: str | None = None
    renderer: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "code": self.code,
            "message": self.message,
            "context": self.context,
            "width": self.width,
            "height": self.height,
            "image_sha256": self.image_sha256,
            "snapshot_version": self.snapshot_version,
            "snapshot_content_sha256": self.snapshot_content_sha256,
            "renderer": self.renderer,
        }


class _WindowsPixelFormatDescriptor(ctypes.Structure):
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


class _WindowsWglContext:
    PFD_DRAW_TO_WINDOW = 0x00000004
    PFD_SUPPORT_OPENGL = 0x00000020
    PFD_DOUBLEBUFFER = 0x00000001
    WS_POPUP = 0x80000000

    def __init__(self) -> None:
        self.user32: Any = None
        self.kernel32: Any = None
        self.gdi32: Any = None
        self.opengl32: Any = None
        self.hwnd: int | None = None
        self.hdc: int | None = None
        self.hglrc: int | None = None
        self._lock = threading.RLock()

    def create(self) -> None:
        if os.name != "nt":
            raise StudioError(
                ErrorCode.E_GL_CONTEXT,
                "WGL offscreen context is only implemented on Windows",
                context={"platform": os.name},
            )
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
        self.opengl32 = ctypes.WinDLL("opengl32", use_last_error=True)
        self.user32.CreateWindowExW.restype = ctypes.c_void_p
        self.user32.CreateWindowExW.argtypes = [
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
        self.kernel32.GetModuleHandleW.restype = ctypes.c_void_p
        self.kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
        self.user32.GetDC.restype = ctypes.c_void_p
        self.user32.GetDC.argtypes = [ctypes.c_void_p]
        self.user32.ReleaseDC.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.user32.DestroyWindow.argtypes = [ctypes.c_void_p]
        self.gdi32.ChoosePixelFormat.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_WindowsPixelFormatDescriptor),
        ]
        self.gdi32.SetPixelFormat.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(_WindowsPixelFormatDescriptor),
        ]
        self.opengl32.wglCreateContext.restype = ctypes.c_void_p
        self.opengl32.wglCreateContext.argtypes = [ctypes.c_void_p]
        self.opengl32.wglMakeCurrent.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.opengl32.wglDeleteContext.argtypes = [ctypes.c_void_p]

        instance = self.kernel32.GetModuleHandleW(None)
        self.hwnd = self.user32.CreateWindowExW(
            0,
            "STATIC",
            "pointcloud-studio-offscreen",
            self.WS_POPUP,
            0,
            0,
            64,
            64,
            None,
            None,
            instance,
            None,
        )
        if not self.hwnd:
            raise StudioError(
                ErrorCode.E_GL_CONTEXT,
                "hidden Win32 window creation failed",
                context={"win32_error": ctypes.get_last_error()},
            )
        self.hdc = self.user32.GetDC(self.hwnd)
        if not self.hdc:
            raise StudioError(
                ErrorCode.E_GL_CONTEXT,
                "device context creation failed",
                context={"win32_error": ctypes.get_last_error()},
            )
        descriptor = _WindowsPixelFormatDescriptor()
        descriptor.nSize = ctypes.sizeof(_WindowsPixelFormatDescriptor)
        descriptor.nVersion = 1
        descriptor.dwFlags = self.PFD_DRAW_TO_WINDOW | self.PFD_SUPPORT_OPENGL | self.PFD_DOUBLEBUFFER
        descriptor.iPixelType = 0
        descriptor.cColorBits = 32
        descriptor.cDepthBits = 24
        descriptor.cAlphaBits = 8
        pixel_format = self.gdi32.ChoosePixelFormat(self.hdc, ctypes.byref(descriptor))
        if not pixel_format:
            raise StudioError(
                ErrorCode.E_GL_CONTEXT,
                "no suitable WGL pixel format was found",
                context={"win32_error": ctypes.get_last_error()},
            )
        if not self.gdi32.SetPixelFormat(self.hdc, pixel_format, ctypes.byref(descriptor)):
            raise StudioError(
                ErrorCode.E_GL_CONTEXT,
                "SetPixelFormat failed",
                context={"win32_error": ctypes.get_last_error(), "pixel_format": pixel_format},
            )
        self.hglrc = self.opengl32.wglCreateContext(self.hdc)
        if not self.hglrc:
            raise StudioError(
                ErrorCode.E_GL_CONTEXT,
                "wglCreateContext failed",
                context={"win32_error": ctypes.get_last_error()},
            )
        if not self.opengl32.wglMakeCurrent(self.hdc, self.hglrc):
            raise StudioError(
                ErrorCode.E_GL_CONTEXT,
                "wglMakeCurrent failed",
                context={"win32_error": ctypes.get_last_error()},
            )

    def make_current(self) -> None:
        if not self.opengl32.wglMakeCurrent(self.hdc, self.hglrc):
            raise StudioError(
                ErrorCode.E_GL_CONTEXT,
                "wglMakeCurrent failed while entering the view thread",
                context={"win32_error": ctypes.get_last_error()},
            )

    def close(self) -> None:
        if self.opengl32 is not None:
            try:
                self.opengl32.wglMakeCurrent(None, None)
            except Exception:
                pass
            if self.hglrc:
                self.opengl32.wglDeleteContext(self.hglrc)
                self.hglrc = None
        if self.user32 is not None:
            if self.hdc and self.hwnd:
                self.user32.ReleaseDC(self.hwnd, self.hdc)
                self.hdc = None
            if self.hwnd:
                self.user32.DestroyWindow(self.hwnd)
                self.hwnd = None


class WglFboRenderer:
    """Windows-only hidden-window WGL plus FBO renderer."""

    def __init__(self) -> None:
        self.context = _WindowsWglContext()
        self.context.create()
        try:
            import OpenGL.GL as gl
        except Exception as exc:
            self.context.close()
            raise StudioError(
                ErrorCode.E_GL_CONTEXT,
                "PyOpenGL could not be imported after WGL context creation",
                context={"reason": str(exc)},
            ) from exc
        self.gl = gl
        self.version_string = str(gl.glGetString(gl.GL_VERSION).decode("ascii", errors="replace"))
        self.renderer_string = str(gl.glGetString(gl.GL_RENDERER).decode("ascii", errors="replace"))
        self._fbo = None
        self._color = None
        self._depth = None
        self._size = (0, 0)
        self._program = None
        self._vao = None
        self._vbo_position = None
        self._vbo_color = None
        self._vbo_mesh = None
        self._vbo_mesh_color = None
        self._lock = threading.RLock()
        self._initialize_gl()

    def _initialize_gl(self) -> None:
        gl = self.gl
        vertex_source = """
            #version 330 core
            layout(location = 0) in vec3 a_position;
            layout(location = 1) in vec3 a_color;
            uniform mat4 u_mvp;
            out vec3 v_color;
            void main() {
                gl_Position = u_mvp * vec4(a_position, 1.0);
                gl_PointSize = 2.0;
                v_color = a_color;
            }
        """
        fragment_source = """
            #version 330 core
            in vec3 v_color;
            out vec4 frag_color;
            void main() { frag_color = vec4(v_color, 1.0); }
        """

        def compile_shader(source: str, shader_type: int) -> int:
            shader = gl.glCreateShader(shader_type)
            gl.glShaderSource(shader, source)
            gl.glCompileShader(shader)
            if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
                log = gl.glGetShaderInfoLog(shader).decode("utf-8", errors="replace")
                raise StudioError(
                    ErrorCode.E_GL_CONTEXT,
                    "offscreen shader compilation failed",
                    context={"log": log},
                )
            return shader

        vertex = compile_shader(vertex_source, gl.GL_VERTEX_SHADER)
        fragment = compile_shader(fragment_source, gl.GL_FRAGMENT_SHADER)
        program = gl.glCreateProgram()
        gl.glAttachShader(program, vertex)
        gl.glAttachShader(program, fragment)
        gl.glLinkProgram(program)
        if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
            log = gl.glGetProgramInfoLog(program).decode("utf-8", errors="replace")
            raise StudioError(
                ErrorCode.E_GL_CONTEXT,
                "offscreen shader link failed",
                context={"log": log},
            )
        gl.glDeleteShader(vertex)
        gl.glDeleteShader(fragment)
        self._program = program
        self._vao = gl.glGenVertexArrays(1)
        self._vbo_position = gl.glGenBuffers(1)
        self._vbo_color = gl.glGenBuffers(1)
        self._vbo_mesh = gl.glGenBuffers(1)
        self._vbo_mesh_color = gl.glGenBuffers(1)

    def _ensure_fbo(self, width: int, height: int) -> None:
        if self._fbo is not None and self._size == (width, height):
            return
        gl = self.gl
        if self._fbo is not None:
            gl.glDeleteFramebuffers(1, [self._fbo])
            gl.glDeleteTextures([self._color])
            gl.glDeleteRenderbuffers(1, [self._depth])
        self._fbo = gl.glGenFramebuffers(1)
        self._color = gl.glGenTextures(1)
        self._depth = gl.glGenRenderbuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._fbo)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._color)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA8,
            width,
            height,
            0,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            None,
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self._color, 0
        )
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self._depth)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT24, width, height)
        gl.glFramebufferRenderbuffer(
            gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, self._depth
        )
        status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        if status != gl.GL_FRAMEBUFFER_COMPLETE:
            raise StudioError(
                ErrorCode.E_GL_CONTEXT,
                "offscreen FBO is incomplete",
                context={"status": int(status), "width": width, "height": height},
            )
        self._size = (width, height)

    def _upload(self, buffer: int, array: np.ndarray) -> None:
        gl = self.gl
        data = np.ascontiguousarray(array, dtype=np.float32)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_DYNAMIC_DRAW)

    def render(self, snapshot: SceneSnapshot, width: int, height: int, camera: CameraPose) -> bytes:
        if width <= 0 or height <= 0 or width > 8192 or height > 8192:
            raise StudioError(
                ErrorCode.E_GL_CONTEXT,
                "offscreen render dimensions are invalid",
                context={"width": width, "height": height},
            )
        with self._lock:
            self.context.make_current()
            gl = self.gl
            self._ensure_fbo(width, height)
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._fbo)
            gl.glViewport(0, 0, width, height)
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
            gl.glClearColor(0.035, 0.045, 0.07, 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glUseProgram(self._program)
            gl.glBindVertexArray(self._vao)
            mvp = perspective(camera.fov_y_degrees, width / height, camera.near, camera.far) @ look_at(
                np.asarray(camera.eye, dtype=np.float64),
                np.asarray(camera.target, dtype=np.float64),
                np.asarray(camera.up, dtype=np.float64),
            )
            location = gl.glGetUniformLocation(self._program, "u_mvp")
            gl.glUniformMatrix4fv(location, 1, gl.GL_FALSE, np.ascontiguousarray(mvp.T, dtype=np.float32))

            point_count = snapshot.positions.shape[0]
            point_colors = np.empty((point_count, 3), dtype=np.float32)
            point_colors[:, 0] = 0.35 + 0.45 * snapshot.confidence
            point_colors[:, 1] = 0.55 + 0.35 * snapshot.confidence
            point_colors[:, 2] = 0.95
            self._upload(self._vbo_position, snapshot.positions)
            self._upload(self._vbo_color, point_colors)
            gl.glEnableVertexAttribArray(0)
            gl.glEnableVertexAttribArray(1)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_position)
            gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_color)
            gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
            gl.glDrawArrays(gl.GL_POINTS, 0, point_count)

            mesh = snapshot.mesh
            if mesh.triangles.shape[0]:
                vertex_count = mesh.vertices.shape[0]
                mesh_colors = np.empty((vertex_count, 3), dtype=np.float32)
                extent = np.maximum(mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0), 1.0e-9)
                normalized = (mesh.vertices - mesh.vertices.min(axis=0)) / extent
                mesh_colors[:, 0] = 0.24 + 0.55 * normalized[:, 1]
                mesh_colors[:, 1] = 0.52 + 0.26 * normalized[:, 0]
                mesh_colors[:, 2] = 0.42 + 0.36 * normalized[:, 2]
                self._upload(self._vbo_mesh, mesh.vertices)
                self._upload(self._vbo_mesh_color, mesh_colors)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_mesh)
                gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_mesh_color)
                gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
                index_buffer = gl.glGenBuffers(1)
                indices = np.ascontiguousarray(mesh.triangles, dtype=np.uint32)
                gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, index_buffer)
                gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_DYNAMIC_DRAW)
                gl.glDrawElements(gl.GL_TRIANGLES, int(indices.size), gl.GL_UNSIGNED_INT, None)
                gl.glDeleteBuffers(1, [index_buffer])
                # Overlay points after the surface so the viewport communicates
                # both data representations rather than hiding points behind it.
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_position)
                gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_color)
                gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
                gl.glDrawArrays(gl.GL_POINTS, 0, point_count)
            gl.glFinish()
            pixels = gl.glReadPixels(0, 0, width, height, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
            return np.asarray(pixels, dtype=np.uint8).tobytes(order="C")

    def close(self) -> None:
        with self._lock:
            gl = getattr(self, "gl", None)
            if gl is not None:
                try:
                    if self._fbo is not None:
                        gl.glDeleteFramebuffers(1, [self._fbo])
                    if self._color is not None:
                        gl.glDeleteTextures([self._color])
                    if self._depth is not None:
                        gl.glDeleteRenderbuffers(1, [self._depth])
                    if self._program is not None:
                        gl.glDeleteProgram(self._program)
                    gl.glDeleteBuffers(
                        1,
                        [
                            buffer
                            for buffer in (
                                self._vbo_position,
                                self._vbo_color,
                                self._vbo_mesh,
                                self._vbo_mesh_color,
                            )
                            if buffer is not None
                        ],
                    )
                    if self._vao is not None:
                        gl.glDeleteVertexArrays(1, [self._vao])
                finally:
                    self.context.close()


class OffscreenViewport:
    """View-thread helper. It only reads atomic SceneSnapshot values."""

    def __init__(self, store: SnapshotStore, *, forced_unavailable_reason: str | None = None) -> None:
        self.store = store
        self._renderer: WglFboRenderer | None = None
        self._lock = threading.Lock()
        self._unavailable: StudioError | None = None
        self._forced_unavailable_reason = forced_unavailable_reason

    def _get_renderer(self) -> WglFboRenderer:
        if self._unavailable is not None:
            raise self._unavailable
        if self._renderer is not None:
            return self._renderer
        with self._lock:
            if self._renderer is None:
                try:
                    self._renderer = WglFboRenderer()
                except StudioError as exc:
                    self._unavailable = exc
                    raise
        return self._renderer

    def render(
        self,
        *,
        width: int = 960,
        height: int = 640,
        expected_version: int | None = None,
        camera: CameraPose | None = None,
    ) -> RenderResult:
        if self._forced_unavailable_reason is not None:
            snapshot = None
            try:
                snapshot = self.store.get(expected_version)
            except StudioError:
                pass
            return RenderResult(
                ok=False,
                code=ErrorCode.E_GL_CONTEXT.value,
                message="offscreen context was made unavailable by a serialized injection parameter",
                context={
                    "injected": True,
                    "reason": self._forced_unavailable_reason,
                    "snapshot_version_visible_cpu_path": snapshot.version if snapshot else None,
                },
                width=width,
                height=height,
                snapshot_version=snapshot.version if snapshot else None,
                snapshot_content_sha256=snapshot.content_sha256 if snapshot else None,
            )
        try:
            snapshot = self.store.get(expected_version)
        except StudioError as exc:
            return RenderResult(
                ok=False,
                code=exc.code.value,
                message=exc.message,
                context=exc.context,
                width=width,
                height=height,
            )
        if os.name != "nt":
            return RenderResult(
                ok=False,
                code=ErrorCode.E_GL_CONTEXT.value,
                message="offscreen WGL/FBO context is unavailable on this platform; CPU mesh and ledger verification remain available",
                context={
                    "platform": os.name,
                    "expected_version": expected_version,
                    "snapshot_version_visible_cpu_path": snapshot.version,
                },
                width=width,
                height=height,
                snapshot_version=snapshot.version,
                snapshot_content_sha256=snapshot.content_sha256,
            )
        try:
            renderer = self._get_renderer()
            pixels = renderer.render(snapshot, width, height, camera or CameraPose())
            return RenderResult(
                ok=True,
                code=None,
                message="offscreen FBO frame captured",
                context={"fbo_visible": False, "gl_version": renderer.version_string},
                width=width,
                height=height,
                pixels_rgba=pixels,
                image_sha256=hashlib.sha256(pixels).hexdigest(),
                snapshot_version=snapshot.version,
                snapshot_content_sha256=snapshot.content_sha256,
                renderer=renderer.renderer_string,
            )
        except StudioError as exc:
            return RenderResult(
                ok=False,
                code=exc.code.value,
                message=exc.message,
                context=exc.context,
                width=width,
                height=height,
                snapshot_version=snapshot.version,
                snapshot_content_sha256=snapshot.content_sha256,
            )
        except Exception as exc:
            return RenderResult(
                ok=False,
                code=ErrorCode.E_GL_CONTEXT.value,
                message="offscreen rendering failed with a native or driver exception",
                context={
                    "exception_type": type(exc).__name__,
                    "reason": str(exc),
                    "platform": os.name,
                },
                width=width,
                height=height,
                snapshot_version=snapshot.version,
                snapshot_content_sha256=snapshot.content_sha256,
            )

    def close(self) -> None:
        with self._lock:
            if self._renderer is not None:
                self._renderer.close()
                self._renderer = None


def result_to_ppm(result: RenderResult, path: str | Path) -> None:
    if not result.ok or result.pixels_rgba is None:
        raise StudioError(
            ErrorCode.E_GL_CONTEXT,
            "cannot write an image for a failed render",
            context=result.to_dict(),
        )
    width, height = result.width, result.height
    rgba = np.frombuffer(result.pixels_rgba, dtype=np.uint8).reshape(height, width, 4)
    rgb = np.ascontiguousarray(rgba[::-1, :, :3])
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    Path(path).write_bytes(header + rgb.tobytes(order="C"))
