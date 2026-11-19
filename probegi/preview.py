"""Offscreen preview rendering of a reconstructed irradiance frame.

Primary path: PyOpenGL on a native EGL surfaceless context (Mesa software
rendering when no display/GPU is present).  The irradiance frame is uploaded
as a float texture, tone-mapped in a fragment shader and read back.

If the GL context cannot be created the caller receives E_GL_CONTEXT; the
CLI then records the code and writes a clearly-flagged software fallback
preview so the pipeline output stays complete and honest.
"""
import numpy as np

from .errors import GLContextError
from .png import write_png

VERTEX_SRC = b"""#version 330 core
layout(location=0) in vec2 pos;
out vec2 uv;
void main(){ uv = pos*0.5+0.5; gl_Position = vec4(pos,0.0,1.0); }
"""
FRAGMENT_SRC = b"""#version 330 core
uniform sampler2D irradiance;
in vec2 uv;
out vec4 frag;
void main(){
    vec3 e = texture(irradiance, uv).rgb;
    vec3 c = e / (e + vec3(1.0));              // Reinhard tone map
    c = pow(c, vec3(1.0/2.2));                 // gamma
    frag = vec4(c, 1.0);
}
"""


def tone_map_software(frame: np.ndarray) -> np.ndarray:
    e = np.asarray(frame, dtype=np.float64)
    c = e / (e + 1.0)
    return (np.clip(c, 0.0, 1.0) ** (1.0 / 2.2) * 255.0 + 0.5).astype(np.uint8)


def render_frame_gl(frame: np.ndarray) -> np.ndarray:
    """Render (H, W, 3) float irradiance to (H, W, 3) uint8 via OpenGL.

    Raises GLContextError (E_GL_CONTEXT) if no offscreen context is
    available."""
    frame = np.ascontiguousarray(frame, dtype=np.float32)
    h, w, _ = frame.shape
    egl = _EGL()
    try:
        from OpenGL import GL
        prog = _build_program(GL)
        fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB32F, w, h, 0,
                        GL.GL_RGB, GL.GL_FLOAT, frame.tobytes())
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER,
                           GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER,
                           GL.GL_NEAREST)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                                  GL.GL_TEXTURE_2D, tex, 0)
        if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != \
                GL.GL_FRAMEBUFFER_COMPLETE:
            raise GLContextError("incomplete framebuffer")
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32)
        vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, quad.nbytes, quad,
                        GL.GL_STATIC_DRAW)
        GL.glViewport(0, 0, w, h)
        GL.glUseProgram(prog)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, False, 0, None)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
        buf = GL.glReadPixels(0, 0, w, h, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
        img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)[::-1]
        return img.copy()
    except GLContextError:
        raise
    except Exception as exc:  # any GL failure surfaces as E_GL_CONTEXT
        raise GLContextError(f"GL rendering failed: {exc}") from exc
    finally:
        egl.close()


def _build_program(GL):
    vs = GL.glCreateShader(GL.GL_VERTEX_SHADER)
    GL.glShaderSource(vs, VERTEX_SRC)
    GL.glCompileShader(vs)
    if not GL.glGetShaderiv(vs, GL.GL_COMPILE_STATUS):
        raise GLContextError("vertex shader: "
                             + GL.glGetShaderInfoLog(vs).decode())
    fs = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
    GL.glShaderSource(fs, FRAGMENT_SRC)
    GL.glCompileShader(fs)
    if not GL.glGetShaderiv(fs, GL.GL_COMPILE_STATUS):
        raise GLContextError("fragment shader: "
                             + GL.glGetShaderInfoLog(fs).decode())
    prog = GL.glCreateProgram()
    GL.glAttachShader(prog, vs)
    GL.glAttachShader(prog, fs)
    GL.glLinkProgram(prog)
    if not GL.glGetProgramiv(prog, GL.GL_LINK_STATUS):
        raise GLContextError("program link failed")
    return prog


class _EGL:
    """Minimal EGL surfaceless context via ctypes (Mesa-friendly)."""

    EGL_PLATFORM_SURFACELESS_MESA = 0x31DD
    EGL_NO_CONTEXT = 0
    EGL_NO_DISPLAY = 0
    EGL_NO_SURFACE = 0
    EGL_NONE = 0x3038
    EGL_CONTEXT_CLIENT_VERSION = 0x3098
    EGL_OPENGL_API = 0x30A2

    def __init__(self):
        import ctypes
        try:
            self._lib = ctypes.CDLL("libEGL.so.1")
        except OSError as exc:
            raise GLContextError(f"libEGL unavailable: {exc}") from exc
        c = ctypes
        L = self._lib
        L.eglGetPlatformDisplay.restype = c.c_void_p
        L.eglGetPlatformDisplay.argtypes = [c.c_uint, c.c_void_p, c.c_void_p]
        L.eglInitialize.argtypes = [c.c_void_p, c.c_void_p, c.c_void_p]
        L.eglBindAPI.argtypes = [c.c_uint]
        L.eglCreateContext.restype = c.c_void_p
        L.eglCreateContext.argtypes = [c.c_void_p, c.c_void_p, c.c_void_p,
                                       c.POINTER(c.c_int)]
        L.eglMakeCurrent.argtypes = [c.c_void_p] * 4
        L.eglDestroyContext.argtypes = [c.c_void_p, c.c_void_p]
        L.eglTerminate.argtypes = [c.c_void_p]
        L.eglGetError.restype = c.c_int
        self.display = L.eglGetPlatformDisplay(
            self.EGL_PLATFORM_SURFACELESS_MESA, None, None)
        if not self.display or self.display == self.EGL_NO_DISPLAY:
            raise GLContextError("eglGetPlatformDisplay(surfaceless) failed",
                                 egl_error=hex(L.eglGetError()))
        if not L.eglInitialize(self.display, None, None):
            raise GLContextError("eglInitialize failed",
                                 egl_error=hex(L.eglGetError()))
        if not L.eglBindAPI(self.EGL_OPENGL_API):
            raise GLContextError("eglBindAPI(OPENGL_API) failed")
        attribs = (c.c_int * 5)(self.EGL_CONTEXT_CLIENT_VERSION, 3,
                                self.EGL_NONE, 0, 0)
        self.context = L.eglCreateContext(self.display, None, None, attribs)
        if not self.context:
            raise GLContextError("eglCreateContext failed",
                                 egl_error=hex(L.eglGetError()))
        if not L.eglMakeCurrent(self.display, self.EGL_NO_SURFACE,
                                self.EGL_NO_SURFACE, self.context):
            raise GLContextError("eglMakeCurrent(surfaceless) failed",
                                 egl_error=hex(L.eglGetError()))

    def close(self):
        try:
            self._lib.eglMakeCurrent(self.display, 0, 0, 0)
            self._lib.eglDestroyContext(self.display, self.context)
            self._lib.eglTerminate(self.display)
        except Exception:
            pass


def render_preview(frame: np.ndarray, path: str) -> dict:
    """Render a preview PNG; returns a status record for the report."""
    try:
        img = render_frame_gl(frame)
        renderer = "opengl-egl-surfaceless"
        status = "ok"
    except GLContextError as exc:
        img = tone_map_software(frame)
        renderer = "software-fallback"
        status = exc.code
    write_png(path, img)
    from .hashing import sha256_bytes
    with open(path, "rb") as fh:
        digest = sha256_bytes(fh.read())
    return {"renderer": renderer, "status": status, "path": path,
            "sha256": digest}
