"""Optional offscreen OpenGL/FBO readback probe using PyOpenGL and GLFW."""

from __future__ import annotations

from typing import Any


def check_offscreen_fbo(width: int = 64, height: int = 32) -> dict[str, Any]:
    try:
        import glfw
        from OpenGL import GL
    except Exception as exc:
        return {"available": False, "backend": "pyopengl+glfw", "reason": f"import failed: {exc}"}
    if not glfw.init():
        return {"available": False, "backend": "pyopengl+glfw", "reason": "glfw.init failed; no local display/context"}
    window = None
    try:
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.NATIVE_CONTEXT_API)
        window = glfw.create_window(width, height, "acoustic-offscreen", None, None)
        if not window:
            return {"available": False, "backend": "pyopengl+glfw", "reason": "hidden GL context creation failed"}
        glfw.make_context_current(window)
        framebuffer = GL.glGenFramebuffers(1)
        texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, width, height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, framebuffer)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, texture, 0)
        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            return {"available": False, "backend": "pyopengl+glfw", "reason": f"framebuffer incomplete: {status}"}
        GL.glViewport(0, 0, width, height)
        GL.glClearColor(0.05, 0.20, 0.35, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        pixels = GL.glReadPixels(0, 0, width, height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
        first = tuple(int(pixels[index]) for index in range(4))
        expected = (13, 51, 89, 255)
        if first != expected:
            return {"available": False, "backend": "pyopengl+glfw", "reason": f"pixel readback mismatch: {first} != {expected}"}
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glDeleteTextures([texture])
        GL.glDeleteFramebuffers(1, [framebuffer])
        return {
            "available": True,
            "backend": "pyopengl+glfw offscreen FBO",
            "width": width,
            "height": height,
            "readback_first_pixel": list(first),
        }
    except Exception as exc:
        return {"available": False, "backend": "pyopengl+glfw", "reason": str(exc)}
    finally:
        if window is not None:
            glfw.destroy_window(window)
        glfw.terminate()
