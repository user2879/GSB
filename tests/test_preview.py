import numpy as np

from probegi.preview import render_preview


def test_preview_written(tmp_path):
    rng = np.random.default_rng(0)
    frame = rng.uniform(0.0, 4.0, (48, 48, 3))
    rec = render_preview(frame, str(tmp_path / "p.png"))
    assert (tmp_path / "p.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert rec["renderer"] in ("opengl-egl-surfaceless", "software-fallback")
    if rec["renderer"] == "software-fallback":
        assert rec["status"] == "E_GL_CONTEXT"
    else:
        assert rec["status"] == "ok"


def test_gl_context_error_code():
    from probegi.errors import GLContextError
    exc = GLContextError("no display")
    assert exc.code == "E_GL_CONTEXT"
    assert exc.to_dict()["code"] == "E_GL_CONTEXT"
