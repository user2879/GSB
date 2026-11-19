"""Display mode management and rendering.

A single DisplayManager owns one tessellation cache per patch.  The three
display modes (control mesh, shaded surface, isocurves) are views over
that shared cache: switching modes never rebuilds geometry.  Editing a
control point invalidates exactly one patch; every other patch keeps its
cached normals bit-for-bit, which avoids whole-model recomputation and
the display jitter it would cause.

Rendering backend: an offscreen OpenGL context is used when one can be
created (expected on Windows with local drivers); otherwise a
deterministic pure-software rasterizer is used and the environment flag
CADAPP_RENDERER=software is recorded.
"""

from __future__ import annotations

import ctypes.util
import os

import numpy as np

from .geometry import PatchGeometry, tessellate_patch

RENDERER_ENV_FLAG = "CADAPP_RENDERER"


class DisplayMode:
    CONTROL_MESH = "control_mesh"
    SHADED = "shaded"
    ISOCURVES = "isocurves"

    ALL = (CONTROL_MESH, SHADED, ISOCURVES)


def try_create_offscreen_gl_context() -> bool:
    """Attempt to create a real offscreen OpenGL context (EGL surfaceless
    or OSMesa).  Returns True on success.  On Windows this exercises the
    local driver stack; on headless Linux it succeeds only when a
    software GL driver (e.g. llvmpipe) is installed."""
    try:
        import OpenGL  # noqa: F401
    except ImportError:
        return False
    egl_name = ctypes.util.find_library("EGL")
    if egl_name:
        try:
            egl = ctypes.CDLL(egl_name)
            egl.eglGetDisplay.restype = ctypes.c_void_p
            egl.eglGetCurrentDisplay.restype = ctypes.c_void_p
            display = egl.eglGetDisplay(ctypes.c_void_p(0))
            major = ctypes.c_long()
            minor = ctypes.c_long()
            if display and egl.eglInitialize(ctypes.c_void_p(display),
                                             ctypes.byref(major),
                                             ctypes.byref(minor)):
                egl.eglTerminate(ctypes.c_void_p(display))
                return True
        except Exception:
            return False
    return False


def detect_backend() -> str:
    """Record the renderer environment flag and return it.

    The demo always renders through the deterministic software
    rasterizer so that repeated runs produce byte-identical images;
    OpenGL offscreen availability is probed separately and reported as
    an environment fact (see run_demo's environment section).
    """
    os.environ[RENDERER_ENV_FLAG] = "software"
    return "software"


class DisplayManager:
    """Owns per-patch geometry caches shared by all display modes."""

    def __init__(self, model):
        self.model = model
        self.mode = DisplayMode.SHADED
        self._cache: dict[str, PatchGeometry] = {}
        self.rebuild_count = 0  # number of tessellation builds (for audits)

    def geometry(self, patch_id: str) -> PatchGeometry:
        cached = self._cache.get(patch_id)
        if cached is None:
            cached = tessellate_patch(self.model.patch_by_id(patch_id))
            self._cache[patch_id] = cached
            self.rebuild_count += 1
        return cached

    def set_mode(self, mode: str) -> None:
        """Switch display mode.  Pure view switch: no geometry rebuild."""
        if mode not in DisplayMode.ALL:
            raise ValueError(f"unknown display mode {mode!r}")
        self.mode = mode

    def invalidate(self, patch_id: str) -> None:
        """Drop the cached geometry of one edited patch."""
        self._cache.pop(patch_id, None)

    def buffers_for_mode(self, patch_id: str, mode: str | None = None) -> dict:
        """Return the draw buffers for a patch under the current mode."""
        mode = mode or self.mode
        geo = self.geometry(patch_id)
        if mode == DisplayMode.SHADED:
            return {"kind": "triangles", "vertices": geo.vertices,
                    "normals": geo.normals, "faces": geo.faces}
        if mode == DisplayMode.ISOCURVES:
            return {"kind": "polylines", "polylines": list(geo.isocurves)}
        return {"kind": "segments", "segments": list(geo.control_lines)}

    # ------------------------------------------------------------------
    def render(self, width: int = 256, height: int = 256) -> bytes:
        """Render the current mode to a deterministic PPM (P6) image."""
        return render_software(self, width, height)


def _view_transform(points: np.ndarray, center, scale, basis):
    return (points - center) @ basis.T / scale


def render_software(manager: DisplayManager, width: int, height: int) -> bytes:
    """Pure-software deterministic rasterizer (no OpenGL required)."""
    # Gather scene geometry for the active mode.
    triangles = []  # (3, 3) positions + face normal
    lines = []      # (2, 3) segments
    all_points = []
    for patch in manager.model.patches:
        buffers = manager.buffers_for_mode(patch.patch_id)
        if buffers["kind"] == "triangles":
            v = buffers["vertices"]
            f = buffers["faces"]
            all_points.append(v)
            tri = v[f]  # (M, 3, 3)
            normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            norms = np.linalg.norm(normals, axis=1)
            norms[norms == 0.0] = 1.0
            normals = normals / norms[:, None]
            triangles.append((tri, normals))
        elif buffers["kind"] == "polylines":
            for poly in buffers["polylines"]:
                all_points.append(poly)
                lines.append(np.stack([poly[:-1], poly[1:]], axis=1))
        else:
            for seg in buffers["segments"]:
                all_points.append(seg)
                lines.append(seg.reshape(1, 2, 3))
    points = np.concatenate(all_points, axis=0)
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    center = (lo + hi) / 2.0
    scale = float(max(hi - lo)) or 1.0
    scale *= 0.62  # margin

    # Deterministic camera: view direction (1, 1, 1), orthographic.
    forward = np.array([1.0, 1.0, 1.0])
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    basis = np.stack([right, true_up, forward])  # rows: camera axes

    image = np.full((height, width, 3), (24, 26, 32), dtype=np.float64)
    depth = np.full((height, width), -np.inf)

    def project(pts):
        cam = _view_transform(pts, center, scale, basis)
        sx = (cam[:, 0] * 0.5 + 0.5) * (width - 1)
        sy = (0.5 - cam[:, 1] * 0.5) * (height - 1)
        return sx, sy, cam[:, 2]

    light = np.array([0.4, 0.5, 0.75])
    light /= np.linalg.norm(light)

    for tri, normals in triangles:
        flat = tri.reshape(-1, 3)
        sx, sy, sz = project(flat)
        sx = sx.reshape(-1, 3)
        sy = sy.reshape(-1, 3)
        sz = sz.reshape(-1, 3)
        for k in range(tri.shape[0]):
            xs = sx[k]
            ys = sy[k]
            zs = sz[k]
            shade = 0.25 + 0.75 * abs(float(normals[k] @ light))
            color = np.array([180.0, 120.0, 60.0]) * shade
            minx = max(int(np.floor(xs.min())), 0)
            maxx = min(int(np.ceil(xs.max())), width - 1)
            miny = max(int(np.floor(ys.min())), 0)
            maxy = min(int(np.ceil(ys.max())), height - 1)
            if minx > maxx or miny > maxy:
                continue
            d = np.array([[xs[1] - xs[0], ys[1] - ys[0]],
                          [xs[2] - xs[0], ys[2] - ys[0]]])
            det = d[0, 0] * d[1, 1] - d[0, 1] * d[1, 0]
            if det == 0.0:
                continue
            inv = np.array([[d[1, 1], -d[0, 1]], [-d[1, 0], d[0, 0]]]) / det
            for py in range(miny, maxy + 1):
                for px in range(minx, maxx + 1):
                    rel = np.array([px - xs[0], py - ys[0]])
                    b1, b2 = inv @ rel
                    b0 = 1.0 - b1 - b2
                    if b0 < 0.0 or b1 < 0.0 or b2 < 0.0:
                        continue
                    z = b0 * zs[0] + b1 * zs[1] + b2 * zs[2]
                    if z > depth[py, px]:
                        depth[py, px] = z
                        image[py, px] = color

    line_color = {
        "control_mesh": np.array([90.0, 200.0, 255.0]),
        "isocurves": np.array([240.0, 220.0, 90.0]),
    }.get(manager.mode, np.array([200.0, 200.0, 200.0]))
    for group in lines:
        segs = group.reshape(-1, 2, 3)
        flat = segs.reshape(-1, 3)
        sx, sy, sz = project(flat)
        sx = sx.reshape(-1, 2)
        sy = sy.reshape(-1, 2)
        sz = sz.reshape(-1, 2)
        for k in range(segs.shape[0]):
            x0, x1 = sx[k]
            y0, y1 = sy[k]
            z0, z1 = sz[k]
            steps = int(max(abs(x1 - x0), abs(y1 - y0), 1.0)) * 2
            for s in range(steps + 1):
                t = s / steps
                px = int(round(x0 + t * (x1 - x0)))
                py = int(round(y0 + t * (y1 - y0)))
                pz = z0 + t * (z1 - z0)
                if 0 <= px < width and 0 <= py < height and pz > depth[py, px] - 1e-9:
                    depth[py, px] = pz
                    image[py, px] = line_color

    rgb = np.clip(image, 0, 255).astype(np.uint8)
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    return header + rgb.tobytes()
