using System.Drawing.Drawing2D;
using LevelWorkbench.Core;

namespace LevelWorkbench.Desktop;

public sealed class ViewportControl : Control
{
    private enum PointerMode
    {
        None,
        DragProxy,
        BoxSelect,
        Pan
    }

    private PointerMode _mode;
    private Vec2 _lastScenePoint;
    private Vec2 _boxStart;
    private Vec2? _boxCurrent;
    private bool _dragMoved;
    private PickHit? _downHit;
    private double _controlScale = 1;
    private double _offsetX;
    private double _offsetY;

    public ViewportControl()
    {
        SetStyle(
            ControlStyles.AllPaintingInWmPaint
            | ControlStyles.OptimizedDoubleBuffer
            | ControlStyles.UserPaint
            | ControlStyles.ResizeRedraw
            | ControlStyles.Selectable,
            true);
        TabStop = true;
        BackColor = Color.FromArgb(16, 20, 27);
    }

    public EditorSession? Session { get; set; }

    public string? ActiveFilter { get; set; } = "default";

    public PickHit? PreviewHit { get; set; }

    public event Action<PickHit>? HitCommitted;

    public event Action<string>? StatusChanged;

    protected override void OnPaint(PaintEventArgs e)
    {
        base.OnPaint(e);
        e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
        e.Graphics.Clear(BackColor);
        if (Session is null)
        {
            DrawCenteredText(e.Graphics, "Load a scene to begin");
            return;
        }

        UpdateControlMapping();
        DrawGrid(e.Graphics);
        DrawProxies(e.Graphics);
        DrawPreview(e.Graphics);
        DrawRubberBand(e.Graphics);
        DrawHud(e.Graphics);
    }

    protected override void OnMouseDown(MouseEventArgs e)
    {
        base.OnMouseDown(e);
        Focus();
        if (Session is null)
        {
            return;
        }

        UpdateControlMapping();
        var point = MapToScene(e.Location);
        _lastScenePoint = point;
        if (e.Button == MouseButtons.Middle)
        {
            _mode = PointerMode.Pan;
            Cursor = Cursors.SizeAll;
            return;
        }

        if (e.Button != MouseButtons.Left)
        {
            return;
        }

        if ((ModifierKeys & Keys.Shift) != 0)
        {
            _mode = PointerMode.BoxSelect;
            _boxStart = point;
            _boxCurrent = point;
            Invalidate();
            return;
        }

        if (!IsInsideSceneViewport(point))
        {
            return;
        }

        var hit = Session.Pick(point, ActiveFilter);
        PreviewHit = hit;
        if (hit is not null)
        {
            _downHit = hit;
            _mode = PointerMode.DragProxy;
            _dragMoved = false;
            Session.BeginDrag(hit, point, ActiveFilter, additiveSelection: (ModifierKeys & Keys.Control) != 0);
            StatusChanged?.Invoke($"Picked {hit.StableId} face {hit.FaceIndex}");
        }
        else
        {
            Session.SelectAt(point, ActiveFilter, additive: (ModifierKeys & Keys.Control) != 0);
            StatusChanged?.Invoke("No pick (filters or viewport clipping excluded all proxies).");
        }

        Invalidate();
    }

    protected override void OnMouseMove(MouseEventArgs e)
    {
        base.OnMouseMove(e);
        if (Session is null)
        {
            return;
        }

        UpdateControlMapping();
        var point = MapToScene(e.Location);
        switch (_mode)
        {
            case PointerMode.Pan:
                Session.PanBy(point - _lastScenePoint);
                _lastScenePoint = point;
                Invalidate();
                break;
            case PointerMode.DragProxy:
                if ((point - _lastScenePoint).Length > 2)
                {
                    _dragMoved = true;
                }

                PreviewHit = Session.UpdateDrag(point);
                _lastScenePoint = point;
                if (PreviewHit is not null)
                {
                    StatusChanged?.Invoke($"Dragging · hit {PreviewHit.StableId} face {PreviewHit.FaceIndex}");
                }
                else
                {
                    StatusChanged?.Invoke("Dragging · no current hit");
                }

                Invalidate();
                break;
            case PointerMode.BoxSelect:
                _boxCurrent = point;
                Invalidate();
                break;
        }
    }

    protected override void OnMouseUp(MouseEventArgs e)
    {
        base.OnMouseUp(e);
        if (Session is null)
        {
            _mode = PointerMode.None;
            return;
        }

        UpdateControlMapping();
        var point = MapToScene(e.Location);
        if (_mode == PointerMode.BoxSelect && _boxCurrent is not null)
        {
            var selected = Session.BoxSelect(
                ScreenRect.FromPoints(_boxStart, _boxCurrent.Value),
                ActiveFilter,
                additive: (ModifierKeys & Keys.Control) != 0);
            StatusChanged?.Invoke($"Box select · {selected.Count} proxy(s) selected");
            Invalidate();
        }
        else if (_mode == PointerMode.DragProxy)
        {
            Session.EndDrag();
            var finalHit = Session.Pick(point, ActiveFilter);
            PreviewHit = finalHit;
            if (finalHit is not null)
            {
                HitCommitted?.Invoke(finalHit);
                StatusChanged?.Invoke($"Recorded final drag hit · {finalHit.StableId} face {finalHit.FaceIndex}");
            }
            else if (!_dragMoved && _downHit is not null)
            {
                PreviewHit = _downHit;
                HitCommitted?.Invoke(_downHit);
                StatusChanged?.Invoke($"Recorded click hit · {_downHit.StableId} face {_downHit.FaceIndex}");
            }

            Invalidate();
        }

        _mode = PointerMode.None;
        _downHit = null;
        _boxCurrent = null;
        Cursor = Cursors.Default;
    }

    protected override void OnMouseWheel(MouseEventArgs e)
    {
        base.OnMouseWheel(e);
        if (Session is null)
        {
            return;
        }

        UpdateControlMapping();
        var anchor = MapToScene(e.Location);
        if (!IsInsideSceneViewport(anchor))
        {
            return;
        }

        var factor = e.Delta > 0 ? 1.15 : 1 / 1.15;
        Session.ZoomAt(anchor, factor);
        StatusChanged?.Invoke($"Zoom {Session.ViewportTransform.Zoom:F2}×");
        Invalidate();
    }

    protected override bool ProcessCmdKey(ref Message msg, Keys keyData)
    {
        if (keyData == (Keys.Control | Keys.Z) && Session is not null)
        {
            if (Session.Undo(out var restored))
            {
                PreviewHit = null;
                StatusChanged?.Invoke($"Undo · {restored.Selection.Count} proxy(s) selected");
            }
            else
            {
                StatusChanged?.Invoke("Nothing to undo");
            }

            Invalidate();
            return true;
        }

        if (keyData == Keys.Escape && Session is not null)
        {
            if (Session.IsDragging)
            {
                Session.CancelDrag();
            }
            else
            {
                Session.ClearSelection();
            }

            PreviewHit = null;
            _mode = PointerMode.None;
            Invalidate();
            return true;
        }

        return base.ProcessCmdKey(ref msg, keyData);
    }

    private void DrawGrid(Graphics graphics)
    {
        using var pen = new Pen(Color.FromArgb(34, 44, 58), 1);
        const double spacing = 40;
        var x = _offsetX + ((double.IsFinite(_offsetX) ? _offsetX : 0) % spacing);
        for (var factor = 0; x + factor * spacing < Width; factor++)
        {
            var px = (float)(x + factor * spacing);
            graphics.DrawLine(pen, px, 0, px, Height);
        }

        var y = _offsetY + ((double.IsFinite(_offsetY) ? _offsetY : 0) % spacing);
        for (var factor = 0; y + factor * spacing < Height; factor++)
        {
            var py = (float)(y + factor * spacing);
            graphics.DrawLine(pen, 0, py, Width, py);
        }
    }

    private void DrawProxies(Graphics graphics)
    {
        if (Session is null)
        {
            return;
        }

        var scene = Session.Scene;
        var snapshot = Session.Snapshot();
        var basis = CameraBasis.FromCamera(scene.Camera);
        var triangles = new List<RenderTriangle>();
        foreach (var proxy in snapshot.Proxies)
        {
            var layer = snapshot.Layer(proxy.Definition.LayerId);
            if (!layer.Visible || !proxy.Definition.Visible)
            {
                continue;
            }

            var selected = Session.Selection.Contains(proxy.Definition.Id);
            var color = LayerColor(layer, proxy.Definition, selected);
            for (var face = 0; face < proxy.Definition.Geometry.TriangleCount; face++)
            {
                var triangle = proxy.Definition.Geometry.TriangleWorld(face, proxy.Transform);
                var a = basis.ProjectWorld(triangle.A, scene.Viewport, Session.ViewportTransform, scene.Camera.NearMeters);
                var b = basis.ProjectWorld(triangle.B, scene.Viewport, Session.ViewportTransform, scene.Camera.NearMeters);
                var c = basis.ProjectWorld(triangle.C, scene.Viewport, Session.ViewportTransform, scene.Camera.NearMeters);
                if (a is null || b is null || c is null)
                {
                    continue;
                }

                var points = new[]
                {
                    SceneToControl(a.Value.Screen),
                    SceneToControl(b.Value.Screen),
                    SceneToControl(c.Value.Screen)
                };
                triangles.Add(new RenderTriangle(points, (a.Value.Depth + b.Value.Depth + c.Value.Depth) / 3.0, color, selected));
            }
        }

        foreach (var triangle in triangles.OrderByDescending(t => t.Depth))
        {
            using var fill = new SolidBrush(Color.FromArgb(triangle.Selected ? 110 : 72, triangle.Color));
            using var outline = new Pen(triangle.Selected ? Color.FromArgb(245, 178, 70) : Color.FromArgb(175, triangle.Color), triangle.Selected ? 2f : 1f);
            graphics.FillPolygon(fill, triangle.Points);
            graphics.DrawPolygon(outline, triangle.Points);
        }
    }

    private void DrawPreview(Graphics graphics)
    {
        if (Session is null || PreviewHit is null)
        {
            return;
        }

        var point = SceneToControl(PreviewHit.ScreenPoint);
        using var marker = new Pen(Color.FromArgb(255, 214, 92), 2f);
        graphics.DrawLine(marker, point.X - 8, point.Y, point.X + 8, point.Y);
        graphics.DrawLine(marker, point.X, point.Y - 8, point.X, point.Y + 8);

        var scene = Session.Scene;
        var basis = CameraBasis.FromCamera(scene.Camera);
        var normalEnd = basis.ProjectWorld(
            PreviewHit.HitPointMeters + PreviewHit.FaceNormal * 0.7,
            scene.Viewport,
            Session.ViewportTransform,
            scene.Camera.NearMeters);
        if (normalEnd is not null)
        {
            using var normalPen = new Pen(Color.FromArgb(82, 214, 255), 2f);
            graphics.DrawLine(normalPen, point, SceneToControl(normalEnd.Value.Screen));
        }

        using var font = new Font("Segoe UI", 9f, FontStyle.Regular);
        using var textBrush = new SolidBrush(Color.FromArgb(235, 240, 248));
        using var background = new SolidBrush(Color.FromArgb(180, 20, 27, 37));
        var text = $"{PreviewHit.StableId}  face {PreviewHit.FaceIndex}  distance {PreviewHit.ScreenDistanceMeters:F3} m";
        var size = graphics.MeasureString(text, font);
        var rect = new RectangleF((float)_offsetX + 12, Height - 58, size.Width + 16, size.Height + 8);
        graphics.FillRectangle(background, rect);
        graphics.DrawString(text, font, textBrush, rect.X + 8, rect.Y + 4);
    }

    private void DrawRubberBand(Graphics graphics)
    {
        if (_mode != PointerMode.BoxSelect || _boxCurrent is null)
        {
            return;
        }

        var a = SceneToControl(_boxStart);
        var b = SceneToControl(_boxCurrent.Value);
        var rect = RectangleF.FromLTRB(System.Math.Min(a.X, b.X), System.Math.Min(a.Y, b.Y), System.Math.Max(a.X, b.X), System.Math.Max(a.Y, b.Y));
        using var fill = new SolidBrush(Color.FromArgb(36, 92, 196, 255));
        using var pen = new Pen(Color.FromArgb(210, 92, 196, 255), 1.5f) { DashStyle = DashStyle.Dash };
        graphics.FillRectangle(fill, rect);
        graphics.DrawRectangle(pen, rect.X, rect.Y, rect.Width, rect.Height);
    }

    private void DrawHud(Graphics graphics)
    {
        if (Session is null)
        {
            return;
        }

        using var font = new Font("Segoe UI", 9f);
        using var brush = new SolidBrush(Color.FromArgb(190, 214, 226, 240));
        var filter = ActiveFilter ?? "(any)";
        var text = $"filter {filter}   zoom {Session.ViewportTransform.Zoom:F2}×   pan {Session.ViewportTransform.PanXPixels:F0},{Session.ViewportTransform.PanYPixels:F0}   selected {Session.Selection.Count}";
        graphics.DrawString(text, font, brush, (float)_offsetX + 12, 10);
    }

    private void DrawCenteredText(Graphics graphics, string text)
    {
        using var font = new Font("Segoe UI", 14f);
        using var brush = new SolidBrush(Color.FromArgb(140, 150, 165));
        var size = graphics.MeasureString(text, font);
        graphics.DrawString(text, font, brush, (Width - size.Width) / 2, (Height - size.Height) / 2);
    }

    private void UpdateControlMapping()
    {
        if (Session is null)
        {
            return;
        }

        var sceneWidth = Session.Scene.Viewport.Width;
        var sceneHeight = Session.Scene.Viewport.Height;
        _controlScale = System.Math.Min(Width / (double)sceneWidth, Height / (double)sceneHeight);
        if (!double.IsFinite(_controlScale) || _controlScale <= 0)
        {
            _controlScale = 1;
        }

        _offsetX = (Width - sceneWidth * _controlScale) / 2.0;
        _offsetY = (Height - sceneHeight * _controlScale) / 2.0;
    }

    private Vec2 MapToScene(Point controlPoint) => new(
        (controlPoint.X - _offsetX) / _controlScale,
        (controlPoint.Y - _offsetY) / _controlScale);

    private PointF SceneToControl(Vec2 scenePoint) => new(
        (float)(_offsetX + scenePoint.X * _controlScale),
        (float)(_offsetY + scenePoint.Y * _controlScale));

    private bool IsInsideSceneViewport(Vec2 point) => Session is not null
        && point.X >= 0 && point.X < Session.Scene.Viewport.Width
        && point.Y >= 0 && point.Y < Session.Scene.Viewport.Height;

    private static Color LayerColor(LayerDefinition layer, ProxyDefinition proxy, bool selected)
    {
        if (selected)
        {
            return Color.FromArgb(245, 178, 70);
        }

        if (!layer.Pickable || !proxy.Pickable)
        {
            return Color.FromArgb(126, 137, 150);
        }

        return layer.PickFilter switch
        {
            "gizmo" => Color.FromArgb(76, 205, 224),
            _ => Color.FromArgb(86, 142, 235)
        };
    }

    private sealed record RenderTriangle(PointF[] Points, double Depth, Color Color, bool Selected);
}
