namespace LevelWorkbench.Core;

public readonly record struct ScreenRect(double MinX, double MinY, double MaxX, double MaxY)
{
    public static ScreenRect FromPoints(Vec2 a, Vec2 b) => new(
        System.Math.Min(a.X, b.X),
        System.Math.Min(a.Y, b.Y),
        System.Math.Max(a.X, b.X),
        System.Math.Max(a.Y, b.Y));

    public bool Contains(Vec2 point) => point.X >= MinX && point.X <= MaxX && point.Y >= MinY && point.Y <= MaxY;

    public bool Intersects(double minX, double minY, double maxX, double maxY) =>
        maxX >= MinX && minX <= MaxX && maxY >= MinY && minY <= MaxY;
}

public sealed class EditorSnapshot
{
    public required IReadOnlyDictionary<string, Transform3> Poses { get; init; }
    public required IReadOnlySet<string> Selection { get; init; }
    public required ViewportTransform ViewportTransform { get; init; }
}

public sealed class EditorSession
{
    private readonly SceneDocument _scene;
    private readonly Dictionary<string, Transform3> _poses;
    private readonly HashSet<string> _selection = new(StringComparer.Ordinal);
    private readonly Stack<EditorSnapshot> _undo = new();
    private readonly PickEngine _pickEngine = new();
    private DragOperation? _drag;

    public EditorSession(SceneDocument scene)
    {
        _scene = scene;
        _poses = scene.Proxies.ToDictionary(proxy => proxy.Id, proxy => proxy.Transform, StringComparer.Ordinal);
        ViewportTransform = ViewportTransform.Identity;
    }

    public SceneDocument Scene => _scene;

    public ViewportTransform ViewportTransform { get; private set; }

    public IReadOnlyCollection<string> Selection => _selection;

    public bool IsDragging => _drag is not null;

    public SceneSnapshot Snapshot() => SceneSnapshot.FromPoses(_scene, _poses);

    public Transform3 PoseOf(string proxyId) => _poses[proxyId];

    public IReadOnlyDictionary<string, Transform3> Poses => _poses;

    public void SetViewportTransform(ViewportTransform transform)
    {
        if (!transform.IsValid)
        {
            throw new WorkbenchException(ErrorCodes.Schema, "Viewport transform is invalid.");
        }

        ViewportTransform = transform;
    }

    public void ZoomAt(Vec2 screenAnchor, double factor) =>
        ViewportTransform = ViewportTransform.ZoomAround(screenAnchor, factor, _scene.Viewport);

    public void PanBy(Vec2 delta)
    {
        if (!delta.IsFinite)
        {
            throw new WorkbenchException(ErrorCodes.Schema, "Pan delta must be finite.");
        }

        ViewportTransform = new ViewportTransform(
            ViewportTransform.Zoom,
            ViewportTransform.PanXPixels + delta.X,
            ViewportTransform.PanYPixels + delta.Y);
    }

    public PickHit? Pick(Vec2 screen, string? filter = null, bool parallel = true, CancellationToken cancellationToken = default)
    {
        EnsureNotDragging();
        var query = Query(filter);
        return parallel
            ? _pickEngine.PickParallelAsync(Snapshot(), screen, query, cancellationToken).GetAwaiter().GetResult()
            : _pickEngine.PickSerial(Snapshot(), screen, query, cancellationToken);
    }

    public async Task<PickHit?> PickAsync(Vec2 screen, string? filter = null, CancellationToken cancellationToken = default)
    {
        EnsureNotDragging();
        return await _pickEngine.PickParallelAsync(Snapshot(), screen, Query(filter), cancellationToken).ConfigureAwait(false);
    }

    public PickHit? SelectAt(Vec2 screen, string? filter, bool additive)
    {
        EnsureNotDragging();
        var before = Capture();
        var hit = _pickEngine.PickParallelAsync(Snapshot(), screen, Query(filter), CancellationToken.None)
            .GetAwaiter().GetResult();
        if (!additive)
        {
            _selection.Clear();
        }

        if (hit is not null)
        {
            if (additive && _selection.Contains(hit.ProxyId))
            {
                _selection.Remove(hit.ProxyId);
            }
            else
            {
                _selection.Add(hit.ProxyId);
            }
        }

        PushIfChanged(before);
        return hit;
    }

    public IReadOnlyCollection<string> BoxSelect(ScreenRect rect, string? filter, bool additive)
    {
        EnsureNotDragging();
        var before = Capture();
        var basis = CameraBasis.FromCamera(_scene.Camera);
        var selected = new HashSet<string>(StringComparer.Ordinal);
        var query = Query(filter);
        var snapshot = Snapshot();
        foreach (var proxy in snapshot.Proxies.OrderBy(p => p.Definition.StableId, StringComparer.Ordinal))
        {
            if (!PickEngine.IsProxyPickable(proxy, snapshot.Layer(proxy.Definition.LayerId), query))
            {
                continue;
            }

            var projected = proxy.Definition.Geometry.Vertices
                .Select(vertex => basis.ProjectWorld(
                    proxy.Transform.TransformPoint(vertex),
                    _scene.Viewport,
                    ViewportTransform,
                    _scene.Camera.NearMeters))
                .Where(point => point is not null)
                .Select(point => point!.Value.Screen)
                .ToArray();
            if (projected.Length == 0)
            {
                continue;
            }

            var minX = projected.Min(p => p.X);
            var minY = projected.Min(p => p.Y);
            var maxX = projected.Max(p => p.X);
            var maxY = projected.Max(p => p.Y);
            if (rect.Intersects(minX, minY, maxX, maxY))
            {
                selected.Add(proxy.Definition.Id);
            }
        }

        if (!additive)
        {
            _selection.Clear();
        }

        _selection.UnionWith(selected);
        PushIfChanged(before);
        return _selection.ToArray();
    }

    public void BeginDrag(PickHit hit, Vec2 startScreen, string? filter, bool additiveSelection)
    {
        EnsureNotDragging();
        var before = Capture();
        if (!_selection.Contains(hit.ProxyId))
        {
            if (!additiveSelection)
            {
                _selection.Clear();
            }

            _selection.Add(hit.ProxyId);
        }

        var basis = CameraBasis.FromCamera(_scene.Camera);
        var plane = new Plane3(hit.HitPointMeters, basis.Forward);
        _drag = new DragOperation
        {
            Before = before,
            Plane = plane,
            StartScreen = startScreen,
            Filter = filter,
            StartPlanePoint = hit.HitPointMeters,
            StartPoses = _selection.ToDictionary(id => id, id => _poses[id], StringComparer.Ordinal)
        };
    }

    public PickHit? UpdateDrag(Vec2 screen)
    {
        if (_drag is null)
        {
            throw new WorkbenchException(ErrorCodes.Schema, "No drag operation is active.");
        }

        if (!(screen.X >= 0 && screen.X < _scene.Viewport.Width && screen.Y >= 0 && screen.Y < _scene.Viewport.Height))
        {
            return null;
        }

        var basis = CameraBasis.FromCamera(_scene.Camera);
        var ray = basis.ScreenRay(screen, _scene.Viewport, ViewportTransform);
        var t = _drag.Plane.Intersect(ray);
        if (t is null || t.Value < 0)
        {
            return null;
        }

        var planePoint = ray.PointAt(t.Value);
        var delta = planePoint - _drag.StartPlanePoint;
        foreach (var (proxyId, original) in _drag.StartPoses)
        {
            _poses[proxyId] = original with { PositionMeters = original.PositionMeters + delta };
        }

        _drag.LastScreen = screen;
        return _pickEngine.PickParallelAsync(Snapshot(), screen, Query(_drag.Filter), CancellationToken.None)
            .GetAwaiter().GetResult();
    }

    public void EndDrag()
    {
        if (_drag is null)
        {
            return;
        }

        var before = _drag.Before;
        _drag = null;
        PushIfChanged(before, force: true);
    }

    public void CancelDrag()
    {
        if (_drag is null)
        {
            return;
        }

        Restore(_drag.Before);
        _drag = null;
    }

    public bool Undo(out EditorSnapshot restored)
    {
        EnsureNotDragging();
        if (_undo.Count == 0)
        {
            restored = Capture();
            return false;
        }

        restored = _undo.Pop();
        Restore(restored);
        return true;
    }

    public void ClearSelection()
    {
        EnsureNotDragging();
        var before = Capture();
        _selection.Clear();
        PushIfChanged(before);
    }

    public void SetSelection(IEnumerable<string> proxyIds)
    {
        EnsureNotDragging();
        var before = Capture();
        _selection.Clear();
        foreach (var proxyId in proxyIds)
        {
            if (_scene.Proxies.Any(p => p.Id == proxyId))
            {
                _selection.Add(proxyId);
            }
        }

        PushIfChanged(before);
    }

    public void RestoreProxyPoseForRecovery(string proxyId, Transform3 transform)
    {
        if (!_poses.ContainsKey(proxyId))
        {
            throw new WorkbenchException(ErrorCodes.State, "Persisted editor state references an unknown proxy.", new System.Text.Json.Nodes.JsonObject
            {
                ["proxyId"] = proxyId
            });
        }

        _poses[proxyId] = transform;
    }

    public void RestoreSelectionForRecovery(IEnumerable<string> proxyIds)
    {
        _selection.Clear();
        foreach (var proxyId in proxyIds)
        {
            if (_scene.Proxies.Any(proxy => proxy.Id == proxyId))
            {
                _selection.Add(proxyId);
            }
        }
    }

    private PickQuery Query(string? filter) => new()
    {
        Filter = filter,
        ViewportTransform = ViewportTransform
    };

    private EditorSnapshot Capture() => new()
    {
        Poses = _poses.ToDictionary(pair => pair.Key, pair => pair.Value, StringComparer.Ordinal),
        Selection = new HashSet<string>(_selection, StringComparer.Ordinal),
        ViewportTransform = ViewportTransform
    };

    private void Restore(EditorSnapshot snapshot)
    {
        _poses.Clear();
        foreach (var (id, pose) in snapshot.Poses)
        {
            _poses[id] = pose;
        }

        _selection.Clear();
        _selection.UnionWith(snapshot.Selection);
        ViewportTransform = snapshot.ViewportTransform;
    }

    private void PushIfChanged(EditorSnapshot before, bool force = false)
    {
        if (force || !PosesEqual(before) || !before.Selection.SetEquals(_selection) || before.ViewportTransform != ViewportTransform)
        {
            _undo.Push(before);
        }
    }

    private bool PosesEqual(EditorSnapshot snapshot) => snapshot.Poses.Count == _poses.Count
        && snapshot.Poses.All(pair => _poses.TryGetValue(pair.Key, out var current) && current == pair.Value);

    private void EnsureNotDragging()
    {
        if (_drag is not null)
        {
            throw new WorkbenchException(ErrorCodes.Schema, "Operation is not allowed while a drag is active.");
        }
    }

    private sealed class DragOperation
    {
        public required EditorSnapshot Before { get; init; }
        public required Plane3 Plane { get; init; }
        public required Vec2 StartScreen { get; init; }
        public required Vec3 StartPlanePoint { get; init; }
        public required IReadOnlyDictionary<string, Transform3> StartPoses { get; init; }
        public required string? Filter { get; init; }
        public Vec2 LastScreen { get; set; }
    }
}
