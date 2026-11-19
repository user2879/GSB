namespace LevelWorkbench.Core;

public readonly record struct ViewportTransform(double Zoom, double PanXPixels, double PanYPixels)
{
    public static ViewportTransform Identity { get; } = new(1.0, 0.0, 0.0);

    public bool IsValid => double.IsFinite(Zoom) && Zoom > 0 && double.IsFinite(PanXPixels) && double.IsFinite(PanYPixels);

    public Vec2 ScreenToViewport(Vec2 screen, ViewportDefinition viewport) => new(
        (screen.X - PanXPixels - viewport.Width / 2.0) / Zoom + viewport.Width / 2.0,
        (screen.Y - PanYPixels - viewport.Height / 2.0) / Zoom + viewport.Height / 2.0);

    public Vec2 ViewportToScreen(Vec2 viewportPoint, ViewportDefinition viewport) => new(
        (viewportPoint.X - viewport.Width / 2.0) * Zoom + viewport.Width / 2.0 + PanXPixels,
        (viewportPoint.Y - viewport.Height / 2.0) * Zoom + viewport.Height / 2.0 + PanYPixels);

    public ViewportTransform ZoomAround(Vec2 screenAnchor, double factor, ViewportDefinition viewport)
    {
        if (!double.IsFinite(factor) || factor <= 0)
        {
            throw new WorkbenchException(ErrorCodes.Schema, "Zoom factor must be positive and finite.");
        }

        var anchorInViewport = ScreenToViewport(screenAnchor, viewport);
        var newZoom = Zoom * factor;
        var panX = screenAnchor.X - (anchorInViewport.X - viewport.Width / 2.0) * newZoom - viewport.Width / 2.0;
        var panY = screenAnchor.Y - (anchorInViewport.Y - viewport.Height / 2.0) * newZoom - viewport.Height / 2.0;
        return new ViewportTransform(newZoom, panX, panY);
    }
}

public readonly record struct CameraBasis(Vec3 Position, Vec3 Forward, Vec3 Right, Vec3 Up, double TanHalfVerticalFov)
{
    public static CameraBasis FromCamera(CameraDefinition camera)
    {
        var forward = (camera.TargetMeters - camera.PositionMeters).Normalized();
        var right = Vec3.Cross(forward, camera.Up).NormalizedOrZero();
        if (right.LengthSquared <= 1e-20)
        {
            throw new WorkbenchException(ErrorCodes.Schema, "Camera up vector is parallel to view direction.");
        }

        var up = Vec3.Cross(right, forward).Normalized();
        var tanHalf = System.Math.Tan(camera.VerticalFovDegrees * System.Math.PI / 360.0);
        return new CameraBasis(camera.PositionMeters, forward, right, up, tanHalf);
    }

    public Ray3 ScreenRay(Vec2 screen, ViewportDefinition viewport, ViewportTransform viewportTransform)
    {
        if (!(screen.X >= 0 && screen.X < viewport.Width && screen.Y >= 0 && screen.Y < viewport.Height))
        {
            throw new WorkbenchException(ErrorCodes.Schema, "Screen point lies outside the viewport.", new System.Text.Json.Nodes.JsonObject
            {
                ["x"] = screen.X,
                ["y"] = screen.Y,
                ["width"] = viewport.Width,
                ["height"] = viewport.Height
            });
        }

        if (!viewportTransform.IsValid)
        {
            throw new WorkbenchException(ErrorCodes.Schema, "Viewport transform is invalid.");
        }

        var vp = viewportTransform.ScreenToViewport(screen, viewport);
        var ndcX = vp.X / viewport.Width * 2.0 - 1.0;
        var ndcY = 1.0 - vp.Y / viewport.Height * 2.0;
        var aspect = viewport.Width / (double)viewport.Height;
        var direction = (Forward + Right * (ndcX * TanHalfVerticalFov * aspect) + Up * (ndcY * TanHalfVerticalFov)).Normalized();
        return new Ray3(Position, direction);
    }

    public ScreenProjection? ProjectWorld(Vec3 world, ViewportDefinition viewport, ViewportTransform transform, double nearMeters)
    {
        var relative = world - Position;
        var depth = Vec3.Dot(relative, Forward);
        if (depth <= nearMeters || !double.IsFinite(depth))
        {
            return null;
        }

        var aspect = viewport.Width / (double)viewport.Height;
        var ndcX = Vec3.Dot(relative, Right) / depth / (TanHalfVerticalFov * aspect);
        var ndcY = Vec3.Dot(relative, Up) / depth / TanHalfVerticalFov;
        var viewportPoint = new Vec2(
            (ndcX + 1.0) * 0.5 * viewport.Width,
            (1.0 - ndcY) * 0.5 * viewport.Height);
        var screen = transform.ViewportToScreen(viewportPoint, viewport);
        return new ScreenProjection(screen, viewportPoint, depth);
    }
}

public readonly record struct ScreenProjection(Vec2 Screen, Vec2 BaseViewport, double Depth);
