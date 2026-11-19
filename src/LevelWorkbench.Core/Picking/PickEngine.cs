namespace LevelWorkbench.Core;

public sealed class PickEngine
{
    public PickHit? PickSerial(SceneSnapshot scene, Vec2 screenPoint, PickQuery query, CancellationToken cancellationToken = default)
    {
        var context = Prepare(scene, screenPoint, query, cancellationToken);
        if (context is null)
        {
            return null;
        }

        PickHit? best = null;
        foreach (var proxy in context.Candidates.OrderBy(p => p.Definition.StableId, StringComparer.Ordinal))
        {
            cancellationToken.ThrowIfCancellationRequested();
            var hit = IntersectProxy(proxy, context.Ray, screenPoint, query.ViewportTransform, context);
            if (hit is not null && IsBetter(hit, best))
            {
                best = hit;
            }
        }

        return best;
    }

    public async Task<PickHit?> PickParallelAsync(
        SceneSnapshot scene,
        Vec2 screenPoint,
        PickQuery query,
        CancellationToken cancellationToken = default)
    {
        var context = Prepare(scene, screenPoint, query, cancellationToken);
        if (context is null)
        {
            return null;
        }

        var candidates = context.Candidates.OrderBy(p => p.Definition.StableId, StringComparer.Ordinal).ToArray();
        var hits = await BoundedParallel.RunAsync(
            candidates,
            proxy => proxy.Definition.StableId,
            proxy => EstimateProxyBytes(proxy),
            query.ConcurrencyLimit ?? scene.Scene.ConcurrencyLimit,
            query.MemoryBudgetBytes ?? scene.Scene.MemoryBudgetBytes,
            (proxy, token) => Task.FromResult(IntersectProxy(proxy, context.Ray, screenPoint, query.ViewportTransform, context)),
            cancellationToken).ConfigureAwait(false);

        PickHit? best = null;
        foreach (var hit in hits)
        {
            if (hit is not null && IsBetter(hit, best))
            {
                best = hit;
            }
        }

        return best;
    }

    public static bool IsProxyPickable(ProxyInstanceState proxy, LayerDefinition layer, PickQuery query)
    {
        if (query.RespectVisibility && (!layer.Visible || !proxy.Definition.Visible))
        {
            return false;
        }

        if (!layer.Pickable || !proxy.Definition.Pickable)
        {
            return false;
        }

        if (query.Filter is not null)
        {
            return string.Equals(layer.PickFilter, query.Filter, StringComparison.Ordinal)
                && string.Equals(proxy.Definition.PickFilter, query.Filter, StringComparison.Ordinal);
        }

        return true;
    }

    private static PickContext? Prepare(SceneSnapshot scene, Vec2 screenPoint, PickQuery query, CancellationToken cancellationToken)
    {
        var viewport = query.ViewportOverride ?? scene.Scene.Viewport;
        if (!(screenPoint.X >= 0 && screenPoint.X < viewport.Width
              && screenPoint.Y >= 0 && screenPoint.Y < viewport.Height))
        {
            return null;
        }

        var basis = CameraBasis.FromCamera(scene.Scene.Camera);
        var ray = basis.ScreenRay(screenPoint, viewport, query.ViewportTransform);
        var candidates = scene.Proxies
            .Where(proxy => IsProxyPickable(proxy, scene.Layer(proxy.Definition.LayerId), query))
            .ToArray();
        return new PickContext(scene, query, ray, candidates, viewport, cancellationToken);
    }

    private static PickHit? IntersectProxy(
        ProxyInstanceState proxy,
        Ray3 ray,
        Vec2 screenPoint,
        ViewportTransform viewportTransform,
        PickContext context)
    {
        var geometry = proxy.Definition.Geometry;
        var near = context.Scene.Scene.Camera.NearMeters;
        var far = context.Scene.Scene.Camera.FarMeters;
        double? bestT = null;
        var bestFace = -1;
        Vec3 bestNormal = Vec3.Zero;

        for (var face = 0; face < geometry.TriangleCount; face++)
        {
            context.Token.ThrowIfCancellationRequested();
            var triangle = geometry.TriangleWorld(face, proxy.Transform);
            if (!TryIntersectTriangle(ray, triangle, near, far, out var t))
            {
                continue;
            }

            if (bestT is null || t < bestT.Value - 1e-12)
            {
                bestT = t;
                bestFace = face;
                bestNormal = triangle.UnitNormal();
            }
        }

        if (bestT is null || bestFace < 0)
        {
            return null;
        }

        var normal = bestNormal;
        if (Vec3.Dot(normal, ray.Direction) > 0)
        {
            normal = -normal;
        }

        var hitPoint = ray.PointAt(bestT.Value);
        var basePoint = viewportTransform.ScreenToViewport(screenPoint, context.Viewport);
        var layer = context.Scene.Layer(proxy.Definition.LayerId);
        return new PickHit
        {
            StableId = proxy.Definition.StableId,
            ProxyNumber = proxy.Definition.ProxyNumber,
            ProxyId = proxy.Definition.Id,
            LayerId = proxy.Definition.LayerId,
            FaceIndex = bestFace,
            HitPointMeters = hitPoint,
            FaceNormal = normal,
            ScreenDistanceMeters = bestT.Value,
            ScreenPoint = screenPoint,
            BaseViewportPoint = basePoint,
            Ray = ray,
            PickFilter = proxy.Definition.PickFilter,
            LayerVisible = layer.Visible,
            LayerPickable = layer.Pickable,
            ProxyVisible = proxy.Definition.Visible,
            ProxyPickable = proxy.Definition.Pickable,
            ViewportTransform = viewportTransform
        };
    }

    private static bool TryIntersectTriangle(Ray3 ray, Triangle3 triangle, double near, double far, out double t)
    {
        const double epsilon = 1e-10;
        var edge1 = triangle.B - triangle.A;
        var edge2 = triangle.C - triangle.A;
        var pvec = Vec3.Cross(ray.Direction, edge2);
        var determinant = Vec3.Dot(edge1, pvec);
        if (System.Math.Abs(determinant) <= epsilon)
        {
            t = 0;
            return false;
        }

        var invDet = 1.0 / determinant;
        var tvec = ray.Origin - triangle.A;
        var u = Vec3.Dot(tvec, pvec) * invDet;
        if (u < -epsilon || u > 1 + epsilon)
        {
            t = 0;
            return false;
        }

        var qvec = Vec3.Cross(tvec, edge1);
        var v = Vec3.Dot(ray.Direction, qvec) * invDet;
        if (v < -epsilon || u + v > 1 + epsilon)
        {
            t = 0;
            return false;
        }

        var distance = Vec3.Dot(edge2, qvec) * invDet;
        if (!double.IsFinite(distance) || distance < near || distance > far)
        {
            t = 0;
            return false;
        }

        t = distance;
        return true;
    }

    private static bool IsBetter(PickHit candidate, PickHit? current)
    {
        if (current is null)
        {
            return true;
        }

        var delta = candidate.ScreenDistanceMeters - current.ScreenDistanceMeters;
        if (System.Math.Abs(delta) > 1e-12)
        {
            return delta < 0;
        }

        return string.CompareOrdinal(candidate.StableId, current.StableId) < 0;
    }

    private static long EstimateProxyBytes(ProxyInstanceState proxy) =>
        proxy.Definition.Geometry.EstimatedTriangleBufferBytes + 512;

    private sealed record PickContext(
        SceneSnapshot Scene,
        PickQuery Query,
        Ray3 Ray,
        IReadOnlyList<ProxyInstanceState> Candidates,
        ViewportDefinition Viewport,
        CancellationToken Token);
}
