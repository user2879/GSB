namespace LevelWorkbench.Core;

public readonly record struct Ray3(Vec3 Origin, Vec3 Direction)
{
    public Vec3 PointAt(double t) => Origin + Direction * t;
}

public readonly record struct Triangle3(Vec3 A, Vec3 B, Vec3 C)
{
    public Vec3 GeometricNormal() => Vec3.Cross(B - A, C - A);

    public Vec3 UnitNormal()
    {
        var normal = GeometricNormal();
        if (normal.LengthSquared <= 1e-24)
        {
            return Vec3.Zero;
        }

        return normal.NormalizedOrZero();
    }
}

public readonly record struct Plane3(Vec3 Point, Vec3 Normal)
{
    public double? Intersect(Ray3 ray)
    {
        var denom = Vec3.Dot(Normal, ray.Direction);
        if (System.Math.Abs(denom) <= 1e-12)
        {
            return null;
        }

        var t = Vec3.Dot(Point - ray.Origin, Normal) / denom;
        return double.IsFinite(t) ? t : null;
    }
}

public readonly record struct Bounds3(Vec3 Min, Vec3 Max)
{
    public static Bounds3 FromPoints(IEnumerable<Vec3> points)
    {
        using var enumerator = points.GetEnumerator();
        if (!enumerator.MoveNext())
        {
            return new Bounds3(Vec3.Zero, Vec3.Zero);
        }

        var first = enumerator.Current;
        var min = first;
        var max = first;
        while (enumerator.MoveNext())
        {
            var p = enumerator.Current;
            min = new Vec3(System.Math.Min(min.X, p.X), System.Math.Min(min.Y, p.Y), System.Math.Min(min.Z, p.Z));
            max = new Vec3(System.Math.Max(max.X, p.X), System.Math.Max(max.Y, p.Y), System.Math.Max(max.Z, p.Z));
        }

        return new Bounds3(min, max);
    }
}
