using System.Globalization;

namespace LevelWorkbench.Core;

public readonly record struct Vec2(double X, double Y)
{
    public bool IsFinite => double.IsFinite(X) && double.IsFinite(Y);

    public static Vec2 operator +(Vec2 a, Vec2 b) => new(a.X + b.X, a.Y + b.Y);

    public static Vec2 operator -(Vec2 a, Vec2 b) => new(a.X - b.X, a.Y - b.Y);

    public static Vec2 operator *(Vec2 a, double s) => new(a.X * s, a.Y * s);

    public static Vec2 operator /(Vec2 a, double s) => new(a.X / s, a.Y / s);

    public double Length => System.Math.Sqrt(X * X + Y * Y);

    public string ToInvariantString() =>
        FormattableString.Invariant($"({X.ToString("R", CultureInfo.InvariantCulture)},{Y.ToString("R", CultureInfo.InvariantCulture)})");
}
