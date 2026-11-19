using System.Globalization;

namespace LevelWorkbench.Core;

public readonly record struct Vec3(double X, double Y, double Z)
{
    public static readonly Vec3 Zero = new(0, 0, 0);
    public static readonly Vec3 UnitX = new(1, 0, 0);
    public static readonly Vec3 UnitY = new(0, 1, 0);
    public static readonly Vec3 UnitZ = new(0, 0, 1);

    public bool IsFinite => double.IsFinite(X) && double.IsFinite(Y) && double.IsFinite(Z);

    public double LengthSquared => X * X + Y * Y + Z * Z;

    public double Length => System.Math.Sqrt(LengthSquared);

    public static Vec3 operator +(Vec3 a, Vec3 b) => new(a.X + b.X, a.Y + b.Y, a.Z + b.Z);

    public static Vec3 operator -(Vec3 a, Vec3 b) => new(a.X - b.X, a.Y - b.Y, a.Z - b.Z);

    public static Vec3 operator -(Vec3 a) => new(-a.X, -a.Y, -a.Z);

    public static Vec3 operator *(Vec3 a, double s) => new(a.X * s, a.Y * s, a.Z * s);

    public static Vec3 operator *(double s, Vec3 a) => a * s;

    public static Vec3 operator /(Vec3 a, double s) => new(a.X / s, a.Y / s, a.Z / s);

    public static double Dot(Vec3 a, Vec3 b) => a.X * b.X + a.Y * b.Y + a.Z * b.Z;

    public static Vec3 Cross(Vec3 a, Vec3 b) => new(
        a.Y * b.Z - a.Z * b.Y,
        a.Z * b.X - a.X * b.Z,
        a.X * b.Y - a.Y * b.X);

    public Vec3 NormalizedOrZero()
    {
        var length = Length;
        return length > 1e-15 && double.IsFinite(length) ? this / length : Zero;
    }

    public Vec3 Normalized()
    {
        var length = Length;
        if (length <= 1e-15 || !double.IsFinite(length))
        {
            throw new WorkbenchException(ErrorCodes.Geometry, "Cannot normalize a zero or non-finite vector.");
        }

        return this / length;
    }

    public static double Distance(Vec3 a, Vec3 b) => (a - b).Length;

    public string ToInvariantString() => FormattableString.Invariant(
        $"({X.ToString("R", CultureInfo.InvariantCulture)},{Y.ToString("R", CultureInfo.InvariantCulture)},{Z.ToString("R", CultureInfo.InvariantCulture)})");
}
