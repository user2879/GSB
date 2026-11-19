namespace LevelWorkbench.Core;

public readonly record struct Transform3(Vec3 PositionMeters, Vec3 RotationEulerDegrees, Vec3 Scale)
{
    public static Transform3 Identity { get; } = new(Vec3.Zero, Vec3.Zero, new Vec3(1, 1, 1));

    public Vec3 TransformPoint(Vec3 local)
    {
        var sx = local.X * Scale.X;
        var sy = local.Y * Scale.Y;
        var sz = local.Z * Scale.Z;

        var rx = RotationEulerDegrees.X * System.Math.PI / 180.0;
        var ry = RotationEulerDegrees.Y * System.Math.PI / 180.0;
        var rz = RotationEulerDegrees.Z * System.Math.PI / 180.0;

        var cx = System.Math.Cos(rx);
        var sxr = System.Math.Sin(rx);
        var cy = System.Math.Cos(ry);
        var syr = System.Math.Sin(ry);
        var cz = System.Math.Cos(rz);
        var szr = System.Math.Sin(rz);

        // Intrinsic X, then Y, then Z rotation: Rz * Ry * Rx.
        var y1 = sy * cx - sz * sxr;
        var z1 = sy * sxr + sz * cx;
        var x2 = sx * cy + z1 * syr;
        var z2 = -sx * syr + z1 * cy;
        var x3 = x2 * cz - y1 * szr;
        var y3 = x2 * szr + y1 * cz;

        return new Vec3(x3 + PositionMeters.X, y3 + PositionMeters.Y, z2 + PositionMeters.Z);
    }

    public Vec3 TransformDirection(Vec3 local) => TransformPoint(local) - TransformPoint(Vec3.Zero);
}
