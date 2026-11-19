using System.Security.Cryptography;
using System.Text;

namespace LevelWorkbench.Core;

public static class Sha256Util
{
    public const string Prefix = "sha256:";

    public static string HashHex(string text) => HashHex(Encoding.UTF8.GetBytes(text));

    public static string HashHex(ReadOnlySpan<byte> bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();

    public static string HashWithPrefix(string text) => Prefix + HashHex(text);

    public static string HashBytesWithPrefix(ReadOnlySpan<byte> bytes) => Prefix + HashHex(bytes);

    public static string HashFile(string path)
    {
        using var stream = File.OpenRead(path);
        return Prefix + Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }

    public static bool IsDigest(string? value)
    {
        if (value is null || !value.StartsWith(Prefix, StringComparison.Ordinal))
        {
            return false;
        }

        var hex = value[Prefix.Length..];
        return hex.Length == 64 && hex.All(c => (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'));
    }
}
