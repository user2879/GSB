using System.Text;

namespace LevelWorkbench.Core;

public static class AtomicFile
{
    public static void WriteAllText(string path, string content, string businessCode = ErrorCodes.Write)
        => WriteAllBytes(path, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false).GetBytes(content), businessCode);

    public static void WriteAllBytes(string path, byte[] bytes, string businessCode = ErrorCodes.Write)
    {
        var fullPath = Path.GetFullPath(path);
        var directory = Path.GetDirectoryName(fullPath)
                        ?? throw new WorkbenchException(businessCode, "Target path has no directory.", new System.Text.Json.Nodes.JsonObject
                        {
                            ["path"] = path
                        });
        var tempPath = Path.Combine(directory, $".{Path.GetFileName(fullPath)}.{Guid.NewGuid():N}.tmp");
        try
        {
            Directory.CreateDirectory(directory);
            using (var stream = new FileStream(tempPath, FileMode.CreateNew, FileAccess.Write, FileShare.None, 4096, FileOptions.WriteThrough))
            {
                stream.Write(bytes, 0, bytes.Length);
                stream.Flush(flushToDisk: true);
            }

            File.Move(tempPath, fullPath, overwrite: true);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or NotSupportedException or ArgumentException)
        {
            TryDelete(tempPath);
            throw new WorkbenchException(businessCode, "Atomic file write failed; the previous file was preserved.", new System.Text.Json.Nodes.JsonObject
            {
                ["path"] = fullPath,
                ["reason"] = ex.Message
            }, ex);
        }
    }

    private static void TryDelete(string path)
    {
        try
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
        catch
        {
            // Preserve the original error; a stale temp file is safe and ignored by readers.
        }
    }
}
