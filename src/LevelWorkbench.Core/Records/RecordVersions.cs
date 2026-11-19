namespace LevelWorkbench.Core;

public static class RecordVersions
{
    public const int Legacy = 1;
    public const int StableIdentity = 2;
    public const int Current = 3;

    public static readonly IReadOnlySet<int> Known = new HashSet<int> { Legacy, StableIdentity, Current };

    public static bool IsKnown(int version) => Known.Contains(version);
}
