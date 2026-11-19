namespace LevelWorkbench.Core;

public static class BoundedParallel
{
    public static async Task<IReadOnlyList<TResult>> RunAsync<TSource, TResult>(
        IReadOnlyList<TSource> source,
        Func<TSource, string> orderKey,
        Func<TSource, long> estimatedBytes,
        int concurrencyLimit,
        long memoryBudgetBytes,
        Func<TSource, CancellationToken, Task<TResult>> work,
        CancellationToken cancellationToken = default)
    {
        if (source.Count == 0)
        {
            return Array.Empty<TResult>();
        }

        if (concurrencyLimit <= 0)
        {
            throw new WorkbenchException(ErrorCodes.Resource, "Concurrency limit must be positive.");
        }

        var items = source
            .Select((item, index) => new WorkItem<TSource>(item, index, orderKey(item), estimatedBytes(item)))
            .OrderBy(i => i.OrderKey, StringComparer.Ordinal)
            .ThenBy(i => i.OriginalIndex)
            .Select((item, sequence) => item with { Sequence = sequence })
            .ToArray();

        var results = new TResult[source.Count];
        Exception? firstError = null;
        var firstErrorSequence = int.MaxValue;
        var errorGate = new object();
        using var memory = new MemoryBudget(memoryBudgetBytes);
        using var slots = new SemaphoreSlim(concurrencyLimit, concurrencyLimit);
        using var linked = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        var tasks = new List<Task>(items.Length);
        var stopped = false;

        foreach (var item in items)
        {
            if (linked.IsCancellationRequested)
            {
                stopped = true;
                break;
            }

            IDisposable? lease = null;
            var slotAcquired = false;
            try
            {
                lease = await memory.AcquireAsync(item.EstimatedBytes, linked.Token).ConfigureAwait(false);
                await slots.WaitAsync(linked.Token).ConfigureAwait(false);
                slotAcquired = true;
            }
            catch (OperationCanceledException) when (linked.IsCancellationRequested)
            {
                stopped = true;
                lease?.Dispose();
                break;
            }

            var capturedLease = lease;
            var capturedSlot = slotAcquired;
            if (linked.IsCancellationRequested)
            {
                capturedLease?.Dispose();
                if (capturedSlot)
                {
                    slots.Release();
                }

                stopped = true;
                break;
            }

            tasks.Add(Task.Run(async () =>
            {
                try
                {
                    results[item.OriginalIndex] = await work(item.Item, linked.Token).ConfigureAwait(false);
                }
                catch (Exception ex)
                {
                    if (ex is OperationCanceledException && linked.IsCancellationRequested)
                    {
                        return;
                    }

                    lock (errorGate)
                    {
                        if (item.Sequence < firstErrorSequence)
                        {
                            firstErrorSequence = item.Sequence;
                            firstError = ex;
                        }
                    }

                    linked.Cancel();
                }
                finally
                {
                    if (capturedSlot)
                    {
                        slots.Release();
                    }

                    capturedLease?.Dispose();
                }
            }, CancellationToken.None));
        }

        await Task.WhenAll(tasks).ConfigureAwait(false);

        if (firstError is not null)
        {
            if (firstError is WorkbenchException)
            {
                throw firstError;
            }

            throw new WorkbenchException(ErrorCodes.Internal, "A bounded parallel task failed.", new System.Text.Json.Nodes.JsonObject
            {
                ["reason"] = firstError.Message
            }, firstError);
        }

        if (cancellationToken.IsCancellationRequested)
        {
            throw new OperationCanceledException(cancellationToken);
        }

        if (stopped || linked.IsCancellationRequested)
        {
            throw new WorkbenchException(ErrorCodes.Cancelled, "Parallel work was cancelled after a task failure.");
        }

        return results;
    }

    private sealed record WorkItem<T>(T Item, int OriginalIndex, string OrderKey, long EstimatedBytes)
    {
        public int Sequence { get; init; }
    }
}
