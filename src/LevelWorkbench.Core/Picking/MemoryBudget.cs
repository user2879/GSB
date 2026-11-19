namespace LevelWorkbench.Core;

public sealed class MemoryBudget : IDisposable
{
    private readonly object _gate = new();
    private readonly Queue<Waiter> _waiters = new();
    private long _used;
    private bool _disposed;

    public MemoryBudget(long limitBytes)
    {
        if (limitBytes <= 0)
        {
            throw new WorkbenchException(ErrorCodes.Resource, "Memory budget must be positive.");
        }

        LimitBytes = limitBytes;
    }

    public long LimitBytes { get; }

    public async Task<IDisposable> AcquireAsync(long bytes, CancellationToken cancellationToken)
    {
        if (bytes < 0 || bytes > LimitBytes)
        {
            throw new WorkbenchException(ErrorCodes.Resource, "A single task exceeds the configured memory budget.", new System.Text.Json.Nodes.JsonObject
            {
                ["requestedBytes"] = bytes,
                ["memoryBudgetBytes"] = LimitBytes
            });
        }

        Waiter waiter;
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_waiters.Count == 0 && _used + bytes <= LimitBytes)
            {
                _used += bytes;
                return new Lease(this, bytes);
            }

            waiter = new Waiter(bytes);
            _waiters.Enqueue(waiter);
        }

        if (!cancellationToken.CanBeCanceled)
        {
            return await waiter.Task.Task.ConfigureAwait(false);
        }

        using var registration = cancellationToken.Register(() => CancelWaiter(waiter, cancellationToken));
        return await waiter.Task.Task.ConfigureAwait(false);
    }

    private void Release(long bytes)
    {
        lock (_gate)
        {
            _used -= bytes;
            if (_used < 0)
            {
                _used = 0;
            }

            while (_waiters.Count > 0)
            {
                var waiter = _waiters.Peek();
                if (waiter.Task.Task.IsCompleted)
                {
                    _waiters.Dequeue();
                    continue;
                }

                if (_used + waiter.Bytes > LimitBytes)
                {
                    break;
                }

                _waiters.Dequeue();
                _used += waiter.Bytes;
                waiter.Task.TrySetResult(new Lease(this, waiter.Bytes));
            }
        }
    }

    private void CancelWaiter(Waiter waiter, CancellationToken cancellationToken)
    {
        lock (_gate)
        {
            if (waiter.Task.Task.IsCompleted)
            {
                return;
            }

            waiter.Task.TrySetCanceled(cancellationToken);
            if (_waiters.Count > 0 && ReferenceEquals(_waiters.Peek(), waiter))
            {
                _waiters.Dequeue();
                GrantAvailable();
            }
        }
    }

    private void GrantAvailable()
    {
        while (_waiters.Count > 0)
        {
            var waiter = _waiters.Peek();
            if (waiter.Task.Task.IsCompleted)
            {
                _waiters.Dequeue();
                continue;
            }

            if (_used + waiter.Bytes > LimitBytes)
            {
                return;
            }

            _waiters.Dequeue();
            _used += waiter.Bytes;
            waiter.Task.TrySetResult(new Lease(this, waiter.Bytes));
        }
    }

    public void Dispose()
    {
        List<Waiter> pending;
        lock (_gate)
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            pending = _waiters.ToList();
            _waiters.Clear();
        }

        foreach (var waiter in pending)
        {
            waiter.Task.TrySetCanceled();
        }
    }

    private sealed class Waiter
    {
        public Waiter(long bytes)
        {
            Bytes = bytes;
        }

        public long Bytes { get; }

        public TaskCompletionSource<IDisposable> Task { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
    }

    private sealed class Lease : IDisposable
    {
        private MemoryBudget? _owner;
        private readonly long _bytes;

        public Lease(MemoryBudget owner, long bytes)
        {
            _owner = owner;
            _bytes = bytes;
        }

        public void Dispose()
        {
            var owner = Interlocked.Exchange(ref _owner, null);
            owner?.Release(_bytes);
        }
    }
}
