# kyty

## add Functional

- [Device-local buffer spill](../../../core/kyty/graphics/host_gpu/renderer/cache/streamBuffer.cpp) can still occupy VRAM on Windows even when it uses a non-device-local memory type; that type does not guarantee physical placement. An out-of-budget device-local or stream buffer now retries on the spill allocator with that type filter, including after the spill allocator's budgeted attempt fails, and never retries the main allocator without its budget limit. Driver-reported device-local heap usage and cache eviction remain necessary under pressure. Closes when: driver-usage-based collection prevents VRAM exhaustion under the workload in [#315](https://github.com/simnJS/AnyPS5/issues/315).
- [Buffer budget enforcement](../../../core/kyty/graphics/host_gpu/renderer/cache/streamBuffer.cpp) relies on VMA's `WITHIN_BUDGET` flag before spill. A 64 MiB buffer with a synthetic 1 MiB reported heap budget still succeeded through VMA's automatic dedicated fallback, so the flag does not bound every main allocation. Closes when: budget-exceeding buffers are rejected before that fallback.
