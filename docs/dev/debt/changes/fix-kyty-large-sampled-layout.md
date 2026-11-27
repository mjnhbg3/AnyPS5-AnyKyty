# kyty

## add Functional

- [Persistent descriptor pools](../../../core/kyty/graphics/host_gpu/renderer/pipeline/descriptorHeap.cpp) remain allocated until the heap is destroyed. Table states can keep several sets for each program, so pool counts can grow until allocation fails under memory pressure. Closure requires measuring failed pool requests and reclaiming sets only after submitted GPU work no longer references them.
