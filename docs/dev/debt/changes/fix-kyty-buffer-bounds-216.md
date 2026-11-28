# kyty

## add Functional

- [Buffer bounds in table and swizzled modes](../../../core/kyty/graphics/shader/recompiler/backend/spirv/spirvEmitterMemory.cpp) still rely on the Vulkan descriptor range for mode 3; table-mode lengths retain only whole dwords, so the exact trailing-byte count is unavailable. Mode 3 with nonzero stride needs native verification of `NUM_RECORDS` units before a direct bound can be trusted. The VGPR offset plus instruction immediate wraps at 32 bits on measured gfx1036 hardware, but PS5 gfx1013 and lane-ID carry remain unverified. Closes when mode 3 stride and swizzle cases and the table-mode trailing bytes have a native replay and an exact bound.
