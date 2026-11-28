# kyty

## add Functional

- [Normalized formatted-buffer conversion](../../../core/kyty/graphics/shader/recompiler/backend/spirv/spirvEmitterMemoryHelpers.cpp) produces a correctly rounded binary32 quotient of the integer code over the largest positive code, with the most negative SNORM code clamped to -1. This is the intended RDNA2 behavior in [#215](https://github.com/simnJS/AnyPS5/issues/215), but direct PS5/RDNA2 readback is unverified. The exhaustive GPU replay covers 8- and 16-bit inputs loaded into 32-bit destinations. D16 formatted loads subsequently pack that binary32 value with `PackHalf2x16`; their final half results, including possible double rounding, have not been compared with PS5/RDNA2 hardware. Packed image texels share the conversion helper; native Vulkan image-format conversions remain unverified.
