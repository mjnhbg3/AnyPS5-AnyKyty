#include "TestVulkanDevice.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/pipeline/descriptorHeap.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

using namespace Libs::Graphics;

constexpr std::array<uint32_t, 35> ComputeSpirv {
    0x07230203u, 0x00010000u, 0u, 5u, 0u,
    0x00020011u, 1u,
    0x0003000eu, 0u, 1u,
    0x0005000fu, 5u, 3u, 0x6e69616du, 0u,
    0x00060010u, 3u, 17u, 1u, 1u, 1u,
    0x00020013u, 1u,
    0x00030021u, 2u, 1u,
    0x00050036u, 1u, 3u, 0u, 2u,
    0x000200f8u, 4u,
    0x000100fdu,
    0x00010038u,
};

void DestroyPipeline(GraphicContext& graphics, PipelineCache::Pipeline& pipeline) {
    graphics.device.destroyPipeline(pipeline.pipeline);
    graphics.device.destroyPipelineLayout(pipeline.pipeline_layout);
    graphics.device.destroyDescriptorSetLayout(pipeline.descriptor_set_layout);
}

const char* CreateAndAllocate(GraphicContext& graphics, DescriptorHeap& heap,
                              ShaderModule& module, PipelineCache::Pipeline& pipeline,
                              uint32_t count, bool persistent) {
    module.info.bindings.descriptor_counts[ShaderRecompiler::IR::FirstImageBinding] = count;
    if (CreateComputePipeline(graphics, pipeline, module, nullptr, false) != vk::Result::eSuccess ||
        pipeline.pipeline == nullptr || pipeline.descriptor_set_layout == nullptr ||
        pipeline.uses_push_descriptors || pipeline.native_bindings) {
        return "the compute pipeline did not create an allocatable descriptor layout";
    }
    if (heap.Commit(pipeline.descriptor_set_layout) == nullptr) {
        return "transient descriptor allocation returned a null set";
    }
    if (persistent && heap.CommitPersistent(pipeline.descriptor_set_layout) == nullptr) {
        return "persistent descriptor allocation returned a null set";
    }
    return nullptr;
}

}

int main() {
    KytyShimTests::TestVulkanDevice gpu;
    if (const auto* unavailable = gpu.Initialize()) {
        const bool required = std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr;
        std::fprintf(stderr, "kyty_shim_descriptor_pool_capacity: %s, no usable Vulkan device (%s)\n",
                     required ? "FAILED" : "skipped", unavailable);
        return required ? 1 : 77;
    }

    constexpr uint32_t BeyondDefault = DescriptorHeap::MaxSampledImages + 16u;
    const auto& limits = gpu.graphics.GetPhysicalDeviceProperties().limits;
    if (std::min({limits.maxDescriptorSetSampledImages, limits.maxPerStageDescriptorSampledImages,
                  limits.maxPerStageResources}) < BeyondDefault) {
        std::fprintf(stderr,
                     "kyty_shim_descriptor_pool_capacity: skipped, the device reports a sampled-image or stage-resource limit below %u\n",
                     BeyondDefault);
        return 77;
    }

    const vk::ShaderModuleCreateInfo create_module {
        .codeSize = ComputeSpirv.size() * sizeof(uint32_t), .pCode = ComputeSpirv.data()};
    ShaderModule module;
    if (gpu.graphics.device.createShaderModule(&create_module, nullptr, &module.module) != vk::Result::eSuccess) {
        std::fputs("kyty_shim_descriptor_pool_capacity: FAILED: could not create compute module\n", stderr);
        return 1;
    }
    module.info.stage = ShaderType::Compute;
    module.info.bindings.descriptor_mask = uint64_t {1} << ShaderRecompiler::IR::FirstImageBinding;

    PipelineCache::Pipeline at_default;
    PipelineCache::Pipeline beyond_default;
    const char* failure = nullptr;
    {
        MasterSemaphore master(gpu.graphics);
        const vk::SemaphoreSignalInfo signal {.semaphore = master.Handle(), .value = master.CurrentTick()};
        if (gpu.graphics.device.signalSemaphore(&signal) != vk::Result::eSuccess) {
            failure = "could not signal the descriptor retirement semaphore";
        } else {
            DescriptorHeap heap(gpu.graphics, master);
            failure = CreateAndAllocate(gpu.graphics, heap, module, at_default,
                                        DescriptorHeap::MaxSampledImages, false);
            if (failure == nullptr) {
                failure = CreateAndAllocate(gpu.graphics, heap, module, beyond_default, BeyondDefault, true);
            }
        }
    }
    DestroyPipeline(gpu.graphics, at_default);
    DestroyPipeline(gpu.graphics, beyond_default);
    gpu.graphics.device.destroyShaderModule(module.module);

    if (failure != nullptr) {
        std::fprintf(stderr, "kyty_shim_descriptor_pool_capacity: FAILED: %s\n", failure);
        return 1;
    }
    std::puts("kyty_shim_descriptor_pool_capacity: ok");
    return 0;
}
