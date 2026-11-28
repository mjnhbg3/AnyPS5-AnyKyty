#include "prx/libSceAgcDriver/Execution/include/FrameCapture.hpp"

#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint64_t GuestBase = 0x202000000;
constexpr std::uint64_t Commands = GuestBase + 0x5000;
constexpr std::uint64_t Input8 = GuestBase + 0x10000;
constexpr std::uint64_t Input16 = GuestBase + 0x20000;
constexpr std::uint64_t Output = GuestBase + 0x40000;
constexpr std::uint64_t Display = GuestBase + 0xd0000;
constexpr std::uint64_t GuestBytes = 0xe0000;
constexpr std::uint32_t LocalLanes = 64;
constexpr std::uint32_t Format32UInt = 20;
constexpr std::uint32_t Fill = 0xdeadbeefu;
constexpr std::uint32_t Values8 = 1u << 8u;
constexpr std::uint32_t Values16 = 1u << 16u;
constexpr std::uint64_t OutputBytes = 2ull * (Values8 + Values16) * sizeof(std::uint32_t);

struct Case {
    const char* name;
    std::uint32_t format;
    std::uint32_t bits;
    bool signedValues;
    std::uint32_t values;
    std::uint64_t input;
    std::uint64_t output;
    std::uint64_t header;
    std::uint64_t code;
};

constexpr std::array Cases{
    Case{"8 UNORM", 1u, 8u, false, Values8, Input8, Output, GuestBase + 0x100, GuestBase + 0x800},
    Case{"8 SNORM", 2u, 8u, true, Values8, Input8, Output + Values8 * 4u, GuestBase + 0x1100, GuestBase + 0x1800},
    Case{"16 UNORM", 7u, 16u, false, Values16, Input16, Output + 2u * Values8 * 4u, GuestBase + 0x2100,
         GuestBase + 0x2800},
    Case{"16 SNORM", 8u, 16u, true, Values16, Input16, Output + (2u * Values8 + Values16) * 4u,
         GuestBase + 0x3100, GuestBase + 0x3800},
};

using Sharp = std::array<std::uint32_t, 4>;

std::uint32_t low(std::uint64_t address) {
    return static_cast<std::uint32_t>(address >> 8u);
}

std::uint32_t high(std::uint64_t address) {
    return static_cast<std::uint32_t>(address >> 40u);
}

Sharp sharp(std::uint64_t address, std::uint32_t bytes, std::uint32_t format) {
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes,
            (0x01016facu & ~(0x7fu << 12u)) | (format << 12u) | (3u << 28u)};
}

std::array<std::uint32_t, 10> shader(const Case& test) {
    return {0x8f098608u, 0x4a000009u, test.bits == 8u ? 0x7e020300u : 0x34020081u,
            0xe0001000u, 0x80000201u, 0xbf8c0000u, 0x34020082u, 0xe0701000u,
            0x80010201u, 0xbf810000u};
}

void append(std::vector<std::uint32_t>& words, std::uint32_t opcode, std::initializer_list<std::uint32_t> payload) {
    words.push_back(0xc0000000u | (static_cast<std::uint32_t>(payload.size() - 1) << 16u) | (opcode << 8u));
    words.insert(words.end(), payload);
}

std::vector<std::uint32_t> dispatchWords() {
    std::vector<std::uint32_t> words;
    for (const auto& test: Cases) {
        const auto source = sharp(test.input, test.values * (test.bits / 8u), test.format);
        const auto target = sharp(test.output, test.values * sizeof(std::uint32_t), Format32UInt);
        append(words, 0x76, {0x20c, low(test.code), high(test.code)});
        append(words, 0x76, {0x212, 0u});
        append(words, 0x76, {0x213, (8u << 1u) | (1u << 7u)});
        append(words, 0x76, {0x207, LocalLanes, 1u, 1u});
        append(words, 0x76, {0x240, source[0], source[1], source[2], source[3], target[0], target[1], target[2], target[3]});
        append(words, 0x15, {test.values / LocalLanes, 1u, 1u, 0x41u});
    }
    return words;
}

void write(std::span<std::byte> memory, std::uint64_t address, const void* data, std::size_t bytes) {
    const auto offset = address - GuestBase;
    if (offset > memory.size() || bytes > memory.size() - offset) throw std::runtime_error("fixture memory write past guest mapping");
    std::memcpy(memory.data() + offset, data, bytes);
}

void writeShader(std::span<std::byte> memory, const Case& test) {
    struct Header {
        Shader shader{};
        ShaderUserData user{};
    };
    const auto words = shader(test);
    Header header{};
    header.shader.file_header = 0x34333231;
    header.shader.version = 0x18;
    header.shader.header_size = sizeof(Header);
    header.shader.shader_size = static_cast<std::uint32_t>(sizeof(words));
    header.shader.code = reinterpret_cast<const void*>(test.code);
    header.shader.type = 0;
    header.shader.user_data = reinterpret_cast<ShaderUserData*>(test.header + offsetof(Header, user));
    write(memory, test.header, &header, sizeof(header));
    write(memory, test.code, words.data(), sizeof(words));
}

void makeCapture(const std::filesystem::path& path) {
    using AgcDriver::FrameCallKind;
    std::vector<std::byte> memory(GuestBytes);
    for (const auto& test: Cases) writeShader(memory, test);
    std::array<std::uint8_t, Values8> input8{};
    for (std::uint32_t value = 0; value < Values8; ++value) input8[value] = static_cast<std::uint8_t>(value);
    write(memory, Input8, input8.data(), input8.size());
    std::vector<std::uint8_t> input16(Values16 * 2u);
    for (std::uint32_t value = 0; value < Values16; ++value) {
        input16[value * 2u] = static_cast<std::uint8_t>(value);
        input16[value * 2u + 1u] = static_cast<std::uint8_t>(value >> 8u);
    }
    write(memory, Input16, input16.data(), input16.size());
    std::vector<std::uint32_t> output(OutputBytes / sizeof(std::uint32_t), Fill);
    write(memory, Output, output.data(), OutputBytes);
    const auto commands = dispatchWords();
    write(memory, Commands, commands.data(), commands.size() * sizeof(std::uint32_t));

    AgcDriver::FrameCaptureWriter writer(path);
    writer.WriteRange(GuestBase, memory.size(), AgcDriver::FrameRangeKind::Watched, 0, 7);
    const auto snapshot = writer.WriteSnapshot(GuestBase, memory);
    std::uint64_t ordinal = 0;
    const auto emit = [&](AgcDriver::FrameCall call) {
        call.ordinal = ++ordinal;
        call.timeNs = ordinal * 1000u;
        call.thread = 1;
        writer.WriteCall(call);
    };
    for (const auto& test: Cases) emit({.kind = FrameCallKind::RegisterShader, .address = test.header});
    emit({.kind = FrameCallKind::RegisterVideoOutput, .values = {7}});
    emit({.kind = FrameCallKind::DisplayBuffer, .address = Display,
          .values = {0, 0x8000000022000000ull, 64, 4, 0, 0, 0, 0, 0}});
    emit({.kind = FrameCallKind::FrameStart, .values = {1}});
    emit({.kind = FrameCallKind::Submit, .address = Commands, .size = commands.size()});
    emit({.kind = FrameCallKind::Flip, .accepted = 1, .cpuFlip = true, .values = {7, UINT64_MAX, 1, 0}});
    emit({.kind = FrameCallKind::End, .accepted = 2});
    writer.Finish("{\"magic\":\"aps5-frame-capture 1\",\"arena_base\":65536,\"arena_bytes\":1082331693056,"
                  "\"start_boundary\":1,\"frames\":1,\"calls\":2,\"prologue_calls\":6,\"ranges\":1,"
                  "\"snapshot_bytes\":" + std::to_string(snapshot) +
                  ",\"unreadable_bytes\":0,\"host_only_sample_bytes\":0,\"warnings\":[]}\n");
}

std::string readFile(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

std::string quote(const std::filesystem::path& path) {
    return "\"" + path.string() + "\"";
}

std::uint32_t expected(std::uint32_t raw, const Case& test) {
    const auto sign = 1u << (test.bits - 1u);
    const auto numerator = test.signedValues && (raw & sign) != 0u
                               ? static_cast<std::int32_t>(raw) - static_cast<std::int32_t>(1u << test.bits)
                               : static_cast<std::int32_t>(raw);
    if (test.signedValues && numerator == -static_cast<std::int32_t>(sign)) return 0xbf800000u;
    const auto divisor = test.signedValues ? sign - 1u : (1u << test.bits) - 1u;
    return std::bit_cast<std::uint32_t>(static_cast<float>(static_cast<double>(numerator) / divisor));
}

void printEvidence(const std::string& log) {
    std::istringstream lines(log);
    std::string line;
    while (std::getline(lines, line)) {
        if (line.rfind("device: ", 0) == 0 ||
            line.rfind("driver_info: ", 0) == 0 ||
            line.find("Select device:") != std::string::npos ||
            line.find("Selected GPU name=") != std::string::npos ||
            line.find("Insert instance layer") != std::string::npos ||
            line.find("was not found but was requested") != std::string::npos ||
            line.find("VUID-") != std::string::npos) std::cout << line << '\n';
    }
    const auto key = log.find("\"unfinished\":");
    if (key != std::string::npos) {
        const auto start = key + sizeof("\"unfinished\":") - 1;
        const auto end = log.find_first_not_of("0123456789", start);
        std::cout << "unfinished=" << log.substr(start, end - start) << '\n';
    }
}

}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: kyty_shim_formatted_buffer_norm_tests <agc_frame_replay>");
        const std::filesystem::path tool = argv[1];
        const auto root = std::filesystem::temp_directory_path() /
                          ("kyty-formatted-buffer-norm-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root);
        makeCapture(root / "capture");
        char address[32];
        std::snprintf(address, sizeof(address), "0x%llx", static_cast<unsigned long long>(Output));
        const auto log = root / "replay.log";
        const auto dump = root / "output.bin";
        const auto command = quote(tool) + " " + quote(root / "capture") +
                             " --strict --no-synthesize --repeat 1 --cold-pipelines --frames " + quote(root / "frames") +
                             " --dump " + address + ":" + std::to_string(OutputBytes) + ":" + quote(dump) +
                             " > " + quote(log) + " 2>&1";
        std::fflush(nullptr);
#ifdef _WIN32
        const int status = std::system(('"' + command + '"').c_str());
#else
        const int status = std::system(command.c_str());
#endif
        const auto raw = readFile(dump);
        const auto replayLog = readFile(log);
        const bool vulkanRequired = std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr;
        if (status != 0 || raw.size() != OutputBytes) {
            std::cerr << "replay did not complete" << (vulkanRequired ? "" : "; skipping (no usable Vulkan device?)") << '\n';
            std::cerr << replayLog.substr(replayLog.size() > 4000 ? replayLog.size() - 4000 : 0) << '\n';
            return vulkanRequired ? 1 : 77;
        }
        printEvidence(replayLog);
        const char* requestedLayers = std::getenv("VK_INSTANCE_LAYERS");
        const bool validationRequested = requestedLayers != nullptr &&
                                         std::strstr(requestedLayers, "VK_LAYER_KHRONOS_validation") != nullptr;
        if (vulkanRequired &&
            (replayLog.find("VUID-") != std::string::npos ||
             replayLog.find("was not found but was requested by env var VK_INSTANCE_LAYERS") != std::string::npos ||
             (validationRequested && replayLog.find("Insert instance layer \"VK_LAYER_KHRONOS_validation\"") ==
                                         std::string::npos))) {
            std::cerr << "replay failed Vulkan validation log checks\n";
            return 1;
        }
        int failures = 0;
        for (const auto& test: Cases) {
            const auto start = static_cast<std::size_t>(test.output - Output);
            for (std::uint32_t value = 0; value < test.values; ++value) {
                std::uint32_t got = 0;
                std::memcpy(&got, raw.data() + start + value * sizeof(got), sizeof(got));
                const auto want = expected(value, test);
                if (got == want) continue;
                if (++failures <= 16) {
                    std::fprintf(stderr, "%s 0x%04x: got 0x%08x, expected 0x%08x\n", test.name, value, got, want);
                }
            }
            std::cout << test.name << ": " << test.values << " values checked\n";
        }
        if (failures != 0) {
            std::cerr << failures << " wrong values\n";
            return 1;
        }
        std::filesystem::remove_all(root);
        std::cout << "kyty_shim_formatted_buffer_norm: ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
