#include "prx/libSceAgcDriver/Execution/include/FrameCapture.hpp"

#include <array>
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
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint64_t GuestBase = 0x202800000ull;
constexpr std::uint64_t RawHeader = GuestBase + 0x100u;
constexpr std::uint64_t RawCode = GuestBase + 0x800u;
constexpr std::uint64_t TypedHeader = GuestBase + 0x2100u;
constexpr std::uint64_t TypedCode = GuestBase + 0x2800u;
constexpr std::uint64_t Commands = GuestBase + 0x8000u;
constexpr std::uint64_t Input = GuestBase + 0x10000u;
constexpr std::uint64_t Output = GuestBase + 0x50000u;
constexpr std::uint64_t Guard = Output + 0x1000u;
constexpr std::uint64_t GuestBytes = 0x80000u;
constexpr std::uint32_t Lanes = 32u;
constexpr std::uint32_t RowBytes = Lanes * 8u;
constexpr std::uint32_t SlotBytes = 0x400u;
constexpr std::uint32_t Format32UInt = 20u;
constexpr std::uint32_t Format32x2UInt = 62u;
constexpr std::uint32_t Fill = 0xdeadbeefu;

struct Case {
    const char* name;
    bool typed;
    std::uint32_t source_records;
    std::uint32_t guard_records;
    std::uint32_t stride;
    std::uint32_t mode;
    std::uint32_t soffset;
    bool unaligned_store = false;
    bool uniform_input = false;
};

constexpr std::array Cases{
    Case{"raw exact bound", false, 256u, 256u, 0u, 3u, 0u},
    Case{"raw load straddles", false, 256u, 0u, 0u, 3u, 2u, false, true},
    Case{"raw store straddles", false, 1024u, 256u, 0u, 3u, 2u, true, true},
    Case{"raw load soffset beyond", false, 256u, 0u, 0u, 3u, 260u},
    Case{"raw store soffset beyond", false, 1024u, 256u, 0u, 3u, 260u},
    Case{"raw soffset does not wrap", false, 256u, 256u, 0u, 3u, 0xfffffffcu},
    Case{"structured stride", false, 32u, 32u, 12u, 0u, 0u},
    Case{"typed load straddles", true, 252u, 0u, 0u, 3u, 0u},
    Case{"typed store straddles", true, 1024u, 252u, 0u, 3u, 0u},
};

constexpr std::uint32_t OutputBytes = static_cast<std::uint32_t>(Cases.size()) * RowBytes;
constexpr std::uint32_t DumpBytes = 0x1000u + static_cast<std::uint32_t>(Cases.size()) * SlotBytes;

constexpr std::array<std::uint32_t, 9> RawWords{
    0x34020083u, 0xe0341000u, 0x0c000201u, 0xbf8c0000u, 0xe0741000u,
    0x0c020201u, 0xe0741000u, 0x80010201u, 0xbf810000u,
};

constexpr std::array<std::uint32_t, 9> TypedWords{
    0x34020083u, 0xe9f11000u, 0x0c000201u, 0xbf8c0000u, 0xe9f51000u,
    0x0c020201u, 0xe0741000u, 0x80010201u, 0xbf810000u,
};

using Sharp = std::array<std::uint32_t, 4>;

Sharp sharp(std::uint64_t address, std::uint32_t records, std::uint32_t stride, std::uint32_t mode,
            std::uint32_t format = Format32UInt) {
    return {static_cast<std::uint32_t>(address),
            static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (stride << 16u),
            records, 0xfacu | (format << 12u) | (1u << 24u) | (mode << 28u)};
}

std::uint32_t low(std::uint64_t address) {
    return static_cast<std::uint32_t>(address >> 8u);
}

std::uint32_t high(std::uint64_t address) {
    return static_cast<std::uint32_t>(address >> 40u);
}

void append(std::vector<std::uint32_t>& words, std::uint32_t opcode, std::initializer_list<std::uint32_t> payload) {
    words.push_back(0xc0000000u | (static_cast<std::uint32_t>(payload.size() - 1u) << 16u) | (opcode << 8u));
    words.insert(words.end(), payload);
}

std::vector<std::uint32_t> dispatchWords() {
    std::vector<std::uint32_t> words;
    for (std::size_t index = 0; index < Cases.size(); ++index) {
        const auto& test = Cases[index];
        const auto format = test.typed ? Format32x2UInt : Format32UInt;
        const auto source = sharp(Input + index * SlotBytes, test.source_records, test.stride, test.mode, format);
        const auto output = sharp(Output + index * RowBytes, RowBytes, 0u, 3u);
        const auto guard = sharp(Guard + index * SlotBytes, test.guard_records, test.stride, test.mode, format);
        const auto code = test.typed ? TypedCode : RawCode;
        append(words, 0x76u, {0x20cu, low(code), high(code)});
        append(words, 0x76u, {0x212u, 0u});
        append(words, 0x76u, {0x213u, 13u << 1u});
        append(words, 0x76u, {0x207u, Lanes, 1u, 1u});
        append(words, 0x76u,
               {0x240u, source[0], source[1], source[2], source[3],
                output[0], output[1], output[2], output[3],
                guard[0], guard[1], guard[2], guard[3], test.soffset});
        append(words, 0x15u, {1u, 1u, 1u, 0x41u});
    }
    return words;
}

template <typename T, std::size_t Extent>
void write(std::span<std::byte> memory, std::uint64_t address, std::span<const T, Extent> values) {
    const auto offset = address - GuestBase;
    if (offset > memory.size() || values.size_bytes() > memory.size() - offset) {
        throw std::runtime_error("fixture write past guest mapping");
    }
    std::memcpy(memory.data() + offset, values.data(), values.size_bytes());
}

void writeShader(std::span<std::byte> memory, std::uint64_t header_address, std::uint64_t code_address,
                 std::span<const std::uint32_t> code) {
    struct Header {
        Shader shader{};
        ShaderUserData user{};
    };
    Header header{};
    header.shader.file_header = 0x34333231u;
    header.shader.version = 0x18u;
    header.shader.header_size = sizeof(Header);
    header.shader.shader_size = static_cast<std::uint32_t>(code.size_bytes());
    header.shader.code = reinterpret_cast<const void*>(code_address);
    header.shader.type = 0;
    header.shader.user_data = reinterpret_cast<ShaderUserData*>(header_address + offsetof(Header, user));
    write(memory, header_address, std::span<const Header>(&header, 1));
    write(memory, code_address, code);
}

std::uint32_t inputWord(const Case& test, std::size_t ordinal, std::uint32_t word) {
    if (test.uniform_input) return 0x5a5a5a5au;
    return 0x9e3779b9u * (word + 1u) ^ (0x12345678u + static_cast<std::uint32_t>(ordinal) * 0x01010101u);
}

void makeCapture(const std::filesystem::path& path) {
    using AgcDriver::FrameCallKind;
    std::vector<std::byte> memory(GuestBytes);
    writeShader(memory, RawHeader, RawCode, RawWords);
    writeShader(memory, TypedHeader, TypedCode, TypedWords);
    const auto commands = dispatchWords();
    write(memory, Commands, std::span(commands));
    for (std::size_t index = 0; index < Cases.size(); ++index) {
        std::array<std::uint32_t, SlotBytes / 4u> input{};
        std::array<std::uint32_t, SlotBytes / 4u> guard{};
        for (std::uint32_t word = 0; word < input.size(); ++word) input[word] = inputWord(Cases[index], index, word);
        guard.fill(Fill);
        write(memory, Input + index * SlotBytes, std::span<const std::uint32_t>(input));
        write(memory, Guard + index * SlotBytes, std::span<const std::uint32_t>(guard));
    }
    std::vector<std::uint32_t> output(OutputBytes / 4u, Fill);
    write(memory, Output, std::span<const std::uint32_t>(output));
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
    emit({.kind = FrameCallKind::RegisterShader, .address = RawHeader});
    emit({.kind = FrameCallKind::RegisterShader, .address = TypedHeader});
    emit({.kind = FrameCallKind::RegisterVideoOutput, .values = {7}});
    emit({.kind = FrameCallKind::DisplayBuffer, .address = Output,
          .values = {0, 0x8000000022000000ull, 64, Cases.size(), 0, 0, 0, 0, 0}});
    emit({.kind = FrameCallKind::FrameStart, .values = {1}});
    emit({.kind = FrameCallKind::Submit, .address = Commands, .size = commands.size()});
    emit({.kind = FrameCallKind::Flip, .accepted = 1, .cpuFlip = true, .values = {7, UINT64_MAX, 1, 0}});
    emit({.kind = FrameCallKind::End, .accepted = 2});
    writer.Finish("{\"magic\":\"aps5-frame-capture 1\",\"arena_base\":65536,\"arena_bytes\":1082331693056,"
                  "\"start_boundary\":1,\"frames\":1,\"calls\":2,\"prologue_calls\":4,\"ranges\":1,"
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

bool inBounds(const Case& test, std::uint32_t records, std::uint32_t lane, std::uint32_t component) {
    if (records == 0u) return false;
    const std::uint64_t offset = lane * 8u + (test.typed ? 0u : component * 4u);
    const std::uint64_t bytes = test.typed ? 8u : 4u;
    if (test.stride != 0u) return offset + bytes <= test.stride;
    return test.soffset + offset + bytes <= records;
}

std::uint32_t loaded(const Case& test, std::size_t index, std::uint32_t lane, std::uint32_t component) {
    if (!inBounds(test, test.source_records, lane, component)) return 0u;
    return inputWord(test, index, (test.soffset + lane * 8u + component * 4u) / 4u);
}

void printEvidence(const std::string& log) {
    std::size_t at = 0;
    while (at < log.size()) {
        const auto end = log.find('\n', at);
        const auto line = log.substr(at, end == std::string::npos ? end : end - at);
        if (line.find("Select device:") != std::string::npos ||
            line.find("Selected GPU name=") != std::string::npos ||
            line.find("device:") != std::string::npos ||
            line.find("driver_info:") != std::string::npos ||
            line.find("Insert instance layer") != std::string::npos ||
            line.find("was not found but was requested") != std::string::npos ||
            line.find("VUID-") != std::string::npos) std::cout << line << '\n';
        if (end == std::string::npos) break;
        at = end + 1u;
    }
}

}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: kyty_shim_buffer_bounds_replay_tests <agc_frame_replay>");
        const std::filesystem::path tool = argv[1];
        const auto root = std::filesystem::temp_directory_path() /
                          ("kyty-buffer-bounds-replay-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root);
        makeCapture(root / "capture");
        char address[32];
        std::snprintf(address, sizeof(address), "0x%llx", static_cast<unsigned long long>(Output));
        const auto log = root / "replay.log";
        const auto dump = root / "output.bin";
        const auto command = quote(tool) + " " + quote(root / "capture") +
                             " --strict --no-synthesize --repeat 1 --cold-pipelines --frames " + quote(root / "frames") +
                             " --dump " + address + ":" + std::to_string(DumpBytes) + ":" + quote(dump) +
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
        if (status != 0 || raw.size() != DumpBytes) {
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
        for (std::size_t index = 0; index < Cases.size(); ++index) {
            const auto& test = Cases[index];
            std::array<std::uint32_t, SlotBytes / 4u> expectedGuard{};
            std::array<bool, SlotBytes> guardCheck{};
            expectedGuard.fill(Fill);
            guardCheck.fill(true);
            for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
                for (std::uint32_t component = 0; component < 2u; ++component) {
                    const auto want = loaded(test, index, lane, component);
                    const auto at = index * RowBytes + lane * 8u + component * 4u;
                    std::uint32_t got = 0;
                    std::memcpy(&got, raw.data() + at, sizeof(got));
                    if (got != want && ++failures <= 16) {
                        std::fprintf(stderr, "%s lane %u load %u: got 0x%08x, expected 0x%08x\n",
                                     test.name, lane, component, got, want);
                    }
                    if (inBounds(test, test.guard_records, lane, component)) {
                        const auto byte = test.soffset + lane * 8u + component * 4u;
                        if (byte > SlotBytes || sizeof(want) > SlotBytes - byte) {
                            throw std::runtime_error("expected guard write past fixture slot");
                        }
                        if (test.unaligned_store) {
                            for (std::uint32_t part = 0; part < sizeof(want); ++part) {
                                guardCheck[byte + part] = false;
                                guardCheck[(byte & ~3u) + part] = false;
                            }
                        } else {
                            std::memcpy(reinterpret_cast<std::byte*>(expectedGuard.data()) + byte, &want, sizeof(want));
                        }
                    }
                }
            }
            for (std::uint32_t byte = 0; byte < SlotBytes; ++byte) {
                if (!guardCheck[byte]) continue;
                const auto at = Guard - Output + index * SlotBytes + byte;
                const auto got = static_cast<std::uint8_t>(raw[at]);
                const auto want = reinterpret_cast<const std::uint8_t*>(expectedGuard.data())[byte];
                if (got != want && ++failures <= 16) {
                    std::fprintf(stderr, "%s guard byte %u: got 0x%02x, expected 0x%02x\n",
                                 test.name, byte, got, want);
                }
            }
            std::cout << test.name << ": checked\n";
        }
        if (failures != 0) return 1;
        std::filesystem::remove_all(root);
        std::cout << "kyty_shim_buffer_bounds_replay: ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
