#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace flowstate {
struct LoadPhase {
    std::string name;
    std::uint64_t duration_ms;
    std::uint64_t requests_per_second;
    std::size_t burst;
    std::size_t top_k;
};
std::vector<LoadPhase> read_trace(const std::string& path, std::size_t vectors);
} // namespace flowstate
