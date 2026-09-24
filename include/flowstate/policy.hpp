#pragma once
#include "flowstate/telemetry.hpp"
#include <string_view>

namespace flowstate {
enum class Policy { CpuLatency, GpuImmediate, GpuBatch, Balanced };
std::string_view policy_name(Policy policy);
Policy parse_policy(std::string_view name);
struct HeuristicConfig {
    double low_rate = 25;
    double balanced_rate = 80;
    double batch_rate = 300;
    std::size_t high_queue = 16;
    double gpu_busy = .90;
    double cpu_busy = .50;
    std::uint64_t min_gpu_free_bytes = 64*1024*1024;
    void validate() const;
};
Policy choose_policy(const RuntimeStats& stats, const HeuristicConfig& config = {});
} // namespace flowstate
