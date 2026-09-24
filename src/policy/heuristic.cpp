#include "flowstate/policy.hpp"
#include <cmath>
#include <stdexcept>

namespace flowstate {
std::string_view policy_name(Policy policy) {
    switch(policy) {
        case Policy::CpuLatency: return "cpu_latency";
        case Policy::GpuImmediate: return "gpu_immediate";
        case Policy::GpuBatch: return "gpu_batch";
        case Policy::Balanced: return "balanced";
    }
    throw std::invalid_argument("Unknown policy");
}
Policy parse_policy(std::string_view name) {
    for(auto policy:{Policy::CpuLatency,Policy::GpuImmediate,Policy::GpuBatch,Policy::Balanced})
        if(policy_name(policy)==name) return policy;
    throw std::invalid_argument("Unknown policy: "+std::string(name));
}
void HeuristicConfig::validate() const {
    if(!std::isfinite(low_rate) || !std::isfinite(balanced_rate) || !std::isfinite(batch_rate) ||
       low_rate<0 || balanced_rate<=low_rate || batch_rate<=balanced_rate || high_queue==0 ||
       !std::isfinite(gpu_busy) || gpu_busy<=0 || gpu_busy>1 ||
       !std::isfinite(cpu_busy) || cpu_busy<=0 || cpu_busy>1)
        throw std::invalid_argument("Invalid heuristic thresholds");
}
Policy choose_policy(const RuntimeStats& stats, const HeuristicConfig& config) {
    config.validate();
    if(!stats.gpu_available) return Policy::CpuLatency;
    if(stats.system.gpu_memory_free_bytes && *stats.system.gpu_memory_free_bytes<config.min_gpu_free_bytes)
        return Policy::CpuLatency;
    const auto rate=stats.windows[0].arrival_rate;
    if(rate<=config.low_rate && stats.queue_depth<config.high_queue) return Policy::CpuLatency;
    if(stats.system.gpu_utilization && *stats.system.gpu_utilization>=config.gpu_busy) return Policy::Balanced;
    if(stats.queue_depth>=config.high_queue || rate>=config.batch_rate) return Policy::GpuBatch;
    if(rate>=config.balanced_rate || (stats.system.cpu_utilization && *stats.system.cpu_utilization>=config.cpu_busy))
        return Policy::Balanced;
    return Policy::GpuImmediate;
}
} // namespace flowstate
