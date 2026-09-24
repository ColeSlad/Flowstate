#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>

namespace flowstate {

enum class MetricBackend : std::size_t { Scalar, Simd, Cuda, CudaBatch };
struct LatencyStats {
    std::uint64_t count = 0;
    std::optional<double> p50_ms, p95_ms, p99_ms;
};
struct WindowStats {
    double seconds = 0;
    double arrival_rate = 0;
    double throughput = 0;
    std::uint64_t arrivals = 0, completed = 0, failed = 0;
    std::array<LatencyStats, 4> backends;
    LatencyStats latency;
    std::optional<double> mean_gpu_batch_size;
};
struct SystemStats {
    std::optional<double> cpu_utilization; // System-wide fraction, not worker occupancy.
    std::optional<double> gpu_utilization;
    std::optional<std::uint64_t> gpu_memory_free_bytes;
};
struct RuntimeStats {
    std::array<WindowStats, 3> windows; // 1, 5, 30 seconds.
    std::size_t queue_depth = 0, pending_gpu_jobs = 0;
    bool gpu_available = false;
    SystemStats system;
};

// Bounded histograms in 100 ms buckets. Caller provides synchronization.
// Percentiles are upper bin bounds (at most 10% rounding, 1 us minimum).
class RollingTelemetry {
public:
    using Clock = std::chrono::steady_clock;
    explicit RollingTelemetry(Clock::time_point start = Clock::now());
    ~RollingTelemetry();
    void arrival(Clock::time_point now);
    void completion(Clock::time_point now, MetricBackend backend, double latency_ms, bool failed);
    void gpu_batch(Clock::time_point now, std::size_t size);
    std::array<WindowStats, 3> snapshot(Clock::time_point now) const;
private:
    struct Storage;
    std::unique_ptr<Storage> storage_;
};

// Called only by the slow control plane. Unsupported readings remain absent.
class SystemSampler {
public:
    explicit SystemSampler(bool gpu_enabled);
    ~SystemSampler();
    SystemSampler(const SystemSampler&) = delete;
    SystemSampler& operator=(const SystemSampler&) = delete;
    SystemStats sample();
private:
    struct State;
    std::unique_ptr<State> state_;
};
} // namespace flowstate
