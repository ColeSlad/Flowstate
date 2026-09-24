#pragma once

#include "flowstate/backend.hpp"
#include "flowstate/policy.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace flowstate {

enum class Route { Cpu, GpuImmediate, GpuBatch };

struct RuntimeConfig {
    std::size_t cpu_workers = 2;
    std::size_t queue_capacity = 1024; // Total waiting requests, across both queues.
    std::size_t max_batch_size = 32;
    std::chrono::microseconds max_wait{2000};
    bool enable_cuda = false;
};

class QueueFull : public std::runtime_error {
public:
    QueueFull() : std::runtime_error("Runtime request queue is full") {}
};
class RuntimeStopped : public std::runtime_error {
public:
    RuntimeStopped() : std::runtime_error("Runtime is shutting down") {}
};

struct Completion {
    std::uint64_t request_id;
    SearchResult result;
    std::string backend;
    Route route;
    std::size_t batch_size;
    double queue_wait_ms;
    double execution_ms;
    double latency_ms;
    // Whole-batch device timings, not divided or counted once per query.
    std::optional<DeviceTimings> device_timings;
};

struct RuntimeSnapshot {
    std::uint64_t submitted = 0;
    std::uint64_t completed = 0; // Includes failures; failed is a subset.
    std::uint64_t failed = 0;
    std::uint64_t rejected = 0; // Queue-full or shutdown rejection.
    std::uint64_t cpu_completed = 0;
    std::uint64_t gpu_completed = 0;
    std::uint64_t gpu_batches = 0;
    std::uint64_t timed_gpu_batches = 0;
    DeviceTimings total_device_timings;
    std::uint64_t size_flushes = 0;
    std::uint64_t timeout_flushes = 0;
    std::size_t cpu_queue_depth = 0;
    std::size_t gpu_queue_depth = 0;
    std::size_t cpu_inflight = 0;
    std::size_t gpu_inflight = 0;
    std::size_t max_queue_depth = 0;
    double total_queue_wait_ms = 0;
    double total_latency_ms = 0;
    std::optional<DeviceTimings> last_device_timings;
};

class Runtime {
public:
    explicit Runtime(std::shared_ptr<const Dataset> dataset, RuntimeConfig config = {});
    // Explicit backends enable deterministic runtime tests. CPU must be reentrant;
    // both backends must search the supplied dataset. GPU is called serially.
    Runtime(std::shared_ptr<const Dataset> dataset, RuntimeConfig config,
            std::unique_ptr<Backend> cpu, std::unique_ptr<Backend> gpu);
    ~Runtime();
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    std::future<Completion> submit(Query query); // Routes using the current policy.
    std::future<Completion> submit(Query query, Route route);
    void set_policy(Policy policy);
    [[nodiscard]] Policy policy() const { return policy_.load(std::memory_order_relaxed); }
    // Idempotent and safe for concurrent callers. Drain accepted jobs, then join.
    void shutdown();
    [[nodiscard]] RuntimeSnapshot snapshot() const;
    [[nodiscard]] RuntimeStats telemetry() const;
    [[nodiscard]] bool has_gpu() const { return gpu_ != nullptr; }
private:
    using Clock = std::chrono::steady_clock;
    struct Job {
        Query query;
        Route route;
        std::uint64_t id;
        Clock::time_point submitted;
        std::promise<Completion> promise;
    };
    void cpu_loop();
    void gpu_loop();
    void complete(Job& job, SearchResult result, Clock::time_point started, Clock::time_point ended,
                  std::size_t batch_size, std::optional<DeviceTimings> timings, std::exception_ptr error);

    std::shared_ptr<const Dataset> dataset_;
    RuntimeConfig config_;
    std::unique_ptr<Backend> cpu_, gpu_;
    std::string cpu_name_, gpu_name_;
    mutable std::mutex mutex_;
    std::mutex shutdown_mutex_;
    std::condition_variable cpu_ready_, gpu_ready_;
    std::deque<Job> cpu_queue_, gpu_queue_;
    bool stopping_ = false;
    std::uint64_t next_id_ = 1;
    RuntimeSnapshot stats_;
    RollingTelemetry telemetry_;
    std::atomic<Policy> policy_{Policy::CpuLatency};
    std::atomic<std::uint64_t> balanced_sequence_{0};
    // Reserved before threads start; only the GPU worker accesses this scratch space.
    std::vector<Job> gpu_jobs_;
    std::vector<Query> gpu_queries_;
    std::vector<std::thread> cpu_threads_;
    std::thread gpu_thread_;
};

} // namespace flowstate
