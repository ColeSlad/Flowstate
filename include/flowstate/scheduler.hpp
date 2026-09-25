#pragma once
#include "flowstate/runtime.hpp"
#include <functional>

namespace flowstate {
struct LayaTick {
    bool attempted = false, fallback = false, error = false;
    std::optional<double> confidence, latency_ms;
    std::string status;
};
struct PolicySelection {
    Policy policy;
    std::optional<LayaTick> laya;
};
struct ControllerSnapshot {
    RuntimeStats stats;
    Policy policy = Policy::CpuLatency;
    std::uint64_t changes = 0, errors = 0;
    std::chrono::steady_clock::time_point sampled_at{};
    std::uint64_t laya_calls = 0, laya_fallbacks = 0, laya_errors = 0;
    double total_laya_latency_ms = 0, max_laya_latency_ms = 0;
    std::optional<LayaTick> last_laya;
};
// Own this after Runtime and stop/destroy it before Runtime. No request calls
// the controller; the only write into the data plane is an atomic policy choice.
class PolicyController {
public:
    PolicyController(Runtime& runtime, HeuristicConfig config = {},
                        std::chrono::milliseconds interval = std::chrono::milliseconds(500),
                        std::function<PolicySelection(const RuntimeStats&)> selector = {});
    ~PolicyController();
    void stop();
    ControllerSnapshot snapshot() const;
private:
    void run();
    Runtime& runtime_;
    HeuristicConfig config_;
    std::chrono::milliseconds interval_;
    SystemSampler sampler_;
    std::function<PolicySelection(const RuntimeStats&)> selector_;
    mutable std::mutex mutex_;
    std::mutex stop_mutex_;
    std::condition_variable wake_;
    bool stopping_ = false;
    ControllerSnapshot state_;
    std::thread thread_;
};
using HeuristicController = PolicyController;
} // namespace flowstate
