#pragma once
#include "flowstate/runtime.hpp"

namespace flowstate {
struct ControllerSnapshot {
    RuntimeStats stats;
    Policy policy = Policy::CpuLatency;
    std::uint64_t changes = 0, errors = 0;
    std::chrono::steady_clock::time_point sampled_at{};
};
// Own this after Runtime and stop/destroy it before Runtime. No request calls
// the controller; the only write into the data plane is an atomic policy choice.
class HeuristicController {
public:
    HeuristicController(Runtime& runtime, HeuristicConfig config = {},
                        std::chrono::milliseconds interval = std::chrono::milliseconds(500));
    ~HeuristicController();
    void stop();
    ControllerSnapshot snapshot() const;
private:
    void run();
    Runtime& runtime_;
    HeuristicConfig config_;
    std::chrono::milliseconds interval_;
    SystemSampler sampler_;
    mutable std::mutex mutex_;
    std::mutex stop_mutex_;
    std::condition_variable wake_;
    bool stopping_ = false;
    ControllerSnapshot state_;
    std::thread thread_;
};
} // namespace flowstate
