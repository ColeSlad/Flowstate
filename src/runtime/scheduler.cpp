#include "flowstate/scheduler.hpp"

namespace flowstate {
HeuristicController::HeuristicController(Runtime& runtime, HeuristicConfig config, std::chrono::milliseconds interval)
    : runtime_(runtime),config_(config),interval_(interval),sampler_(runtime.has_gpu()) {
    config_.validate();
    if(interval_<std::chrono::milliseconds(500) || interval_>std::chrono::milliseconds(2000))
        throw std::invalid_argument("Control interval must be 500-2000 ms");
    thread_=std::thread(&HeuristicController::run,this);
}
HeuristicController::~HeuristicController() { stop(); }
void HeuristicController::stop() {
    std::lock_guard stop_lock(stop_mutex_);
    { std::lock_guard lock(mutex_); stopping_=true; }
    wake_.notify_all();
    if(thread_.joinable()) thread_.join();
}
ControllerSnapshot HeuristicController::snapshot() const { std::lock_guard lock(mutex_); return state_; }
void HeuristicController::run() {
    for(;;) {
        const auto started=std::chrono::steady_clock::now();
        try {
            auto stats=runtime_.telemetry();
            stats.system=sampler_.sample();
            const auto selected=choose_policy(stats,config_);
            runtime_.set_policy(selected);
            std::lock_guard lock(mutex_);
            state_.changes+=selected!=state_.policy;
            state_.policy=selected; state_.stats=std::move(stats); state_.sampled_at=std::chrono::steady_clock::now();
        } catch(...) {
            runtime_.set_policy(Policy::CpuLatency);
            std::lock_guard lock(mutex_);
            ++state_.errors; state_.changes+=state_.policy!=Policy::CpuLatency;
            state_.policy=Policy::CpuLatency; state_.stats.system={};
        }
        std::unique_lock lock(mutex_);
        if(wake_.wait_until(lock,started+interval_,[&] { return stopping_; })) return;
    }
}
} // namespace flowstate
