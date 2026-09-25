#include "flowstate/scheduler.hpp"
#include <algorithm>

namespace flowstate {
PolicyController::PolicyController(Runtime& runtime, HeuristicConfig config, std::chrono::milliseconds interval,
                                   std::function<PolicySelection(const RuntimeStats&)> selector)
    : runtime_(runtime),config_(config),interval_(interval),sampler_(runtime.has_gpu()),selector_(std::move(selector)) {
    config_.validate();
    if(interval_<std::chrono::milliseconds(500) || interval_>std::chrono::milliseconds(2000))
        throw std::invalid_argument("Control interval must be 500-2000 ms");
    thread_=std::thread(&PolicyController::run,this);
}
PolicyController::~PolicyController() { stop(); }
void PolicyController::stop() {
    std::lock_guard stop_lock(stop_mutex_);
    { std::lock_guard lock(mutex_); stopping_=true; }
    wake_.notify_all();
    if(thread_.joinable()) thread_.join();
}
ControllerSnapshot PolicyController::snapshot() const { std::lock_guard lock(mutex_); return state_; }
void PolicyController::run() {
    for(;;) {
        const auto started=std::chrono::steady_clock::now();
        try {
            auto stats=runtime_.telemetry();
            stats.system=sampler_.sample();
            const auto sampled_at=std::chrono::steady_clock::now();
            const auto selected=selector_ ? selector_(stats) : PolicySelection{choose_policy(stats,config_),std::nullopt};
            runtime_.set_policy(selected.policy);
            std::lock_guard lock(mutex_);
            state_.changes+=selected.policy!=state_.policy;
            state_.policy=selected.policy; state_.stats=std::move(stats); state_.sampled_at=sampled_at;
            state_.last_laya=selected.laya;
            if(selected.laya) {
                state_.laya_calls+=selected.laya->attempted;
                state_.laya_fallbacks+=selected.laya->fallback;
                state_.laya_errors+=selected.laya->error;
                if(selected.laya->latency_ms) {
                    state_.total_laya_latency_ms+=*selected.laya->latency_ms;
                    state_.max_laya_latency_ms=std::max(state_.max_laya_latency_ms,*selected.laya->latency_ms);
                }
            }
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
