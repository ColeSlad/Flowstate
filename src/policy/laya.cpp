#include "flowstate/laya.hpp"
#include <nlohmann/json.hpp>
#include <cmath>

namespace flowstate {
using Json=nlohmann::json;
namespace {
template<class T> Json nullable(const std::optional<T>& value) {
    return value && std::isfinite(static_cast<double>(*value)) ? Json(std::round(static_cast<double>(*value)*100)/100) : Json(nullptr);
}
std::string_view error_name(LayaError error) {
    switch(error) {
        case LayaError::None: return "accepted";
        case LayaError::Timeout: return "timeout";
        case LayaError::Transport: return "transport_error";
        case LayaError::Http: return "http_error";
        case LayaError::TooLarge: return "response_too_large";
        case LayaError::InvalidResponse: return "invalid_response";
    }
    return "invalid_response";
}
}
void LayaConfig::validate() const {
    if(!port ||
       interval<std::chrono::milliseconds(500) || interval>std::chrono::milliseconds(2000) ||
       timeout<std::chrono::milliseconds(1) || timeout>interval/2 ||
       !std::isfinite(min_confidence) || min_confidence<0 || min_confidence>1 ||
       !std::isfinite(slo_ms) || slo_ms<=0 || slo_ms>60000)
        throw std::invalid_argument("Invalid Laya controller configuration");
}
std::string serialize_laya_state(const RuntimeStats& stats, const LayaConfig& config) {
    const auto& window=stats.windows[0];
    Json state={
        {"slo_ms",config.slo_ms},
        {"qps",std::round(window.arrival_rate)}, {"completed_qps",std::round(window.throughput)},
        {"queued",stats.queue_depth},{"gpu_pending",stats.pending_gpu_jobs},
        {"cpu_use",nullable(stats.system.cpu_utilization)},
        {"gpu_use",nullable(stats.system.gpu_utilization)},
        {"gpu_free_mib",stats.system.gpu_memory_free_bytes ? Json(*stats.system.gpu_memory_free_bytes/1048576) : Json(nullptr)},
        {"cpu_p95_ms",nullable(window.backends[1].p95_ms ? window.backends[1].p95_ms : window.backends[0].p95_ms)},
        {"gpu_p95_ms",nullable(window.backends[2].p95_ms)},
        {"batch_p95_ms",nullable(window.backends[3].p95_ms)},
        {"p99_ms",nullable(window.latency.p99_ms)},
        {"batch_size",nullable(window.mean_gpu_batch_size)}
    };
    Json criteria={
        {"cpu_latency","CPU: sparse traffic or little free GPU memory."},
        {"gpu_immediate","GPU: moderate traffic, spare GPU capacity."},
        {"gpu_batch","Batched GPU: high traffic or backlog, spare GPU capacity."},
        {"balanced","Half CPU and half GPU: busy GPU, spare CPU capacity."}
    };
    Json request={{"model","english"},{"state",std::move(state)},
        {"questions",{{"policy",{{"type","choice"},{"instructions","Select a policy to reduce p99 below slo_ms and maximize throughput. Metrics cover one second; use is 0-1, null is unknown. Only new requests change route."},{"criteria",std::move(criteria)}}}}}};
    auto body=request.dump();
    if(body.size()>16384) throw std::runtime_error("Laya state exceeds limit");
    return body;
}
PolicySelection select_laya_policy(const RuntimeStats& stats,const LayaConfig& config,
                                 const HeuristicConfig& heuristic,const LayaTransport& transport) {
    config.validate();
    PolicySelection result{choose_policy(stats,heuristic),LayaTick{}};
    auto& tick=*result.laya;
    tick.fallback=true;
    if(!stats.gpu_available) { tick.status="gpu_unavailable"; return result; }
    const auto started=std::chrono::steady_clock::now();
    LayaResponse response;
    try {
        tick.attempted=true;
        response=transport(serialize_laya_state(stats,config));
    } catch(...) { response.error=LayaError::Transport; }
    tick.latency_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
    if(response.error==LayaError::None && response.status!=200) response.error=LayaError::Http;
    if(response.error==LayaError::None && response.body.size()>32768) response.error=LayaError::TooLarge;
    if(response.error!=LayaError::None) {
        tick.error=true; tick.status=error_name(response.error); return result;
    }
    try {
        const auto json=Json::parse(response.body,[](int depth,Json::parse_event_t,const Json&) {
            if(depth>16) throw std::invalid_argument("Response nesting exceeds limit");
            return true;
        });
        const auto& answer=json.at("answers").at("policy");
        if(answer.at("type")!="choice" || !answer.at("answer_confidence").is_number()) throw std::invalid_argument("Invalid answer type");
        const auto policy=parse_policy(answer.at("choice").get<std::string>());
        const auto confidence=answer.at("answer_confidence").get<double>();
        if(!std::isfinite(confidence) || confidence<0 || confidence>1) throw std::invalid_argument("Invalid confidence");
        const auto& probabilities=answer.at("probabilities");
        if(!probabilities.is_object() || probabilities.size()!=4) throw std::invalid_argument("Invalid probabilities");
        double total=0,maximum=0;
        for(auto candidate:{Policy::CpuLatency,Policy::GpuImmediate,Policy::GpuBatch,Policy::Balanced}) {
            const auto& field=probabilities.at(std::string(policy_name(candidate)));
            if(!field.is_number()) throw std::invalid_argument("Invalid probability");
            const auto value=field.get<double>();
            if(!std::isfinite(value) || value<0 || value>1) throw std::invalid_argument("Invalid probability");
            total+=value; maximum=std::max(maximum,value);
        }
        if(std::abs(total-1)>0.02) throw std::invalid_argument("Invalid probability sum");
        const auto selected=probabilities.at(std::string(policy_name(policy))).get<double>();
        if(std::abs(confidence-selected)>0.00011 || selected+0.00011<maximum)
            throw std::invalid_argument("Choice/confidence disagree with probabilities");
        if(json.at("routing").at("model")!="english") throw std::invalid_argument("Unexpected checkpoint");
        tick.confidence=confidence;
        if(confidence<config.min_confidence) { tick.status="low_confidence"; return result; }
        if(stats.system.gpu_memory_free_bytes && *stats.system.gpu_memory_free_bytes<heuristic.min_gpu_free_bytes && policy!=Policy::CpuLatency) {
            tick.status="gpu_memory_guard"; return result;
        }
        result.policy=policy; tick.fallback=false; tick.status="accepted";
    } catch(...) { tick.error=true; tick.status="invalid_response"; }
    return result;
}
namespace {
auto selector(LayaConfig config,HeuristicConfig heuristic,LayaTransport transport) {
    config.validate(); heuristic.validate();
    if(!transport) transport=make_laya_transport(config);
    return [config,heuristic,transport=std::move(transport)](const RuntimeStats& stats) {
            return select_laya_policy(stats,config,heuristic,transport);
        };
}
}
LayaPolicyController::LayaPolicyController(Runtime& runtime,LayaConfig config,HeuristicConfig heuristic,LayaTransport transport)
    : controller_(runtime,heuristic,config.interval,selector(config,heuristic,std::move(transport))) {}
} // namespace flowstate
