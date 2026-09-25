#pragma once
#include "flowstate/scheduler.hpp"

namespace flowstate {
struct LayaConfig {
    std::uint16_t port = 8000;
    std::chrono::milliseconds interval{2000};
    std::chrono::milliseconds timeout{800};
    double min_confidence = .7; // Max class probability; not calibrated scheduling accuracy.
    double slo_ms = 15;
    void validate() const;
};
enum class LayaError { None, Timeout, Transport, Http, TooLarge, InvalidResponse };
struct LayaResponse {
    LayaError error = LayaError::None;
    long status = 200;
    std::string body;
};
using LayaTransport = std::function<LayaResponse(const std::string&)>;
// Local inference only. Endpoint override is restricted to numeric loopback HTTP.
LayaTransport make_laya_transport(const LayaConfig& config,
    std::string endpoint = {});
std::string serialize_laya_state(const RuntimeStats& stats, const LayaConfig& config);
PolicySelection select_laya_policy(const RuntimeStats& stats, const LayaConfig& config,
                                 const HeuristicConfig& heuristic, const LayaTransport& transport);
class LayaPolicyController {
public:
    LayaPolicyController(Runtime& runtime, LayaConfig config = {}, HeuristicConfig heuristic = {}, LayaTransport transport = {});
    void stop() { controller_.stop(); }
    ControllerSnapshot snapshot() const { return controller_.snapshot(); }
    PolicyController& controller() { return controller_; }
private:
    PolicyController controller_;
};
} // namespace flowstate
