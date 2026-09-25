#include "flowstate/laya.hpp"
#include <nlohmann/json.hpp>
#include <iostream>
#include <atomic>

namespace fs=flowstate;
using Json=nlohmann::json;
using namespace std::chrono_literals;
namespace {
std::size_t checks=0;
#define CHECK(x) do { ++checks; if(!(x)) throw std::runtime_error("Check failed at line "+std::to_string(__LINE__)+": " #x); } while(false)
Json answer(double confidence=.9,std::string choice="gpu_batch") {
    return {{"model","laya"},{"routing",{{"model","english"}}},{"answers",{{"policy",{{"type","choice"},{"choice",choice},{"confidence",.01},{"answer_confidence",confidence},
        {"probabilities",{{"cpu_latency",(1-confidence)/3},{"gpu_immediate",(1-confidence)/3},{"gpu_batch",confidence},{"balanced",(1-confidence)/3}}}}}}}};
}
void selection_tests() {
    fs::RuntimeStats stats; stats.gpu_available=true; stats.windows[0].arrival_rate=100;
    fs::LayaConfig config;
    auto request=Json::parse(fs::serialize_laya_state(stats,config));
    CHECK(request["state"]["cpu_use"].is_null());
    CHECK(request["state"]["cpu_p95_ms"].is_null());
    CHECK(request["questions"]["policy"]["criteria"].size()==4);
    CHECK(request.dump().find("Authorization")==std::string::npos);
    auto select=[&](fs::LayaResponse response) { return fs::select_laya_policy(stats,config,{},[response](const std::string&) { return response; }); };
    auto result=select({fs::LayaError::None,200,answer().dump()});
    CHECK(result.policy==fs::Policy::GpuBatch && !result.laya->fallback && result.laya->attempted);
    CHECK(result.laya->confidence==.9 && result.laya->latency_ms.has_value());
    result=select({fs::LayaError::None,200,answer(.4).dump()});
    CHECK(result.policy==fs::Policy::Balanced && result.laya->fallback && !result.laya->error && result.laya->status=="low_confidence");
    for(auto error:{fs::LayaError::Timeout,fs::LayaError::Transport}) {
        result=select({error,0,{}});
        CHECK(result.policy==fs::Policy::Balanced && result.laya->fallback && result.laya->error);
        CHECK(result.laya->attempted);
    }
    for(const auto& body:{std::string("invalid"),std::string(32769,'x'),answer(.9,"launch_kernel").dump(),answer(2).dump()}) {
        result=select({fs::LayaError::None,200,body});
        CHECK(result.laya->fallback && result.laya->error);
    }
    auto malformed=answer(); malformed["answers"]["policy"].erase("answer_confidence");
    CHECK(select({fs::LayaError::None,200,malformed.dump()}).laya->error);
    malformed=answer(); malformed["answers"]["policy"]["answer_confidence"]="0.9";
    CHECK(select({fs::LayaError::None,200,malformed.dump()}).laya->error);
    malformed=answer(); malformed["answers"]["policy"]["probabilities"]["gpu_batch"]=0;
    CHECK(select({fs::LayaError::None,200,malformed.dump()}).laya->error);
    malformed=answer(); malformed["answers"]["policy"]["answer_confidence"]=.8;
    CHECK(select({fs::LayaError::None,200,malformed.dump()}).laya->error);
    malformed=answer(); malformed["answers"]["policy"]["choice"]="balanced";
    CHECK(select({fs::LayaError::None,200,malformed.dump()}).laya->error);
    malformed=answer(); malformed["routing"]["model"]="multilingual";
    CHECK(select({fs::LayaError::None,200,malformed.dump()}).laya->error);
    CHECK(select({fs::LayaError::None,503,answer().dump()}).laya->status=="http_error");
    std::string nested(32,'['); nested+="0"; nested+=std::string(32,']');
    CHECK(select({fs::LayaError::None,200,nested}).laya->error);
    stats.system.gpu_memory_free_bytes=0;
    CHECK(select({fs::LayaError::None,200,answer().dump()}).laya->status=="gpu_memory_guard");
    stats.gpu_available=false;
    result=fs::select_laya_policy(stats,config,{},[](const std::string&)->fs::LayaResponse { throw std::runtime_error("Must not call service"); });
    CHECK(result.policy==fs::Policy::CpuLatency && !result.laya->attempted && !result.laya->error);
}
void controller_tests() {
    const auto data=fs::Dataset::generate(100,9);
    fs::Runtime runtime(data,{},fs::make_scalar_backend(data),fs::make_scalar_backend(data));
    fs::LayaConfig config; config.interval=500ms; config.timeout=200ms;
    std::mutex mutex;
    std::condition_variable ready;
    std::vector<std::chrono::steady_clock::time_point> calls;
    bool release=false;
    fs::LayaPolicyController controller(runtime,config,{},[&](const std::string&) {
        std::unique_lock lock(mutex);
        calls.push_back(std::chrono::steady_clock::now()); ready.notify_all();
        ready.wait_for(lock,2s,[&] { return release; });
        return fs::LayaResponse{fs::LayaError::None,200,answer().dump()};
    });
    struct Release { std::mutex& mutex; std::condition_variable& ready; bool& release;
        ~Release() { std::lock_guard lock(mutex); release=true; ready.notify_all(); }
    } guard{mutex,ready,release};
    {
        std::unique_lock lock(mutex);
        CHECK(ready.wait_for(lock,2s,[&] { return !calls.empty(); }));
    }
    for(int i=0;i<32;++i) {
        auto future=runtime.submit({std::vector<float>(9,.1f),1});
        CHECK(future.wait_for(1s)==std::future_status::ready);
        CHECK(future.get().route==fs::Route::Cpu); // Network wait does not block submissions/execution.
    }
    {
        std::unique_lock lock(mutex); release=true; ready.notify_all();
        CHECK(ready.wait_for(lock,2s,[&] { return calls.size()>=2; }));
    }
    controller.stop();
    CHECK(calls.size()==2 && calls[1]-calls[0]>=490ms);
    const auto state=controller.snapshot();
    CHECK(state.laya_calls==2 && state.laya_fallbacks==0 && state.laya_errors==0 && state.errors==0);
    CHECK(state.policy==fs::Policy::GpuBatch && state.total_laya_latency_ms>=0);
    runtime.shutdown();
}
void http_tests(const std::string& base) {
    fs::LayaConfig config; config.timeout=100ms;
    auto valid=fs::make_laya_transport(config,base+"/valid")("{}");
    CHECK(valid.error==fs::LayaError::None && valid.status==200);
    const auto start=std::chrono::steady_clock::now();
    auto slow=fs::make_laya_transport(config,base+"/slow")("{}");
    CHECK(slow.error==fs::LayaError::Timeout);
    CHECK(std::chrono::steady_clock::now()-start<1s);
    CHECK(fs::make_laya_transport(config,base+"/error")("{}").error==fs::LayaError::Http);
    CHECK(fs::make_laya_transport(config,base+"/large")("{}").error==fs::LayaError::TooLarge);
    CHECK(fs::make_laya_transport(config,base+"/redirect")("{}").status==302);
}
}
int main(int argc,char** argv) {
    try {
        if(argc==2 && std::string_view(argv[1])=="--sample-request") {
            fs::RuntimeStats stats; stats.gpu_available=true; stats.windows[0].arrival_rate=1200;
            stats.windows[0].throughput=1100; stats.queue_depth=32; stats.pending_gpu_jobs=16;
            stats.system.cpu_utilization=.65; stats.system.gpu_utilization=.45;
            stats.system.gpu_memory_free_bytes=14000ULL*1048576;
            stats.windows[0].backends[1].p95_ms=20; stats.windows[0].backends[2].p95_ms=5;
            stats.windows[0].backends[3].p95_ms=12; stats.windows[0].latency.p99_ms=25;
            stats.windows[0].mean_gpu_batch_size=24;
            std::cout<<fs::serialize_laya_state(stats,{})<<'\n'; return 0;
        }
        if(argc==2) http_tests(argv[1]);
        else { selection_tests(); controller_tests(); }
        std::cout<<checks<<" Laya checks passed\n";
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
