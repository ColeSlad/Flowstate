#include "flowstate/scheduler.hpp"
#include "flowstate/workload.hpp"
#include <cmath>
#include <iostream>
#include <filesystem>
#include <fstream>

namespace fs=flowstate;
using namespace std::chrono_literals;
namespace {
std::size_t checks=0;
#define CHECK(x) do { ++checks; if(!(x)) throw std::runtime_error("Check failed at line "+std::to_string(__LINE__)+": " #x); } while(false)
template<class Function> void invalid(Function f) {
    ++checks; try { f(); } catch(const std::invalid_argument&) { return; }
    throw std::runtime_error("Expected invalid argument");
}
void metrics_tests() {
    const auto start=fs::RollingTelemetry::Clock::time_point{};
    fs::RollingTelemetry metrics(start);
    CHECK(!metrics.snapshot(start)[0].latency.p95_ms);
    metrics.arrival(start+100ms); metrics.arrival(start+500ms);
    metrics.completion(start+500ms,fs::MetricBackend::Simd,1.0,false);
    metrics.completion(start+500ms,fs::MetricBackend::CudaBatch,10.0,false);
    metrics.completion(start+500ms,fs::MetricBackend::Cuda,0,true);
    metrics.gpu_batch(start+500ms,16);
    auto stats=metrics.snapshot(start+900ms);
    CHECK(stats[0].arrivals==2 && stats[0].completed==3 && stats[0].failed==1);
    CHECK(stats[0].arrival_rate==2);
    CHECK(stats[0].latency.count==2 && !stats[0].backends[0].p50_ms && !stats[0].backends[2].p50_ms);
    CHECK(*stats[0].latency.p50_ms>=1 && *stats[0].latency.p50_ms<=1.1);
    CHECK(*stats[0].latency.p99_ms>=10 && *stats[0].latency.p99_ms<=11);
    CHECK(stats[0].mean_gpu_batch_size==16);
    stats=metrics.snapshot(start+2s);
    CHECK(stats[0].arrivals==0 && !stats[0].latency.p50_ms && stats[1].arrivals==2);
    stats=metrics.snapshot(start+6s);
    CHECK(stats[1].arrivals==0 && stats[2].arrivals==2);
    metrics.arrival(start+31s); // Reuses the same bounded bucket storage.
    stats=metrics.snapshot(start+32s);
    CHECK(stats[0].arrivals==0 && stats[2].arrivals==1 && !stats[2].latency.p50_ms);
}
void policy_tests() {
    fs::RuntimeStats stats;
    fs::HeuristicConfig config; config.validate();
    for(auto policy:{fs::Policy::CpuLatency,fs::Policy::GpuImmediate,fs::Policy::GpuBatch,fs::Policy::Balanced})
        CHECK(fs::parse_policy(fs::policy_name(policy))==policy);
    invalid([] { fs::parse_policy("gpu_batch\n"); });
    invalid([] { auto c=fs::HeuristicConfig{}; c.batch_rate=0; c.validate(); });
    CHECK(fs::choose_policy(stats)==fs::Policy::CpuLatency);
    stats.gpu_available=true;
    CHECK(fs::choose_policy(stats)==fs::Policy::CpuLatency);
    stats.windows[0].arrival_rate=50;
    CHECK(fs::choose_policy(stats)==fs::Policy::GpuImmediate);
    stats.windows[0].arrival_rate=150;
    CHECK(fs::choose_policy(stats)==fs::Policy::Balanced);
    stats.windows[0].arrival_rate=1000;
    CHECK(fs::choose_policy(stats)==fs::Policy::GpuBatch); // No invented utilization required.
    stats.system.gpu_utilization=.98;
    CHECK(fs::choose_policy(stats)==fs::Policy::Balanced);
    stats.system.gpu_memory_free_bytes=0;
    CHECK(fs::choose_policy(stats)==fs::Policy::CpuLatency);
    stats.system={}; stats.windows[0].arrival_rate=0; stats.queue_depth=32;
    CHECK(fs::choose_policy(stats)==fs::Policy::GpuBatch);
    stats.gpu_available=false;
    CHECK(fs::choose_policy(stats)==fs::Policy::CpuLatency);
}
void switching_tests() {
    auto data=fs::Dataset::generate(100,9);
    fs::RuntimeConfig config; config.max_batch_size=8; config.max_wait=5s;
    // A real scalar backend stands in for GPU execution only in this routing test.
    fs::Runtime runtime(data,config,fs::make_scalar_backend(data),fs::make_scalar_backend(data));
    runtime.set_policy(fs::Policy::GpuBatch);
    std::vector<std::future<fs::Completion>> pending;
    pending.push_back(runtime.submit({std::vector<float>(9,.1f),1}));
    runtime.set_policy(fs::Policy::CpuLatency);
    CHECK(runtime.submit({std::vector<float>(9,.1f),1}).get().route==fs::Route::Cpu);
    CHECK(pending.front().wait_for(1ms)==std::future_status::timeout); // Queued route stays unchanged.
    runtime.set_policy(fs::Policy::GpuImmediate);
    CHECK(runtime.submit({std::vector<float>(9,.1f),1}).get().route==fs::Route::GpuImmediate);
    CHECK(pending.front().get().route==fs::Route::GpuBatch);
    runtime.set_policy(fs::Policy::Balanced);
    for(int i=0;i<100;++i) pending.push_back(runtime.submit({std::vector<float>(9,.1f),1}));
    std::jthread switcher([&] {
        for(int i=0;i<1000;++i) {
            runtime.set_policy(fs::Policy::GpuBatch); runtime.set_policy(fs::Policy::CpuLatency);
            if(i%100==0) (void)runtime.telemetry();
        }
    });
    std::size_t cpu=0,gpu=0;
    for(std::size_t i=1;i<pending.size();++i) {
        const auto value=pending[i].get();
        cpu+=value.route==fs::Route::Cpu; gpu+=value.route==fs::Route::GpuImmediate;
    }
    CHECK(cpu==50 && gpu==50); switcher.join(); runtime.shutdown();
    const auto stats=runtime.telemetry();
    CHECK(stats.windows[0].completed==103 && stats.pending_gpu_jobs==0);
    fs::Runtime cpu_runtime(data);
    invalid([&] { cpu_runtime.set_policy(fs::Policy::GpuBatch); });
    {
        fs::HeuristicController controller(cpu_runtime);
        for(int i=0;i<10;++i) cpu_runtime.submit({std::vector<float>(9,.1f),1}).get();
        // Controller must stay outside the hot path; shutdown wakes its periodic wait.
        controller.stop(); CHECK(controller.snapshot().errors==0);
    }
    CHECK(cpu_runtime.policy()==fs::Policy::CpuLatency);
    fs::SystemSampler sampler(false);
    const auto system=sampler.sample();
    CHECK(!system.gpu_utilization && !system.gpu_memory_free_bytes);
}
void trace_tests() {
    struct Temporary {
        std::filesystem::path path=std::filesystem::temp_directory_path()/
            ("flowstate-trace-"+std::to_string(fs::RollingTelemetry::Clock::now().time_since_epoch().count())+".csv");
        ~Temporary() { std::filesystem::remove(path); }
    } file;
    auto write=[&](const std::string& body) {
        std::ofstream out(file.path); out<<"name,duration_ms,requests_per_second,burst,top_k\n"<<body;
    };
    write("idle,1000,0,1,1\nburst,2000,500,32,10\n");
    const auto trace=fs::read_trace(file.path.string(),100);
    CHECK(trace.size()==2 && trace[0].requests_per_second==0 && trace[1].burst==32);
    for(const auto* row:{"bad,1000,2,1,101\n","bad,0,2,1,1\n","bad,1000,2,0,1\n",
                         "bad,1000,2,1,1,\n","bad,1000,-1,1,1\n","bad,1000,NaN,1,1\n",""}) {
        write(row); invalid([&] { fs::read_trace(file.path.string(),100); });
    }
}
}
int main() {
    try { metrics_tests(); policy_tests(); switching_tests(); trace_tests(); std::cout<<checks<<" scheduler checks passed\n"; }
    catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
