#include "../src/cli.hpp"
#include "flowstate/scheduler.hpp"
#include "flowstate/workload.hpp"
#ifdef FLOWSTATE_ENABLE_LAYA
#include "flowstate/laya.hpp"
#endif
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numeric>

namespace fs=flowstate;
using Clock=std::chrono::steady_clock;
using namespace std::chrono_literals;
namespace {
struct Pending { std::future<fs::Completion> future; double lag_ms; };
double percentile(std::vector<double> values,double p) {
    if(values.empty()) return 0;
    const auto rank=static_cast<std::size_t>(std::ceil(p*values.size()))-1;
    std::nth_element(values.begin(),values.begin()+rank,values.end()); return values[rank];
}
template<class T> void optional(std::ostream& out,const std::optional<T>& value) {
    if(value && std::isfinite(static_cast<double>(*value))) out<<*value;
}
}
int main(int argc,char** argv) {
    try {
        fs::cli::Options options; options.batch_size=32;
        fs::RuntimeConfig config;
        fs::HeuristicConfig thresholds;
        std::string mode="heuristic",cuda="auto",cpu_backend="auto",trace="benchmarks/traces/adaptive.csv",telemetry_path;
        std::size_t interval_ms=500,slo_ms=15;
        bool interval_set=false;
#ifdef FLOWSTATE_ENABLE_LAYA
        fs::LayaConfig laya_config;
#endif
        for(int i=1;i<argc;++i) {
            const std::string_view key(argv[i]);
            if(key=="--help") {
                std::cout<<"--trace CSV --mode heuristic|laya|cpu_latency|gpu_immediate|gpu_batch|balanced\n"
                  "--vectors N --dimension D --seed N --workers N --queue-capacity N\n"
                  "--batch-size N --max-wait-us N --cuda auto|on|off --cpu-backend auto|scalar|avx2\n"
                  "--telemetry CSV --slo-ms N --interval-ms N --low-rate N --balanced-rate N --batch-rate N --high-queue N\n"
                  "Laya build: --laya-port N --laya-timeout-ms N --laya-min-confidence FRACTION; local inference server on 127.0.0.1\n";
                return 0;
            }
            if(i+1==argc) throw std::invalid_argument("Missing option value");
            const std::string_view value(argv[++i]);
            if(key=="--mode") mode=value;
            else if(key=="--trace") trace=value;
            else if(key=="--cuda") cuda=value;
            else if(key=="--cpu-backend") cpu_backend=value;
            else if(key=="--telemetry") telemetry_path=value;
            else if(key=="--vectors") options.vectors=fs::cli::number(value);
            else if(key=="--dimension") options.dimension=fs::cli::number(value);
            else if(key=="--seed") {
                auto seed=fs::cli::number(value);
                if(seed>UINT32_MAX) throw std::invalid_argument("Seed exceeds uint32");
                options.seed=static_cast<std::uint32_t>(seed);
            }
            else if(key=="--workers") config.cpu_workers=fs::cli::number(value);
            else if(key=="--queue-capacity") config.queue_capacity=fs::cli::number(value);
            else if(key=="--batch-size") config.max_batch_size=fs::cli::number(value);
            else if(key=="--max-wait-us") {
                const auto wait=fs::cli::number(value);
                if(wait>60000000) throw std::invalid_argument("Maximum wait is 60 seconds");
                config.max_wait=std::chrono::microseconds(wait);
            }
            else if(key=="--interval-ms") { interval_ms=fs::cli::number(value); interval_set=true; }
#ifdef FLOWSTATE_ENABLE_LAYA
            else if(key=="--laya-port") {
                auto port=fs::cli::number(value);
                if(port==0 || port>65535) throw std::invalid_argument("Invalid Laya port");
                laya_config.port=static_cast<std::uint16_t>(port);
            }
            else if(key=="--laya-timeout-ms") {
                auto timeout=fs::cli::number(value);
                if(timeout>1000) throw std::invalid_argument("Laya timeout exceeds 1000 ms");
                laya_config.timeout=std::chrono::milliseconds(timeout);
            }
            else if(key=="--laya-min-confidence") {
                const auto [end,error]=std::from_chars(value.data(),value.data()+value.size(),laya_config.min_confidence);
                if(error!=std::errc{} || end!=value.data()+value.size()) throw std::invalid_argument("Invalid confidence threshold");
            }
#endif
            else if(key=="--slo-ms") slo_ms=fs::cli::number(value);
            else if(key=="--low-rate") thresholds.low_rate=fs::cli::number(value);
            else if(key=="--balanced-rate") thresholds.balanced_rate=fs::cli::number(value);
            else if(key=="--batch-rate") thresholds.batch_rate=fs::cli::number(value);
            else if(key=="--high-queue") thresholds.high_queue=fs::cli::number(value);
            else throw std::invalid_argument("Unknown option: "+std::string(key));
        }
#ifndef NDEBUG
        throw std::runtime_error("Benchmark in Release mode");
#endif
        if(!options.vectors || !options.dimension || !slo_ms || interval_ms<500 || interval_ms>2000)
            throw std::invalid_argument("Invalid dataset, SLO, or control interval");
        thresholds.validate();
        if(cuda!="auto" && cuda!="on" && cuda!="off") throw std::invalid_argument("Invalid CUDA mode");
        if(cpu_backend!="auto" && cpu_backend!="scalar" && cpu_backend!="avx2") throw std::invalid_argument("Invalid CPU backend");
        const auto static_policy=mode=="heuristic" || mode=="laya" ? fs::Policy::CpuLatency:fs::parse_policy(mode);
#ifndef FLOWSTATE_ENABLE_LAYA
        if(mode=="laya") throw std::invalid_argument("Configure with FLOWSTATE_ENABLE_LAYA=ON for Laya");
        (void)interval_set;
#endif
        const auto phases=fs::read_trace(trace,options.vectors);
        std::uint64_t scheduled=0;
        for(const auto& phase:phases) scheduled+=phase.requests_per_second*phase.duration_ms/1000;
        if(scheduled>2000000) throw std::invalid_argument("Trace exceeds two million requests");
        config.enable_cuda=cuda=="on" || (cuda=="auto" && fs::cuda_available());
        if(!config.enable_cuda && static_policy!=fs::Policy::CpuLatency) throw std::invalid_argument("Selected policy requires CUDA");
        const auto dataset=fs::Dataset::generate(options.vectors,options.dimension,options.seed);
        options.top_k=std::min<std::size_t>(10,options.vectors);
        auto queries=fs::cli::queries(options);
        auto cpu=fs::cli::backend(cpu_backend,dataset);
        const std::string cpu_name(cpu->name());
        auto gpu=config.enable_cuda?fs::make_cuda_backend(dataset):nullptr;
        // Warm actual backend instances before moving them into an empty runtime.
        for(int i=0;i<3;++i) { cpu->search_batch(queries); if(gpu) gpu->search_batch(queries); }
        fs::Runtime runtime(dataset,config,std::move(cpu),std::move(gpu));
        runtime.set_policy(static_policy);
        std::unique_ptr<fs::HeuristicController> heuristic;
        fs::PolicyController* controller=nullptr;
        if(mode=="heuristic") {
            heuristic=std::make_unique<fs::HeuristicController>(runtime,thresholds,std::chrono::milliseconds(interval_ms));
            controller=heuristic.get();
        }
#ifdef FLOWSTATE_ENABLE_LAYA
        std::unique_ptr<fs::LayaPolicyController> laya;
        if(mode=="laya") {
            if(interval_set) laya_config.interval=std::chrono::milliseconds(interval_ms);
            laya_config.slo_ms=slo_ms;
            laya=std::make_unique<fs::LayaPolicyController>(runtime,laya_config,thresholds);
            controller=&laya->controller();
        }
#endif
        fs::SystemSampler sampler(config.enable_cuda);
        sampler.sample();
        std::ofstream telemetry;
        if(!telemetry_path.empty()) {
            telemetry.open(telemetry_path);
            if(!telemetry) throw std::invalid_argument("Cannot write telemetry CSV");
            telemetry<<"elapsed_ms,phase,policy,arrival_rate,throughput,queue_depth,pending_gpu_jobs,cpu_utilization,gpu_utilization,gpu_memory_free_bytes,p50_ms,p95_ms,p99_ms,scalar_p95_ms,simd_p95_ms,cuda_p95_ms,gpu_batch_p95_ms,mean_gpu_batch_size,policy_changes,controller_errors,sample_age_ms,laya_calls,laya_fallbacks,laya_errors,laya_confidence,laya_latency_ms,laya_status\n";
        }
        std::vector<Pending> pending;
        std::vector<double> latency,offered_latency;
        latency.reserve(scheduled); offered_latency.reserve(scheduled);
        std::uint64_t offered=0,rejected=0,failed=0,cpu_count=0,gpu_count=0,checksum=0,late=0;
        double max_lag_ms=0;
        auto collect=[&] {
            for(std::size_t index=0;index<pending.size();) {
                auto& item=pending[index];
                if(item.future.wait_for(0s)!=std::future_status::ready) { ++index; continue; }
                try {
                    auto value=item.future.get();
                    latency.push_back(value.latency_ms); offered_latency.push_back(value.latency_ms+item.lag_ms);
                    late+=value.latency_ms+item.lag_ms>slo_ms;
                    cpu_count+=value.route==fs::Route::Cpu; gpu_count+=value.route!=fs::Route::Cpu;
                    for(auto id:value.result.ids) checksum+=id;
                } catch(const std::exception&) { ++failed; }
                if(index+1<pending.size()) item=std::move(pending.back());
                pending.pop_back();
            }
        };
        const auto start=Clock::now();
        auto next_sample=start;
        auto sample=[&](std::string_view phase) {
            if(!telemetry) return;
            fs::ControllerSnapshot state;
            if(controller) state=controller->snapshot();
            else { state.stats=runtime.telemetry(); state.stats.system=sampler.sample(); state.policy=runtime.policy(); state.sampled_at=Clock::now(); }
            if(phase=="drained") { const auto system=state.stats.system; state.stats=runtime.telemetry(); state.stats.system=system; }
            const auto now=Clock::now();
            const auto& stats=state.stats; const auto& window=stats.windows[0];
            telemetry<<std::fixed<<std::setprecision(6)<<std::chrono::duration<double,std::milli>(now-start).count()<<','<<phase<<','
                <<fs::policy_name(state.policy)<<','<<window.arrival_rate<<','<<window.throughput<<','<<stats.queue_depth<<','<<stats.pending_gpu_jobs<<',';
            optional(telemetry,stats.system.cpu_utilization); telemetry<<','; optional(telemetry,stats.system.gpu_utilization); telemetry<<',';
            optional(telemetry,stats.system.gpu_memory_free_bytes); telemetry<<','; optional(telemetry,window.latency.p50_ms); telemetry<<',';
            optional(telemetry,window.latency.p95_ms); telemetry<<','; optional(telemetry,window.latency.p99_ms);
            for(const auto& backend:window.backends) { telemetry<<','; optional(telemetry,backend.p95_ms); }
            telemetry<<','; optional(telemetry,window.mean_gpu_batch_size);
            telemetry<<','<<state.changes<<','<<state.errors<<',';
            if(state.sampled_at!=Clock::time_point{}) telemetry<<std::chrono::duration<double,std::milli>(now-state.sampled_at).count();
            telemetry<<','<<state.laya_calls<<','<<state.laya_fallbacks<<','<<state.laya_errors<<',';
            if(state.last_laya) optional(telemetry,state.last_laya->confidence);
            telemetry<<',';
            if(state.last_laya) optional(telemetry,state.last_laya->latency_ms);
            telemetry<<',';
            if(state.last_laya) telemetry<<state.last_laya->status;
            telemetry<<'\n';
        };
        auto phase_start=start;
        for(const auto& phase:phases) {
            const auto end=phase_start+std::chrono::milliseconds(phase.duration_ms);
            const auto count=phase.requests_per_second*phase.duration_ms/1000;
            std::uint64_t emitted=0;
            while(Clock::now()<end || emitted<count) {
                collect();
                auto now=Clock::now();
                if(now>=next_sample) { sample(phase.name); next_sample=now+500ms; }
                if(emitted<count) {
                    const auto due=phase_start+std::chrono::nanoseconds(emitted*1000000000ULL/phase.requests_per_second);
                    if(now>=due) {
                        const auto batch=std::min<std::uint64_t>(phase.burst,count-emitted);
                        for(std::uint64_t i=0;i<batch;++i) {
                            auto query=queries[offered%queries.size()]; query.top_k=phase.top_k;
                            const auto lag=std::max(0.0,std::chrono::duration<double,std::milli>(Clock::now()-due).count());
                            max_lag_ms=std::max(max_lag_ms,lag); ++offered;
                            try { pending.push_back({runtime.submit(std::move(query)),lag}); }
                            catch(const fs::QueueFull&) { ++rejected; }
                        }
                        emitted+=batch;
                        continue;
                    }
                    std::this_thread::sleep_until(std::min({due,next_sample,now+1ms}));
                } else if(now<end) std::this_thread::sleep_until(std::min({end,next_sample,now+1ms}));
            }
            phase_start=end;
        }
        runtime.shutdown(); // Complete every accepted request, including the final partial batch.
        collect();
        const auto finished=Clock::now();
        if(controller) controller->stop();
        const auto seconds=std::chrono::duration<double>(finished-start).count();
        sample("drained");
        if(telemetry.is_open()) { telemetry.flush(); if(!telemetry) throw std::runtime_error("Telemetry write failed"); }
        const auto changes=controller?controller->snapshot().changes:0;
        const auto errors=controller?controller->snapshot().errors:0;
        const auto final=runtime.snapshot();
        if(!pending.empty() || offered!=scheduled || latency.size()+failed+rejected!=offered || final.submitted!=final.completed)
            throw std::runtime_error("Load accounting mismatch");
        std::cout<<"mode,cpu_backend,vectors,dimension,seed,workers,batch_max,max_wait_us,queue_capacity,offered,completed,rejected,failed,elapsed_seconds,queries_per_second,p50_ms,p95_ms,p99_ms,offered_p99_ms,slo_ms,slo_violations,cpu_completed,gpu_completed,policy_changes,controller_errors,max_producer_lag_ms,checksum,laya_calls,laya_fallbacks,laya_errors,mean_laya_latency_ms,max_laya_latency_ms\n"
          <<std::fixed<<std::setprecision(6)<<mode<<','<<cpu_name<<','<<options.vectors<<','<<options.dimension<<','<<options.seed<<','
          <<config.cpu_workers<<','<<config.max_batch_size<<','<<config.max_wait.count()<<','<<config.queue_capacity<<','<<offered<<','
          <<latency.size()<<','<<rejected<<','<<failed<<','<<seconds<<','<<latency.size()/seconds<<',';
        if(!latency.empty()) std::cout<<percentile(latency,.5)<<','<<percentile(latency,.95)<<','<<percentile(latency,.99)<<','<<percentile(offered_latency,.99);
        else std::cout<<",,,";
        std::cout<<','<<slo_ms<<','<<late+failed+rejected<<','
          <<cpu_count<<','<<gpu_count<<','<<changes<<','<<errors<<','<<max_lag_ms<<','<<checksum;
        const auto control=controller?controller->snapshot():fs::ControllerSnapshot{};
        std::cout<<','<<control.laya_calls<<','<<control.laya_fallbacks<<','<<control.laya_errors<<',';
        if(control.laya_calls) std::cout<<control.total_laya_latency_ms/control.laya_calls;
        std::cout<<',';
        if(control.laya_calls) std::cout<<control.max_laya_latency_ms;
        std::cout<<'\n';
        return failed || errors ? 1 : 0;
    } catch(const std::exception& e) { std::cerr<<"flowstate_load: "<<e.what()<<'\n'; return 1; }
}
