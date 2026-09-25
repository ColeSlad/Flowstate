#include "../cli.hpp"
#include "contention.hpp"
#include "flowstate/scheduler.hpp"
#ifdef FLOWSTATE_ENABLE_LAYA
#include "flowstate/laya.hpp"
#endif
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <csignal>
#include <fstream>
#include <semaphore>

namespace fs=flowstate;
using Json=nlohmann::json;
using Clock=std::chrono::steady_clock;
using namespace std::chrono_literals;
namespace {
std::atomic<bool> interrupted{false};
static_assert(std::atomic<bool>::is_always_lock_free);
void signal_handler(int) { interrupted.store(true,std::memory_order_relaxed); }
template<class T> Json nullable(const std::optional<T>& value) { return value ? Json(*value) : Json(nullptr); }
std::string read_file(const std::string& path) {
    std::ifstream input(path);
    if(!input) throw std::runtime_error("Cannot read asset: "+path);
    return {std::istreambuf_iterator<char>(input),{}};
}
struct Controls {
    std::string traffic="low";
    std::size_t top_k=10;
    bool contention=false;
    std::uint64_t version=0;
};
Json controls_json(const Controls& c) { return {{"traffic",c.traffic},{"top_k",c.top_k},{"contention",c.contention}}; }

class Demo {
public:
    Demo(fs::Runtime& runtime,std::vector<fs::Query> queries,std::size_t vectors,std::string controller,
         unsigned laya_port)
        : runtime_(runtime),queries_(std::move(queries)),vectors_(vectors),mode_(std::move(controller)),sampler_(runtime.has_gpu()) {
        controls_.top_k=queries_.front().top_k;
        if(runtime_.has_gpu()) contention_=fs::make_gpu_contention();
        if(mode_=="heuristic") {
            heuristic_=std::make_unique<fs::HeuristicController>(runtime_);
            controller_=heuristic_.get();
        } else {
#ifdef FLOWSTATE_ENABLE_LAYA
            fs::LayaConfig config; config.port=static_cast<std::uint16_t>(laya_port);
            laya_=std::make_unique<fs::LayaPolicyController>(runtime_,config);
            controller_=&laya_->controller();
#else
            (void)laya_port;
            throw std::invalid_argument("Laya requires FLOWSTATE_ENABLE_LAYA=ON");
#endif
        }
        publish();
        thread_=std::thread(&Demo::run,this);
    }
    ~Demo() {
        request_stop();
        if(thread_.joinable()) thread_.join();
        controller_->stop();
        contention_.reset(); // Remove competing GPU work before draining search.
        runtime_.shutdown();
    }
    void request_stop() { stopping_.store(true); wake_.notify_all(); }
    bool stopping() const { return stopping_.load(); }
    bool failed() const { return failed_.load(); }
    std::string snapshot() const { std::lock_guard lock(snapshot_mutex_); return snapshot_; }
    void wait_for_update() {
        std::unique_lock lock(controls_mutex_);
        wake_.wait_for(lock,500ms,[&] { return stopping(); });
    }
    Json update(const Json& input) {
        if(!input.is_object() || input.size()!=3 || !input.at("traffic").is_string() ||
           !input.at("top_k").is_number_unsigned() || !input.at("contention").is_boolean())
            throw std::invalid_argument("Expected traffic, top_k, and contention");
        Controls next;
        next.traffic=input.at("traffic").get<std::string>(); next.top_k=input.at("top_k").get<std::size_t>();
        next.contention=input.at("contention").get<bool>();
        if(next.traffic!="low" && next.traffic!="steady" && next.traffic!="burst") throw std::invalid_argument("Unknown traffic level");
        if(!next.top_k || next.top_k>std::min<std::size_t>(50,vectors_)) throw std::invalid_argument("top_k must be 1–50 and fit the dataset");
        if(next.contention && !contention_) throw std::invalid_argument("GPU contention requires CUDA");
        std::lock_guard lock(controls_mutex_);
        if(contention_) contention_->set_enabled(next.contention);
        next.version=controls_.version+1; controls_=next;
        wake_.notify_all();
        return controls_json(next);
    }
private:
    Controls controls() const { std::lock_guard lock(controls_mutex_); return controls_; }
    void publish() {
        const auto stats=runtime_.telemetry(); const auto totals=runtime_.snapshot();
        const auto system=sampler_.sample(); const auto control=controller_->snapshot();
        const auto contention=contention_?contention_->snapshot():fs::ContentionState{};
        const auto& window=stats.windows[0];
        const auto now=Clock::now();
        Json value={
            {"uptime_ms",std::chrono::duration<double,std::milli>(now-start_).count()},
            {"vectors",vectors_},{"dimension",queries_.front().values.size()},
            {"gpu_available",runtime_.has_gpu()},{"cpu_backend",fs::avx2_available()?"avx2":"scalar"},
            {"controller",mode_},{"policy",fs::policy_name(runtime_.policy())},{"policy_changes",control.changes},
            {"arrival_rate",window.arrival_rate},{"throughput",window.throughput},
            {"queue_depth",stats.queue_depth},{"pending_gpu_jobs",stats.pending_gpu_jobs},
            {"cpu_utilization",nullable(system.cpu_utilization)},{"gpu_utilization",nullable(system.gpu_utilization)},
            {"gpu_memory_free_bytes",nullable(system.gpu_memory_free_bytes)},
            {"p95_ms",nullable(window.latency.p95_ms)},{"p99_ms",nullable(window.latency.p99_ms)},
            {"cpu_completed",window.backends[0].count+window.backends[1].count},
            {"gpu_completed",window.backends[2].count+window.backends[3].count},
            {"mean_batch_size",nullable(window.mean_gpu_batch_size)},
            {"total_completed",totals.completed},{"rejected",totals.rejected},{"failed",totals.failed},
            {"slo_ms",15},{"slo_violations",late_+totals.failed+totals.rejected},
            {"controls",controls_json(controls())},
            {"contention",{{"enabled",contention.enabled},{"active",contention.active},
                {"kernels",contention.kernels},{"error",contention.error}}},
            {"laya",{{"calls",control.laya_calls},{"fallbacks",control.laya_fallbacks},{"errors",control.laya_errors},
                {"confidence",control.last_laya?nullable(control.last_laya->confidence):Json(nullptr)},
                {"latency_ms",control.last_laya?nullable(control.last_laya->latency_ms):Json(nullptr)},
                {"status",control.last_laya?control.last_laya->status:"inactive"},
                {"fallback",control.last_laya && control.last_laya->fallback}}},
            {"control_age_ms",control.sampled_at==Clock::time_point{}?Json(nullptr):
                Json(std::chrono::duration<double,std::milli>(now-control.sampled_at).count())}
        };
        auto encoded=value.dump();
        std::lock_guard lock(snapshot_mutex_); snapshot_=std::move(encoded);
    }
    void run() {
        try {
            std::vector<std::future<fs::Completion>> pending;
            pending.reserve(1100);
            auto due=Clock::now(),next_sample=due;
            auto previous=controls();
            std::size_t query_index=0;
            while(!stopping()) {
                for(std::size_t i=0;i<pending.size();) {
                    if(pending[i].wait_for(0s)!=std::future_status::ready) { ++i; continue; }
                    try { late_+=pending[i].get().latency_ms>15; } catch(const std::exception&) { /* Runtime records failures. */ }
                    if(i+1<pending.size()) pending[i]=std::move(pending.back());
                    pending.pop_back();
                }
                const auto current=controls();
                const auto now=Clock::now();
                if(current.version!=previous.version) { due=now; previous=current; }
                if(now>=next_sample) { publish(); next_sample=now+500ms; }
                if(now>=due) {
                    const unsigned rate=current.traffic=="low"?10:current.traffic=="steady"?150:1200;
                    const unsigned burst=current.traffic=="burst"?32:1;
                    for(unsigned i=0;i<burst;++i) {
                        auto query=queries_[query_index++%queries_.size()]; query.top_k=current.top_k;
                        try { pending.push_back(runtime_.submit(std::move(query))); }
                        catch(const fs::QueueFull&) { /* Visible in telemetry. */ }
                    }
                    due=now+std::chrono::nanoseconds(1000000000ULL*burst/rate);
                }
                std::unique_lock lock(controls_mutex_);
                wake_.wait_until(lock,std::min(due,Clock::now()+1ms),[&] { return stopping() || controls_.version!=current.version; });
            }
            // Runtime drains accepted requests on destruction; destroying futures never cancels jobs.
        } catch(const std::exception& error) {
            std::cerr<<"Demo workload: "<<error.what()<<'\n'; failed_.store(true); request_stop();
        }
    }
    fs::Runtime& runtime_;
    std::vector<fs::Query> queries_;
    std::size_t vectors_;
    std::string mode_;
    fs::SystemSampler sampler_;
    std::unique_ptr<fs::GpuContention> contention_;
    std::unique_ptr<fs::HeuristicController> heuristic_;
#ifdef FLOWSTATE_ENABLE_LAYA
    std::unique_ptr<fs::LayaPolicyController> laya_;
#endif
    fs::PolicyController* controller_=nullptr;
    Clock::time_point start_=Clock::now();
    mutable std::mutex controls_mutex_,snapshot_mutex_;
    std::condition_variable wake_;
    Controls controls_;
    std::string snapshot_;
    std::uint64_t late_=0;
    std::atomic<bool> stopping_{false},failed_{false};
    std::thread thread_;
};
struct StreamLease {
    std::counting_semaphore<4>& slots;
    bool acquired;
    explicit StreamLease(std::counting_semaphore<4>& value):slots(value),acquired(slots.try_acquire()) {}
    ~StreamLease() { if(acquired) slots.release(); }
};
bool local_authority(std::string_view host) {
    if(host=="localhost" || host=="127.0.0.1") return true;
    const auto colon=host.find(':');
    if(colon==std::string_view::npos || (host.substr(0,colon)!="localhost" && host.substr(0,colon)!="127.0.0.1")) return false;
    const auto port=host.substr(colon+1);
    return !port.empty() && port.size()<=5 && port.find_first_not_of("0123456789")==std::string_view::npos;
}
}
int main(int argc,char** argv) {
    try {
        fs::cli::Options options; options.batch_size=32;
        std::string cuda="auto",mode="heuristic";
        unsigned port=8080,laya_port=8000;
        for(int i=1;i<argc;++i) {
            const std::string_view key(argv[i]);
            if(key=="--help") {
                std::cout<<"flowstate_server --vectors N --dimension N --top-k N --seed N\n"
                    "  --cuda auto|on|off --controller heuristic|laya --port N --laya-port N\n"; return 0;
            }
            if(i+1==argc) throw std::invalid_argument("Missing option value");
            const std::string_view value(argv[++i]);
            if(key=="--cuda") cuda=value;
            else if(key=="--controller") mode=value;
            else {
                const auto number=fs::cli::number(value);
                if(key=="--vectors") options.vectors=number;
                else if(key=="--dimension") options.dimension=number;
                else if(key=="--top-k") options.top_k=number;
                else if(key=="--seed" && number<=UINT32_MAX) options.seed=static_cast<std::uint32_t>(number);
                else if((key=="--port" || key=="--laya-port") && number>0 && number<=65535) {
                    (key=="--port"?port:laya_port)=static_cast<unsigned>(number);
                } else throw std::invalid_argument("Invalid option: "+std::string(key));
            }
        }
        if(!options.vectors || options.vectors>1000000 || !options.dimension || options.dimension>2048 ||
           fs::checked_product(options.vectors,options.dimension)>200000000 || !options.top_k ||
           options.top_k>std::min<std::size_t>(50,options.vectors)) throw std::invalid_argument("Invalid demo dataset or top-k");
        if(cuda!="auto" && cuda!="on" && cuda!="off") throw std::invalid_argument("Unknown CUDA mode");
        if(mode!="heuristic" && mode!="laya") throw std::invalid_argument("Unknown controller");
        const auto dataset=fs::Dataset::generate(options.vectors,options.dimension,options.seed);
        auto queries=fs::cli::queries(options);
        auto cpu=fs::cli::backend("auto",dataset);
        fs::RuntimeConfig config; config.enable_cuda=cuda=="on" || (cuda=="auto" && fs::cuda_available());
        auto gpu=config.enable_cuda?fs::make_cuda_backend(dataset):nullptr;
        for(int i=0;i<3;++i) { cpu->search_batch(queries); if(gpu) gpu->search_batch(queries); }
        fs::Runtime runtime(dataset,config,std::move(cpu),std::move(gpu));
        bool failed=false;
        {
            Demo demo(runtime,std::move(queries),options.vectors,mode,laya_port);
            std::counting_semaphore<4> streams(4);
            httplib::Server server;
            server.new_task_queue=[] { return new httplib::ThreadPool(8,8,16); };
            server.set_payload_max_length(2048).set_read_timeout(2,0).set_write_timeout(2,0).set_keep_alive_timeout(2);
            server.set_default_headers({{"Cache-Control","no-store"},{"X-Content-Type-Options","nosniff"},
                {"Content-Security-Policy","default-src 'self'; connect-src 'self'; frame-ancestors 'none'; object-src 'none'"}});
            server.set_pre_routing_handler([](const httplib::Request& request,httplib::Response& response) {
                const auto host=request.get_header_value("Host");
                if(!local_authority(host) || (request.has_header("Origin") && request.get_header_value("Origin")!="http://"+host)) {
                    response.status=403; response.set_content("Local origin required","text/plain"); return httplib::Server::HandlerResponse::Handled;
                }
                return httplib::Server::HandlerResponse::Unhandled;
            });
            for(const auto& asset:std::vector<std::pair<std::string,std::string>>{{"index.html","text/html"},{"app.js","text/javascript"},{"style.css","text/css"},{"favicon.svg","image/svg+xml"}}) {
                const auto content=read_file(std::string(FLOWSTATE_WEB_DIR)+"/"+asset.first);
                server.Get(asset.first=="index.html"?"/":"/"+asset.first,[content,type=asset.second](const auto&,auto& response) { response.set_content(content,type); });
            }
            const auto comparison=read_file(FLOWSTATE_COMPARISON_FILE);
            server.Get("/api/comparison",[comparison](const auto&,auto& response) { response.set_content(comparison,"application/json"); });
            server.Get("/api/state",[&](const auto&,auto& response) { response.set_content(demo.snapshot(),"application/json"); });
            server.Post("/api/controls",[&](const auto& request,auto& response) {
                if(!request.get_header_value("Content-Type").starts_with("application/json")) { response.status=415; return; }
                try {
                    const auto input=Json::parse(request.body,[](int depth,Json::parse_event_t,const Json&) {
                        if(depth>4) throw std::invalid_argument("Control object is too deep");
                        return true;
                    });
                    response.set_content(demo.update(input).dump(),"application/json");
                } catch(const std::exception& error) {
                    response.status=400; response.set_content(Json({{"error",error.what()}}).dump(),"application/json");
                }
            });
            server.Get("/events",[&](const auto&,auto& response) {
                auto lease=std::make_shared<StreamLease>(streams);
                if(!lease->acquired) { response.status=503; response.set_content("Stream limit reached","text/plain"); return; }
                response.set_chunked_content_provider("text/event-stream",[&demo,lease](std::size_t,httplib::DataSink& sink) {
                    if(demo.stopping()) return false;
                    const auto frame="data: "+demo.snapshot()+"\n\n";
                    if(!sink.write(frame.data(),frame.size())) return false;
                    demo.wait_for_update(); return !demo.stopping();
                });
            });
            if(!server.bind_to_port("127.0.0.1",static_cast<int>(port))) throw std::runtime_error("Cannot bind dashboard port");
            std::signal(SIGINT,signal_handler); std::signal(SIGTERM,signal_handler);
            std::jthread monitor([&](std::stop_token stop) {
                while(!stop.stop_requested() && !interrupted && !demo.failed()) std::this_thread::sleep_for(50ms);
                if(interrupted || demo.failed()) { demo.request_stop(); server.stop(); }
            });
            std::cout<<"Flowstate: http://127.0.0.1:"<<port<<" ("<<mode<<", "<<(runtime.has_gpu()?"CUDA":"CPU only")<<")"<<std::endl;
            failed=!server.listen_after_bind() || demo.failed();
            demo.request_stop(); monitor.request_stop(); server.stop();
        }
        const auto final=runtime.snapshot();
        std::cout<<Json({{"submitted",final.submitted},{"completed",final.completed},{"failed",final.failed}}).dump()<<std::endl;
        return failed || final.failed || final.submitted!=final.completed ? 1 : 0;
    } catch(const std::exception& error) { std::cerr<<"flowstate_server: "<<error.what()<<'\n'; return 1; }
}
