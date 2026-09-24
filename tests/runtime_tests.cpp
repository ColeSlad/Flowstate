#include "flowstate/runtime.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <set>

namespace fs = flowstate;
using namespace std::chrono_literals;
namespace {
std::size_t checks = 0;
#define CHECK(condition) do { ++checks; if (!(condition)) throw std::runtime_error( \
    std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " #condition); } while (false)
template<class Exception, class Function> void throws(Function function) {
    ++checks;
    try { function(); } catch (const Exception&) { return; }
    throw std::runtime_error("Expected exception was not thrown");
}
fs::Query query(std::size_t dimension = 9) { return {std::vector<float>(dimension, 0.25f), 3}; }

struct Gate {
    std::mutex mutex;
    std::condition_variable ready;
    bool hold = false;
    bool fail_next = false;
    bool short_batch = false;
    std::size_t entries = 0;
    std::vector<std::size_t> batches;
    std::set<std::thread::id> threads;
    void release() { std::lock_guard lock(mutex); hold = false; ready.notify_all(); }
    void await_entries(std::size_t count) {
        std::unique_lock lock(mutex);
        if (!ready.wait_for(lock, 3s, [&] { return entries >= count; }))
            throw std::runtime_error("Worker did not enter test backend");
    }
};
struct ReleaseGate { std::shared_ptr<Gate> gate; ~ReleaseGate() { gate->release(); } };

class ControlledBackend final : public fs::Backend {
public:
    ControlledBackend(std::shared_ptr<const fs::Dataset> dataset, std::shared_ptr<Gate> gate)
        : scalar_(fs::make_scalar_backend(std::move(dataset))), gate_(std::move(gate)) {}
    fs::SearchResult search(const fs::Query& q) override {
        auto batch = search_batch(std::span<const fs::Query>(&q, 1));
        return std::move(batch.results.front());
    }
    fs::BatchResult search_batch(std::span<const fs::Query> queries) override {
        bool short_batch;
        {
            std::unique_lock lock(gate_->mutex);
            ++gate_->entries;
            gate_->batches.push_back(queries.size());
            gate_->threads.insert(std::this_thread::get_id());
            gate_->ready.notify_all();
            if (!gate_->ready.wait_for(lock, 5s, [&] { return !gate_->hold; })) {
                gate_->hold = false;
                gate_->ready.notify_all();
                throw std::runtime_error("Test gate timed out");
            }
            if (gate_->fail_next) { gate_->fail_next = false; throw std::runtime_error("Injected backend failure"); }
            short_batch = gate_->short_batch;
        }
        auto result = scalar_->search_batch(queries);
        if (short_batch && !result.results.empty()) result.results.pop_back();
        return result;
    }
    std::string_view name() const override { return "controlled"; }
private:
    std::unique_ptr<fs::Backend> scalar_;
    std::shared_ptr<Gate> gate_;
};

auto controlled(std::shared_ptr<const fs::Dataset> data, std::shared_ptr<Gate> gate) {
    return std::make_unique<ControlledBackend>(std::move(data), std::move(gate));
}

void validate(const fs::Completion& completion, const fs::SearchResult& baseline) {
    CHECK(completion.request_id > 0);
    CHECK(completion.result.ids == baseline.ids);
    for (std::size_t i = 0; i < baseline.scores.size(); ++i)
        CHECK(std::abs(completion.result.scores[i] - baseline.scores[i]) < 1e-3);
    CHECK(completion.queue_wait_ms >= 0);
    CHECK(completion.execution_ms >= 0);
    CHECK(completion.latency_ms + 1e-6 >= completion.queue_wait_ms + completion.execution_ms);
}

void runtime_tests() {
    const auto data = fs::Dataset::generate(257, 9);
    const auto baseline = fs::make_scalar_backend(data)->search(query());
    fs::RuntimeConfig config;
    config.cpu_workers = 4;
    throws<std::invalid_argument>([&] { auto bad=config; bad.cpu_workers=0; fs::Runtime runtime(data,bad); });
    throws<std::invalid_argument>([&] { auto bad=config; bad.queue_capacity=0; fs::Runtime runtime(data,bad); });
    throws<std::invalid_argument>([&] { auto bad=config; bad.max_batch_size=0; fs::Runtime runtime(data,bad); });
    throws<std::invalid_argument>([&] { auto bad=config; bad.max_wait=-1us; fs::Runtime runtime(data,bad); });
    throws<std::invalid_argument>([&] { auto bad=config; bad.max_wait=61s; fs::Runtime runtime(data,bad); });
    {
        fs::Runtime runtime(data, config);
        throws<std::invalid_argument>([&] { runtime.submit(query(8)); });
        throws<std::invalid_argument>([&] { runtime.submit(query(), fs::Route::GpuBatch); });
        std::mutex futures_mutex;
        std::vector<std::future<fs::Completion>> futures;
        std::vector<std::jthread> producers;
        for (int producer=0; producer<4; ++producer) producers.emplace_back([&] {
            for (int i=0; i<64; ++i) {
                auto future=runtime.submit(query());
                std::lock_guard lock(futures_mutex);
                futures.push_back(std::move(future));
            }
        });
        producers.clear();
        // Simultaneous shutdown callers must drain accepted jobs and join only once.
        std::jthread stop_a([&] { runtime.shutdown(); });
        std::jthread stop_b([&] { runtime.shutdown(); });
        stop_a.join(); stop_b.join();
        std::set<std::uint64_t> ids;
        for (auto& future : futures) { const auto value=future.get(); validate(value,baseline); ids.insert(value.request_id); }
        CHECK(ids.size()==256);
        const auto stats=runtime.snapshot();
        CHECK(stats.submitted==256 && stats.completed==256 && stats.failed==0);
        CHECK(stats.cpu_inflight==0 && stats.cpu_queue_depth==0);
        throws<fs::RuntimeStopped>([&] { runtime.submit(query()); });
    }
    {
        auto cpu_gate=std::make_shared<Gate>(); cpu_gate->hold=true;
        auto gpu_gate=std::make_shared<Gate>();
        config.cpu_workers=1; config.queue_capacity=2; config.max_batch_size=4; config.max_wait=5s;
        fs::Runtime runtime(data,config,controlled(data,cpu_gate),controlled(data,gpu_gate));
        ReleaseGate release{cpu_gate};
        auto active=runtime.submit(query()); cpu_gate->await_entries(1);
        auto queued_cpu=runtime.submit(query());
        auto queued_gpu=runtime.submit(query(),fs::Route::GpuBatch);
        throws<fs::QueueFull>([&] { runtime.submit(query()); });
        throws<fs::QueueFull>([&] { runtime.submit(query(),fs::Route::GpuImmediate); });
        const auto stats=runtime.snapshot();
        CHECK(stats.cpu_queue_depth+stats.gpu_queue_depth==2);
        CHECK(stats.max_queue_depth==2 && stats.rejected==2 && stats.cpu_inflight==1);
        cpu_gate->release();
        runtime.shutdown(); // Must flush the partial GPU batch without its 5-second delay.
        validate(active.get(),baseline); validate(queued_cpu.get(),baseline); validate(queued_gpu.get(),baseline);
        CHECK(runtime.snapshot().completed==3);
    }
    config.queue_capacity=128; config.max_batch_size=4;
    {
        auto gate=std::make_shared<Gate>();
        fs::Runtime runtime(data,config,fs::make_scalar_backend(data),controlled(data,gate));
        std::vector<std::future<fs::Completion>> futures;
        for(int i=0;i<4;++i) futures.push_back(runtime.submit(query(),fs::Route::GpuBatch));
        CHECK(futures[0].wait_for(3s)==std::future_status::ready);
        for(auto& future:futures) { auto result=future.get(); validate(result,baseline); CHECK(result.batch_size==4);
            CHECK(result.latency_ms > result.queue_wait_ms + result.execution_ms + 1e-6); }
        runtime.shutdown();
        CHECK(runtime.snapshot().size_flushes==1 && runtime.snapshot().timeout_flushes==0);
        CHECK(gate->batches==std::vector<std::size_t>{4} && gate->threads.size()==1);
    }
    {
        config.max_wait=20ms;
        auto gate=std::make_shared<Gate>();
        fs::Runtime runtime(data,config,fs::make_scalar_backend(data),controlled(data,gate));
        auto future=runtime.submit(query(),fs::Route::GpuBatch);
        CHECK(future.wait_for(3s)==std::future_status::ready);
        auto result=future.get(); validate(result,baseline);
        CHECK(result.batch_size==1 && result.queue_wait_ms>=19.99);
        CHECK(runtime.snapshot().timeout_flushes==1);
    }
    {
        config.max_wait=5s;
        auto gate=std::make_shared<Gate>();
        fs::Runtime runtime(data,config,fs::make_scalar_backend(data),controlled(data,gate));
        auto partial=runtime.submit(query(),fs::Route::GpuBatch);
        auto immediate=runtime.submit(query(),fs::Route::GpuImmediate);
        CHECK(immediate.wait_for(3s)==std::future_status::ready);
        validate(partial.get(),baseline); validate(immediate.get(),baseline);
        runtime.shutdown();
        CHECK(gate->batches==std::vector<std::size_t>({1,1}));
        CHECK(runtime.snapshot().timeout_flushes==0);
    }
    for(bool short_batch : {false,true}) {
        config.max_batch_size=2;
        auto gate=std::make_shared<Gate>(); gate->fail_next=!short_batch; gate->short_batch=short_batch;
        fs::Runtime runtime(data,config,fs::make_scalar_backend(data),controlled(data,gate));
        auto first=runtime.submit(query(),fs::Route::GpuBatch);
        auto second=runtime.submit(query(),fs::Route::GpuBatch);
        throws<std::runtime_error>([&] { first.get(); }); throws<std::runtime_error>([&] { second.get(); });
        { std::lock_guard lock(gate->mutex); gate->short_batch=false; }
        validate(runtime.submit(query(),fs::Route::GpuImmediate).get(),baseline);
        runtime.shutdown();
        const auto stats=runtime.snapshot();
        CHECK(stats.completed==3 && stats.failed==2 && stats.gpu_inflight==0);
    }
    {
        auto gate=std::make_shared<Gate>(); gate->fail_next=true;
        fs::Runtime runtime(data,config,controlled(data,gate),nullptr);
        throws<std::runtime_error>([&] { runtime.submit(query()).get(); });
        validate(runtime.submit(query()).get(),baseline);
        CHECK(runtime.snapshot().failed==1);
    }
    {
        config.max_batch_size=2; config.queue_capacity=128;
        auto gate=std::make_shared<Gate>(); gate->hold=true;
        fs::Runtime runtime(data,config,fs::make_scalar_backend(data),controlled(data,gate));
        std::atomic<bool> stopped=false;
        std::jthread closer;
        ReleaseGate release{gate}; // Releases before closer joins if any assertion fails.
        std::vector<std::future<fs::Completion>> accepted;
        for(int i=0;i<2;++i) accepted.push_back(runtime.submit(query(),fs::Route::GpuBatch));
        gate->await_entries(1);
        for(int i=0;i<2;++i) accepted.push_back(runtime.submit(query(),fs::Route::GpuBatch));
        closer=std::jthread([&] { runtime.shutdown(); stopped=true; });
        bool rejected=false;
        const auto deadline=std::chrono::steady_clock::now()+3s;
        while(!rejected && std::chrono::steady_clock::now()<deadline) {
            try { accepted.push_back(runtime.submit(query())); }
            catch(const fs::RuntimeStopped&) { rejected=true; }
            catch(const fs::QueueFull&) {}
            const auto snapshot=runtime.snapshot();
            CHECK(snapshot.cpu_queue_depth+snapshot.gpu_queue_depth<=config.queue_capacity);
            std::this_thread::yield();
        }
        CHECK(rejected && !stopped); // Shutdown waits for the GPU execution already in flight.
        gate->release(); closer.join();
        for(auto& future:accepted) validate(future.get(),baseline);
        const auto stats=runtime.snapshot();
        CHECK(stats.completed==stats.submitted && stats.failed==0 && stats.gpu_inflight==0);
    }
    std::future<fs::Completion> survivor;
    {
        config.max_wait=5s;
        auto gate=std::make_shared<Gate>();
        fs::Runtime runtime(data,config,fs::make_scalar_backend(data),controlled(data,gate));
        survivor=runtime.submit(query(),fs::Route::GpuBatch);
    }
    validate(survivor.get(),baseline); // Result and backend name outlive the runtime.
}

void gpu_tests() {
    const auto data=fs::Dataset::generate(1000,129);
    const auto baseline=fs::make_scalar_backend(data)->search(query(129));
    fs::RuntimeConfig config; config.enable_cuda=true; config.max_batch_size=8;
    fs::Runtime runtime(data,config);
    std::vector<std::future<fs::Completion>> futures;
    for(int i=0;i<128;++i) {
        const auto route=i%4==0?fs::Route::Cpu:(i%4==1?fs::Route::GpuImmediate:fs::Route::GpuBatch);
        futures.push_back(runtime.submit(query(129),route));
    }
    for(auto& future:futures) {
        const auto result=future.get(); validate(result,baseline);
        CHECK(result.batch_size>=1 && result.batch_size<=8);
        if(result.route!=fs::Route::Cpu) CHECK(result.device_timings.has_value() && result.backend=="cuda");
    }
    auto timeout=runtime.submit(query(129),fs::Route::GpuBatch).get();
    CHECK(timeout.batch_size==1 && timeout.queue_wait_ms>=1.99);
    runtime.shutdown();
    const auto stats=runtime.snapshot();
    CHECK(stats.completed==129 && stats.failed==0 && stats.cpu_inflight==0 && stats.gpu_inflight==0);
}
}

int main(int argc,char** argv) {
    try {
        if(argc==2 && std::string_view(argv[1])=="gpu") {
            if(!fs::cuda_available()) { std::cout<<"SKIP: "<<fs::cuda_status()<<'\n'; return 77; }
            gpu_tests();
        } else runtime_tests();
        std::cout<<checks<<" runtime checks passed\n";
        return 0;
    } catch(const std::exception& error) { std::cerr<<"FAIL: "<<error.what()<<'\n'; return 1; }
}
