#include "../src/cli.hpp"
#include "flowstate/runtime.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>

namespace {
double percentile(const std::vector<double>& sorted, double p) {
    return sorted[static_cast<std::size_t>(std::ceil(p * sorted.size())) - 1];
}
}

int main(int argc, char** argv) {
    try {
        const auto options = flowstate::cli::parse(argc, argv, true, true);
        if (options.help) {
            flowstate::cli::help(true);
            std::cout << "Runtime: --workers N --requests N --queue-capacity N --max-wait-us N\n"
                         "--backend all runs CPU, GPU immediate, and GPU batch; cuda runs both GPU modes.\n";
            return 0;
        }
#ifndef NDEBUG
        throw std::runtime_error("Benchmark in Release mode");
#endif
        if (options.require_all && (!flowstate::avx2_available() || !flowstate::cuda_available()))
            throw std::runtime_error("--require-all needs AVX2 and CUDA");
        const auto dataset = flowstate::Dataset::generate(options.vectors, options.dimension, options.seed);
        const auto queries = flowstate::cli::queries(options);
        std::vector<flowstate::Route> routes;
        if (options.backend != "cuda") routes.push_back(flowstate::Route::Cpu);
        if (options.backend == "cuda" || (options.backend == "all" && flowstate::cuda_available())) {
            routes.push_back(flowstate::Route::GpuImmediate);
            routes.push_back(flowstate::Route::GpuBatch);
        }
        if (options.backend == "all" && !flowstate::cuda_available()) std::cerr << "Skipping GPU: " << flowstate::cuda_status() << '\n';
        std::cout << "mode,vectors,dimension,top_k,seed,workers,batch_max,max_wait_us,queue_capacity,requests,iterations,warmup,"
                     "queries_per_second,p50_ms,p95_ms,p99_ms,mean_queue_wait_ms,gpu_batches,mean_gpu_batch_size,"
                     "size_flushes,timeout_flushes,mean_kernel_ms,checksum\n";
        for (const auto route : routes) {
            flowstate::RuntimeConfig config;
            config.cpu_workers = options.workers;
            config.queue_capacity = options.queue_capacity;
            config.max_batch_size = options.batch_size;
            config.max_wait = std::chrono::microseconds(options.max_wait_us);
            auto cpu = flowstate::cli::backend(options.backend == "all" || options.backend == "cuda" ? "auto" : options.backend, dataset);
            auto gpu = route == flowstate::Route::Cpu ? nullptr : flowstate::make_cuda_backend(dataset);
            flowstate::Runtime runtime(dataset, config, std::move(cpu), std::move(gpu));
            const auto name = route == flowstate::Route::Cpu ? "cpu" : route == flowstate::Route::GpuImmediate ? "gpu_immediate" : "gpu_batch";
            std::vector<std::future<flowstate::Completion>> futures;
            futures.reserve(options.requests);
            std::vector<double> latencies;
            latencies.reserve(flowstate::checked_product(options.requests, options.iterations));
            std::uint64_t checksum = 0;
            auto round = [&](bool measure) {
                futures.clear();
                for (std::size_t i = 0; i < options.requests; ++i)
                    futures.push_back(runtime.submit(queries[i % queries.size()], route));
                for (auto& future : futures) {
                    auto result = future.get();
                    if (measure) {
                        latencies.push_back(result.latency_ms);
                        for (const auto id : result.result.ids) checksum += id;
                    }
                }
            };
            for (std::size_t i = 0; i < options.warmup; ++i) round(false);
            const auto before = runtime.snapshot();
            const auto start = std::chrono::steady_clock::now();
            for (std::size_t i = 0; i < options.iterations; ++i) round(true);
            const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            runtime.shutdown();
            const auto after = runtime.snapshot();
            const auto count = latencies.size();
            const auto batches = after.gpu_batches - before.gpu_batches;
            const auto timed_batches = after.timed_gpu_batches - before.timed_gpu_batches;
            std::sort(latencies.begin(), latencies.end());
            std::cout << std::fixed << std::setprecision(6) << name << ',' << options.vectors << ',' << options.dimension << ','
                      << options.top_k << ',' << options.seed << ',' << options.workers << ',' << options.batch_size << ','
                      << options.max_wait_us << ',' << options.queue_capacity << ',' << options.requests << ','
                      << options.iterations << ',' << options.warmup << ',' << count / seconds << ','
                      << percentile(latencies, .5) << ',' << percentile(latencies, .95) << ',' << percentile(latencies, .99) << ','
                      << (after.total_queue_wait_ms - before.total_queue_wait_ms) / count << ',' << batches << ','
                      << (batches ? static_cast<double>(after.gpu_completed - before.gpu_completed) / batches : 0) << ','
                      << after.size_flushes - before.size_flushes << ',' << after.timeout_flushes - before.timeout_flushes << ',';
            if (timed_batches) std::cout << (after.total_device_timings.kernel_ms - before.total_device_timings.kernel_ms) / timed_batches;
            std::cout << ',' << checksum << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "flowstate_runtime_bench: " << error.what() << '\n';
        return 1;
    }
}
