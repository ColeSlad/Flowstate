#include "../src/cli.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <numeric>

namespace {
using Clock = std::chrono::steady_clock;
double percentile(const std::vector<double>& sorted, double p) {
    return sorted[static_cast<std::size_t>(std::ceil(p * static_cast<double>(sorted.size()))) - 1];
}
std::string csv(std::string_view value) {
    std::string escaped = "\"";
    for (const char c : value) { escaped += c; if (c == '"') escaped += '"'; }
    return escaped + '"';
}
#if defined(__aarch64__) || defined(__arm64__)
constexpr std::string_view architecture = "arm64";
#elif defined(__x86_64__)
constexpr std::string_view architecture = "x86_64";
#else
constexpr std::string_view architecture = "other";
#endif
}

int main(int argc, char** argv) {
    try {
        const auto options = flowstate::cli::parse(argc, argv, true);
        if (options.help) { flowstate::cli::help(true); return 0; }
#ifndef NDEBUG
        throw std::runtime_error("Benchmark in Release mode; this build has assertions enabled");
#endif
        if (options.require_all && (!flowstate::avx2_available() || !flowstate::cuda_available()))
            throw std::runtime_error("--require-all: AVX2 and CUDA must both be available");
        const auto dataset = flowstate::Dataset::generate(options.vectors, options.dimension, options.seed);
        const auto queries = flowstate::cli::queries(options);
        const std::vector<std::string> names = options.backend == "all" ?
            std::vector<std::string>{"scalar", "avx2", "cuda"} : std::vector<std::string>{options.backend};
        std::cout << "backend,vectors,dimension,top_k,seed,batch_size,workers,iterations,warmup,"
                     "mean_batch_ms,p50_batch_ms,p95_batch_ms,p99_batch_ms,queries_per_second,"
                     "mean_upload_ms,mean_kernel_ms,mean_download_ms,checksum,arch,compiler,label,cuda_status\n";
        for (const auto& name : names) {
            if (options.backend == "all" && ((name == "avx2" && !flowstate::avx2_available()) ||
                                            (name == "cuda" && !flowstate::cuda_available()))) {
                std::cerr << "Skipping " << name << ": " << (name == "cuda" ? flowstate::cuda_status() : "unavailable on this CPU/OS") << '\n';
                continue;
            }
            auto backend = flowstate::cli::backend(name, dataset);
            for (std::size_t i = 0; i < options.warmup; ++i) (void)backend->search_batch(queries);
            std::vector<double> samples;
            samples.reserve(options.iterations);
            flowstate::DeviceTimings timings;
            bool has_timings = false;
            std::uint64_t checksum = 0;
            for (std::size_t i = 0; i < options.iterations; ++i) {
                const auto start = Clock::now();
                const auto batch = backend->search_batch(queries);
                const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
                samples.push_back(ms);
                for (const auto& result : batch.results)
                    for (const auto id : result.ids) checksum += static_cast<std::uint64_t>(id);
                if (batch.device_timings) {
                    has_timings = true;
                    timings.upload_ms += batch.device_timings->upload_ms;
                    timings.kernel_ms += batch.device_timings->kernel_ms;
                    timings.download_ms += batch.device_timings->download_ms;
                }
            }
            const auto n = static_cast<double>(options.iterations);
            const double total = std::accumulate(samples.begin(), samples.end(), 0.0);
            std::sort(samples.begin(), samples.end());
            std::cout << std::fixed << std::setprecision(6) << backend->name() << ',' << options.vectors << ','
                      << options.dimension << ',' << options.top_k << ',' << options.seed << ',' << options.batch_size
                      << ",1," << options.iterations << ',' << options.warmup << ',' << total / n << ','
                      << percentile(samples, 0.50) << ',' << percentile(samples, 0.95) << ',' << percentile(samples, 0.99)
                      << ',' << n * static_cast<double>(options.batch_size) * 1000.0 / total << ',';
            if (has_timings) std::cout << timings.upload_ms / n << ',' << timings.kernel_ms / n << ',' << timings.download_ms / n;
            else std::cout << ",,";
            std::cout << ',' << checksum << ',' << architecture << ',' << csv(__VERSION__) << ',' << csv(options.label)
                      << ',' << csv(flowstate::cuda_status()) << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "flowstate_bench: " << error.what() << '\n';
        return 1;
    }
}
