#include "flowstate/telemetry.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace flowstate {
namespace {
constexpr std::size_t bins = 256, buckets = 300;
using Histogram = std::array<std::uint64_t, bins>;
std::size_t bin(double ms) {
    if (!std::isfinite(ms) || ms < 0) throw std::invalid_argument("Invalid latency sample");
    if (ms <= .001) return 0;
    return std::min(bins - 1, static_cast<std::size_t>(std::ceil(std::log(ms / .001) / std::log(1.1))));
}
LatencyStats summarize(const Histogram& histogram) {
    LatencyStats result;
    for (auto count : histogram) result.count += count;
    if (!result.count) return result;
    auto percentile = [&](double fraction) {
        const auto rank = static_cast<std::uint64_t>(std::ceil(fraction * result.count));
        std::uint64_t cumulative = 0;
        for (std::size_t i = 0; i < bins; ++i) {
            cumulative += histogram[i];
            if (cumulative >= rank) return i == bins - 1 ? std::numeric_limits<double>::infinity() : .001 * std::pow(1.1, i);
        }
        return 0.0;
    };
    result.p50_ms=percentile(.5); result.p95_ms=percentile(.95); result.p99_ms=percentile(.99);
    return result;
}
}
struct RollingTelemetry::Storage {
    struct Bucket {
        std::int64_t tick = -1;
        std::uint64_t arrivals = 0, completed = 0, failed = 0, gpu_batches = 0, gpu_jobs = 0;
        std::array<Histogram, 4> latency{};
    };
    Clock::time_point start;
    std::array<Bucket, buckets> data{};
    std::int64_t tick(Clock::time_point now) const {
        return std::max<std::int64_t>(0, std::chrono::duration_cast<std::chrono::milliseconds>(now-start).count()/100);
    }
    Bucket& at(Clock::time_point now) {
        const auto index=tick(now);
        auto& bucket=data[static_cast<std::size_t>(index)%buckets];
        if (bucket.tick!=index) { bucket=Bucket{}; bucket.tick=index; }
        return bucket;
    }
};
RollingTelemetry::RollingTelemetry(Clock::time_point start) : storage_(std::make_unique<Storage>()) { storage_->start=start; }
RollingTelemetry::~RollingTelemetry() = default;
void RollingTelemetry::arrival(Clock::time_point now) { ++storage_->at(now).arrivals; }
void RollingTelemetry::completion(Clock::time_point now, MetricBackend backend, double ms, bool failed) {
    auto& bucket=storage_->at(now);
    ++bucket.completed;
    if (failed) ++bucket.failed;
    else ++bucket.latency.at(static_cast<std::size_t>(backend))[bin(ms)];
}
void RollingTelemetry::gpu_batch(Clock::time_point now, std::size_t size) {
    auto& bucket=storage_->at(now); ++bucket.gpu_batches; bucket.gpu_jobs+=size;
}
std::array<WindowStats,3> RollingTelemetry::snapshot(Clock::time_point now) const {
    std::array<WindowStats,3> result;
    constexpr std::array<int,3> lengths{10,50,300};
    const auto current=storage_->tick(now);
    for (std::size_t w=0; w<lengths.size(); ++w) {
        auto& window=result[w];
        window.seconds=lengths[w]/10.0; // Fixed rolling denominator, including an initially empty history.
        std::array<Histogram,4> histograms{};
        std::uint64_t batch_count=0, batch_jobs=0;
        for (const auto& bucket:storage_->data) {
            if (bucket.tick<0 || bucket.tick>current || bucket.tick<=current-lengths[w]) continue;
            window.arrivals+=bucket.arrivals; window.completed+=bucket.completed; window.failed+=bucket.failed;
            batch_count+=bucket.gpu_batches; batch_jobs+=bucket.gpu_jobs;
            for(std::size_t b=0;b<4;++b) for(std::size_t i=0;i<bins;++i) histograms[b][i]+=bucket.latency[b][i];
        }
        Histogram combined{};
        for(std::size_t b=0;b<4;++b) {
            window.backends[b]=summarize(histograms[b]);
            for(std::size_t i=0;i<bins;++i) combined[i]+=histograms[b][i];
        }
        window.latency=summarize(combined);
        if(window.seconds>0) {
            window.arrival_rate=window.arrivals/window.seconds;
            window.throughput=(window.completed-window.failed)/window.seconds;
        }
        if(batch_count) window.mean_gpu_batch_size=static_cast<double>(batch_jobs)/batch_count;
    }
    return result;
}
} // namespace flowstate
