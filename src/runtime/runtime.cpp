#include "flowstate/runtime.hpp"
#include <algorithm>

namespace flowstate {
namespace {
double milliseconds(std::chrono::steady_clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}
std::unique_ptr<Backend> cpu_backend(std::shared_ptr<const Dataset> dataset) {
    return avx2_available() ? make_avx2_backend(std::move(dataset)) : make_scalar_backend(std::move(dataset));
}
}

Runtime::Runtime(std::shared_ptr<const Dataset> dataset, RuntimeConfig config)
    : Runtime(dataset, config, cpu_backend(dataset), config.enable_cuda ? make_cuda_backend(dataset) : nullptr) {}

Runtime::Runtime(std::shared_ptr<const Dataset> dataset, RuntimeConfig config,
                 std::unique_ptr<Backend> cpu, std::unique_ptr<Backend> gpu)
    : dataset_(std::move(dataset)), config_(config), cpu_(std::move(cpu)), gpu_(std::move(gpu)) {
    if (!dataset_ || !cpu_) throw std::invalid_argument("Runtime requires a dataset and CPU backend");
    if (!config_.cpu_workers || !config_.queue_capacity || !config_.max_batch_size || config_.max_wait.count() < 0 ||
        config_.max_wait > std::chrono::seconds(60))
        throw std::invalid_argument("Invalid runtime worker, queue, or batch configuration");
    cpu_name_ = cpu_->name();
    if (gpu_) {
        gpu_name_ = gpu_->name();
        gpu_jobs_.reserve(config_.max_batch_size);
        gpu_queries_.reserve(config_.max_batch_size);
    }
    cpu_threads_.reserve(config_.cpu_workers);
    try {
        for (std::size_t i = 0; i < config_.cpu_workers; ++i)
            cpu_threads_.emplace_back(&Runtime::cpu_loop, this);
        if (gpu_) gpu_thread_ = std::thread(&Runtime::gpu_loop, this);
    } catch (...) {
        shutdown(); // Also joins partially constructed pools.
        throw;
    }
}

Runtime::~Runtime() { shutdown(); }

std::future<Completion> Runtime::submit(Query query, Route route) {
    const auto submitted = Clock::now();
    validate_query(*dataset_, query);
    if (route != Route::Cpu && route != Route::GpuImmediate && route != Route::GpuBatch)
        throw std::invalid_argument("Unknown runtime route");
    if (route != Route::Cpu && !gpu_) throw std::invalid_argument("Runtime has no GPU backend");
    std::unique_lock lock(mutex_);
    if (stopping_) { ++stats_.rejected; throw RuntimeStopped(); }
    const auto queued = cpu_queue_.size() + gpu_queue_.size();
    if (queued >= config_.queue_capacity) { ++stats_.rejected; throw QueueFull(); }
    Job job{std::move(query), route, next_id_, submitted, {}};
    auto future = job.promise.get_future();
    auto& queue = route == Route::Cpu ? cpu_queue_ : gpu_queue_;
    queue.push_back(std::move(job));
    ++next_id_;
    ++stats_.submitted;
    stats_.max_queue_depth = std::max(stats_.max_queue_depth, queued + 1);
    lock.unlock();
    if (route == Route::Cpu) cpu_ready_.notify_one();
    else gpu_ready_.notify_one();
    return future;
}

void Runtime::shutdown() {
    std::lock_guard shutdown_lock(shutdown_mutex_);
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
    }
    cpu_ready_.notify_all();
    gpu_ready_.notify_all();
    for (auto& thread : cpu_threads_) if (thread.joinable()) thread.join();
    if (gpu_thread_.joinable()) gpu_thread_.join();
}

RuntimeSnapshot Runtime::snapshot() const {
    std::lock_guard lock(mutex_);
    auto result = stats_;
    result.cpu_queue_depth = cpu_queue_.size();
    result.gpu_queue_depth = gpu_queue_.size();
    return result;
}

void Runtime::complete(Job& job, SearchResult result, Clock::time_point started, Clock::time_point ended,
                       std::size_t batch_size, std::optional<DeviceTimings> timings, std::exception_ptr error) {
    std::optional<Completion> completion;
    if (!error) {
        try {
            completion.emplace(Completion{job.id, std::move(result), job.route == Route::Cpu ? cpu_name_ : gpu_name_,
                job.route, batch_size, milliseconds(started - job.submitted), milliseconds(ended - started),
                milliseconds(ended - job.submitted), timings});
        } catch (...) { error = std::current_exception(); }
    }
    {
        std::lock_guard lock(mutex_);
        ++stats_.completed;
        stats_.failed += error != nullptr;
        if (job.route == Route::Cpu) { ++stats_.cpu_completed; --stats_.cpu_inflight; }
        else { ++stats_.gpu_completed; --stats_.gpu_inflight; }
        stats_.total_queue_wait_ms += milliseconds(started - job.submitted);
        // Include completion construction, earlier batch publications, and mutex wait.
        // Keep the separate backend endpoint for execution_ms.
        const auto latency_ms = milliseconds(Clock::now() - job.submitted);
        stats_.total_latency_ms += latency_ms;
        if (completion) completion->latency_ms = latency_ms;
    }
    // Fulfill exactly once, outside the backend-error catch blocks and queue lock.
    if (error) job.promise.set_exception(error);
    else job.promise.set_value(std::move(*completion));
}

void Runtime::cpu_loop() {
    for (;;) {
        std::unique_lock lock(mutex_);
        cpu_ready_.wait(lock, [&] { return stopping_ || !cpu_queue_.empty(); });
        if (cpu_queue_.empty()) return;
        auto job = std::move(cpu_queue_.front());
        cpu_queue_.pop_front();
        ++stats_.cpu_inflight;
        lock.unlock();
        const auto started = Clock::now();
        SearchResult result;
        std::exception_ptr error;
        try { result = cpu_->search(job.query); }
        catch (...) { error = std::current_exception(); }
        complete(job, std::move(result), started, Clock::now(), 1, std::nullopt, error);
    }
}

void Runtime::gpu_loop() {
    for (;;) {
        gpu_jobs_.clear();
        gpu_queries_.clear();
        std::unique_lock lock(mutex_);
        gpu_ready_.wait(lock, [&] { return stopping_ || !gpu_queue_.empty(); });
        if (gpu_queue_.empty()) return;
        std::size_t count = 1;
        if (gpu_queue_.front().route == Route::GpuBatch) {
            const auto deadline = gpu_queue_.front().submitted + config_.max_wait;
            for (;;) {
                count = 0;
                while (count < gpu_queue_.size() && count < config_.max_batch_size &&
                       gpu_queue_[count].route == Route::GpuBatch) ++count;
                if (count == config_.max_batch_size) { ++stats_.size_flushes; break; }
                if (stopping_ || count < gpu_queue_.size()) break; // Drain, or an immediate request follows.
                if (Clock::now() >= deadline) { ++stats_.timeout_flushes; break; }
                gpu_ready_.wait_until(lock, deadline);
            }
        }
        for (std::size_t i = 0; i < count; ++i) {
            gpu_jobs_.push_back(std::move(gpu_queue_.front()));
            gpu_queue_.pop_front();
        }
        stats_.gpu_inflight += count;
        ++stats_.gpu_batches;
        lock.unlock();
        const auto started = Clock::now();
        for (auto& job : gpu_jobs_) gpu_queries_.push_back(std::move(job.query));
        BatchResult batch;
        std::exception_ptr error;
        try {
            batch = gpu_->search_batch(gpu_queries_);
            if (batch.results.size() != count) throw std::runtime_error("GPU returned the wrong batch result count");
        } catch (...) { error = std::current_exception(); }
        const auto ended = Clock::now();
        {
            std::lock_guard stats_lock(mutex_);
            stats_.last_device_timings = error ? std::nullopt : batch.device_timings;
            if (!error && batch.device_timings) {
                ++stats_.timed_gpu_batches;
                stats_.total_device_timings.upload_ms += batch.device_timings->upload_ms;
                stats_.total_device_timings.kernel_ms += batch.device_timings->kernel_ms;
                stats_.total_device_timings.download_ms += batch.device_timings->download_ms;
            }
        }
        for (std::size_t i = 0; i < count; ++i)
            complete(gpu_jobs_[i], error ? SearchResult{} : std::move(batch.results[i]),
                     started, ended, count, batch.device_timings, error);
    }
}

} // namespace flowstate
