#include "search_detail.hpp"
#include <cuda_runtime.h>

#include <utility>

namespace flowstate {
namespace {
void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

class DeviceBuffer {
public:
    ~DeviceBuffer() { if (data_) cudaFree(data_); }
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void reserve(std::size_t count) {
        if (count <= capacity_) return;
        DeviceBuffer replacement;
        check(cudaMalloc(reinterpret_cast<void**>(&replacement.data_), checked_product(count, sizeof(float))), "cudaMalloc");
        replacement.capacity_ = count;
        std::swap(data_, replacement.data_);
        std::swap(capacity_, replacement.capacity_);
    }
    float* data() const { return data_; }
private:
    float* data_ = nullptr;
    std::size_t capacity_ = 0;
};

class Stream {
public:
    Stream() = default;
    ~Stream() { if (handle) cudaStreamDestroy(handle); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    void create() { check(cudaStreamCreateWithFlags(&handle, cudaStreamNonBlocking), "cudaStreamCreate"); }
    cudaStream_t handle = nullptr;
};

class Event {
public:
    Event() = default;
    ~Event() { if (handle) cudaEventDestroy(handle); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    void create() { check(cudaEventCreate(&handle), "cudaEventCreate"); }
    void record(cudaStream_t stream) { check(cudaEventRecord(handle, stream), "cudaEventRecord"); }
    cudaEvent_t handle = nullptr;
};

double elapsed(const Event& start, const Event& end) {
    float ms = 0;
    check(cudaEventElapsedTime(&ms, start.handle, end.handle), "cudaEventElapsedTime");
    return ms;
}

constexpr unsigned threads = 256;
__global__ void dot_products(const float* dataset, const float* queries, float* scores,
                             std::size_t count, std::size_t dimension) {
    const std::size_t row = blockIdx.x;
    const std::size_t query = blockIdx.y;
    const unsigned lane = threadIdx.x;
    __shared__ float partial[threads];
    float sum = 0;
    for (std::size_t d = lane; d < dimension; d += threads)
        sum += dataset[row * dimension + d] * queries[query * dimension + d];
    partial[lane] = sum;
    __syncthreads();
    for (unsigned stride = threads / 2; stride > 0; stride /= 2) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    if (lane == 0) scores[query * count + row] = partial[0];
}

class CudaBackend final : public Backend {
public:
    explicit CudaBackend(std::shared_ptr<const Dataset> dataset) : dataset_(std::move(dataset)) {
        if (!dataset_) throw std::invalid_argument("Dataset must not be null");
        check(cudaSetDevice(0), "cudaSetDevice");
        check(cudaGetDeviceProperties(&properties_, 0), "cudaGetDeviceProperties");
        if (dataset_->size() > static_cast<std::size_t>(properties_.maxGridSize[0]))
            throw std::length_error("Dataset exceeds CUDA grid size");
        stream_.create();
        upload_start_.create(); upload_end_.create(); kernel_end_.create(); download_end_.create();
        device_dataset_.reserve(dataset_->values().size());
        check(cudaMemcpy(device_dataset_.data(), dataset_->values().data(),
                         checked_product(dataset_->values().size(), sizeof(float)), cudaMemcpyHostToDevice),
              "Dataset upload");
    }
    ~CudaBackend() override {
        // Destructors cannot throw. All request operations report CUDA errors synchronously.
        cudaSetDevice(0);
        if (stream_.handle) cudaStreamSynchronize(stream_.handle);
    }
    std::string_view name() const override { return "cuda"; }
    SearchResult search(const Query& query) override {
        auto batch = search_batch(std::span<const Query>(&query, 1));
        return std::move(batch.results.front());
    }
    BatchResult search_batch(std::span<const Query> queries) override {
        if (queries.empty()) return {};
        if (queries.size() > static_cast<std::size_t>(properties_.maxGridSize[1]))
            throw std::length_error("Batch exceeds CUDA grid size");
        for (const auto& query : queries) validate_query(*dataset_, query);
        check(cudaSetDevice(0), "cudaSetDevice");
        const auto query_count = checked_product(queries.size(), dataset_->dimension());
        const auto score_count = checked_product(queries.size(), dataset_->size());
        const auto query_bytes = checked_product(query_count, sizeof(float));
        const auto score_bytes = checked_product(score_count, sizeof(float));
        host_queries_.resize(query_count);
        host_scores_.resize(score_count);
        device_queries_.reserve(query_count);
        device_scores_.reserve(score_count);
        for (std::size_t i = 0; i < queries.size(); ++i)
            std::copy(queries[i].values.begin(), queries[i].values.end(), host_queries_.data() + i * dataset_->dimension());
        try {
            upload_start_.record(stream_.handle);
            check(cudaMemcpyAsync(device_queries_.data(), host_queries_.data(), query_bytes,
                                  cudaMemcpyHostToDevice, stream_.handle), "Query upload");
            upload_end_.record(stream_.handle);
            const dim3 grid(static_cast<unsigned>(dataset_->size()), static_cast<unsigned>(queries.size()));
            dot_products<<<grid, threads, 0, stream_.handle>>>(device_dataset_.data(), device_queries_.data(),
                                                              device_scores_.data(), dataset_->size(), dataset_->dimension());
            check(cudaGetLastError(), "Dot-product kernel launch");
            kernel_end_.record(stream_.handle);
            check(cudaMemcpyAsync(host_scores_.data(), device_scores_.data(), score_bytes,
                                  cudaMemcpyDeviceToHost, stream_.handle), "Score download");
            download_end_.record(stream_.handle);
            check(cudaStreamSynchronize(stream_.handle), "Search completion");
        } catch (...) {
            // No pending DMA may outlive or race the reusable host buffers after an error.
            cudaStreamSynchronize(stream_.handle);
            throw;
        }
        BatchResult result;
        result.device_timings = DeviceTimings{elapsed(upload_start_, upload_end_), elapsed(upload_end_, kernel_end_),
                                              elapsed(kernel_end_, download_end_)};
        result.results.reserve(queries.size());
        for (std::size_t i = 0; i < queries.size(); ++i) {
            detail::TopK top(queries[i].top_k);
            for (std::size_t id = 0; id < dataset_->size(); ++id)
                top.add(id, host_scores_[i * dataset_->size() + id]);
            result.results.push_back(top.finish());
        }
        return result;
    }
private:
    std::shared_ptr<const Dataset> dataset_;
    cudaDeviceProp properties_{};
    Stream stream_;
    Event upload_start_, upload_end_, kernel_end_, download_end_;
    DeviceBuffer device_dataset_, device_queries_, device_scores_;
    std::vector<float> host_queries_, host_scores_;
};
} // namespace

bool cuda_available() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return false;
    cudaDeviceProp properties{};
    return cudaGetDeviceProperties(&properties, 0) == cudaSuccess && properties.major > 0;
}

std::string cuda_status() {
    int count = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) return std::string("CUDA unavailable: ") + cudaGetErrorString(status);
    if (count == 0) return "CUDA unavailable: no NVIDIA devices";
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
    int runtime = 0, driver = 0;
    check(cudaRuntimeGetVersion(&runtime), "cudaRuntimeGetVersion");
    check(cudaDriverGetVersion(&driver), "cudaDriverGetVersion");
    return std::string(properties.name) + "; runtime=" + std::to_string(runtime) + "; driver=" + std::to_string(driver);
}

std::unique_ptr<Backend> make_cuda_backend(std::shared_ptr<const Dataset> dataset) {
    if (!dataset) throw std::invalid_argument("Dataset must not be null");
    if (!cuda_available()) throw std::runtime_error(cuda_status());
    return std::make_unique<CudaBackend>(std::move(dataset));
}
} // namespace flowstate
