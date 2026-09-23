#pragma once

#include "flowstate/query.hpp"
#include <optional>
#include <string>
#include <string_view>

namespace flowstate {

struct DeviceTimings {
    double upload_ms = 0;
    double kernel_ms = 0;
    double download_ms = 0;
};

struct BatchResult {
    std::vector<SearchResult> results;
    std::optional<DeviceTimings> device_timings;
};

// Dataset ownership is shared and immutable. CPU backends are reentrant;
// CUDA instances own reusable buffers and must be called by one worker at a time.
class Backend {
public:
    virtual ~Backend() = default;
    virtual SearchResult search(const Query& query) = 0;
    virtual BatchResult search_batch(std::span<const Query> queries);
    [[nodiscard]] virtual std::string_view name() const = 0;
};

std::unique_ptr<Backend> make_scalar_backend(std::shared_ptr<const Dataset> dataset);
std::unique_ptr<Backend> make_avx2_backend(std::shared_ptr<const Dataset> dataset);
std::unique_ptr<Backend> make_cuda_backend(std::shared_ptr<const Dataset> dataset);
bool avx2_available();
bool cuda_available();
// Human-readable driver/device availability, including the reason CUDA is unavailable.
std::string cuda_status();

} // namespace flowstate
