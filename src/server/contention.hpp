#pragma once
#include <cstdint>
#include <memory>
#include <string>

namespace flowstate {
struct ContentionState {
    bool enabled = false, active = false;
    std::uint64_t kernels = 0;
    std::string error;
};
// Demo-only competing work. It owns its stream/buffer/thread and never changes search results.
class GpuContention {
public:
    virtual ~GpuContention() = default;
    virtual void set_enabled(bool enabled) = 0;
    virtual ContentionState snapshot() const = 0;
};
std::unique_ptr<GpuContention> make_gpu_contention();
}
