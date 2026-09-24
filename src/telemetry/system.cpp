#include "flowstate/telemetry.hpp"
#include <fstream>
#include <string>
#ifdef FLOWSTATE_HAVE_NVML
#include <nvml.h>
#include <cuda_runtime_api.h>
#endif

namespace flowstate {
struct SystemSampler::State {
    std::optional<std::uint64_t> previous_total, previous_idle;
#ifdef FLOWSTATE_HAVE_NVML
    bool initialized = false;
    nvmlDevice_t device = nullptr;
    ~State() { if(initialized) nvmlShutdown(); }
#endif
};
SystemSampler::SystemSampler(bool gpu_enabled) : state_(std::make_unique<State>()) {
#ifdef FLOWSTATE_HAVE_NVML
    if(gpu_enabled && nvmlInit_v2()==NVML_SUCCESS) {
        state_->initialized=true;
        // Flowstate uses one GPU, CUDA device 0. Resolve its PCI ID rather than
        // assuming the CUDA and NVML enumeration orders match.
        char pci_id[32]{};
        if(cudaDeviceGetPCIBusId(pci_id,sizeof(pci_id),0)==cudaSuccess)
            nvmlDeviceGetHandleByPciBusId_v2(pci_id,&state_->device);
    }
#else
    (void)gpu_enabled;
#endif
}
SystemSampler::~SystemSampler() = default;
SystemStats SystemSampler::sample() {
    SystemStats result;
#ifdef __linux__
    std::ifstream input("/proc/stat");
    std::string name;
    std::uint64_t user=0,nice=0,system=0,idle=0,iowait=0,irq=0,softirq=0,steal=0;
    if(input>>name>>user>>nice>>system>>idle>>iowait>>irq>>softirq>>steal && name=="cpu") {
        const auto total=user+nice+system+idle+iowait+irq+softirq+steal;
        const auto idle_total=idle+iowait;
        if(state_->previous_total && total>*state_->previous_total && idle_total>=*state_->previous_idle) {
            const auto elapsed=total-*state_->previous_total, idle_elapsed=idle_total-*state_->previous_idle;
            if(idle_elapsed<=elapsed) result.cpu_utilization=1.0-static_cast<double>(idle_elapsed)/elapsed;
        }
        state_->previous_total=total; state_->previous_idle=idle_total;
    } else { state_->previous_total.reset(); state_->previous_idle.reset(); }
#endif
#ifdef FLOWSTATE_HAVE_NVML
    if(state_->device) {
        nvmlUtilization_t utilization{};
        if(nvmlDeviceGetUtilizationRates(state_->device,&utilization)==NVML_SUCCESS && utilization.gpu<=100)
            result.gpu_utilization=utilization.gpu/100.0;
        nvmlMemory_t memory{};
        if(nvmlDeviceGetMemoryInfo(state_->device,&memory)==NVML_SUCCESS) result.gpu_memory_free_bytes=memory.free;
    }
#endif
    return result;
}
} // namespace flowstate
