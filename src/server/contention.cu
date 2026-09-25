#include "contention.hpp"
#include <cuda_runtime.h>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace flowstate {
namespace {
void check(cudaError_t code,const char* operation) {
    if(code!=cudaSuccess) throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(code));
}
struct Resources {
    cudaStream_t stream=nullptr;
    float* buffer=nullptr;
    ~Resources() {
        cudaSetDevice(0);
        if(stream) cudaStreamSynchronize(stream);
        if(buffer) cudaFree(buffer);
        if(stream) cudaStreamDestroy(stream);
    }
};
__global__ void compete(float* output,unsigned long long cycles) {
    const auto start=clock64();
    float value=static_cast<float>(threadIdx.x+1)*.001f;
    while(clock64()-start<cycles) value=fmaf(value,.99999f,.00001f);
    output[blockIdx.x*blockDim.x+threadIdx.x]=value;
}
class CudaContention final : public GpuContention {
public:
    CudaContention() {
        check(cudaSetDevice(0),"Contention device");
        int multiprocessors=0,clock_khz=0;
        check(cudaDeviceGetAttribute(&multiprocessors,cudaDevAttrMultiProcessorCount,0),"Contention SM count");
        check(cudaDeviceGetAttribute(&clock_khz,cudaDevAttrClockRate,0),"Contention clock");
        blocks_=multiprocessors*6;
        cycles_=static_cast<unsigned long long>(clock_khz)*5; // About 5 ms per block, clock-dependent.
        check(cudaStreamCreateWithFlags(&resources_.stream,cudaStreamNonBlocking),"Contention stream");
        check(cudaMalloc(reinterpret_cast<void**>(&resources_.buffer),blocks_*256*sizeof(float)),"Contention buffer");
        thread_=std::thread(&CudaContention::run,this);
    }
    ~CudaContention() override {
        { std::lock_guard lock(mutex_); stopping_=true; state_.enabled=false; }
        wake_.notify_all();
        if(thread_.joinable()) thread_.join();
    }
    void set_enabled(bool enabled) override {
        { std::lock_guard lock(mutex_);
          if(enabled && !state_.error.empty()) throw std::runtime_error(state_.error);
          state_.enabled=enabled; }
        wake_.notify_all();
    }
    ContentionState snapshot() const override { std::lock_guard lock(mutex_); return state_; }
private:
    void run() {
        try {
            check(cudaSetDevice(0),"Contention worker device");
            for(;;) {
                {
                    std::unique_lock lock(mutex_);
                    wake_.wait(lock,[&] { return stopping_ || state_.enabled; });
                    if(stopping_) return;
                    state_.active=true;
                }
                compete<<<blocks_,256,0,resources_.stream>>>(resources_.buffer,cycles_);
                check(cudaGetLastError(),"Contention launch");
                check(cudaStreamSynchronize(resources_.stream),"Contention completion");
                std::lock_guard lock(mutex_);
                ++state_.kernels; state_.active=state_.enabled && !stopping_;
            }
        } catch(const std::exception& error) {
            std::lock_guard lock(mutex_);
            state_.error=error.what(); state_.active=false; state_.enabled=false;
        }
    }
    Resources resources_;
    int blocks_=0;
    unsigned long long cycles_=0;
    mutable std::mutex mutex_;
    std::condition_variable wake_;
    bool stopping_=false;
    ContentionState state_;
    std::thread thread_;
};
}
std::unique_ptr<GpuContention> make_gpu_contention() { return std::make_unique<CudaContention>(); }
}
