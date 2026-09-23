#include "flowstate/backend.hpp"
#include <stdexcept>

namespace flowstate {
bool cuda_available() { return false; }
std::string cuda_status() { return "CUDA disabled at build time"; }
std::unique_ptr<Backend> make_cuda_backend(std::shared_ptr<const Dataset>) {
    throw std::runtime_error(cuda_status());
}
} // namespace flowstate
