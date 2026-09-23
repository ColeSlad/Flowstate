#include "search_detail.hpp"

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define FLOWSTATE_X86 1
#endif

namespace flowstate {

bool avx2_available() {
#ifdef FLOWSTATE_X86
    // GCC/Clang include OS support for preserving the extended register state.
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}

namespace {
#ifdef FLOWSTATE_X86
// Only this function may emit AVX2 instructions. The executable baseline remains portable.
__attribute__((target("avx2")))
float dot_avx2(std::span<const float> a, std::span<const float> b) {
    __m256 sum = _mm256_setzero_ps();
    std::size_t d = 0;
    for (; a.size() - d >= 8; d += 8)
        sum = _mm256_add_ps(sum, _mm256_mul_ps(_mm256_loadu_ps(a.data() + d),
                                             _mm256_loadu_ps(b.data() + d)));
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, sum);
    float result = 0;
    for (const float lane : lanes) result += lane;
    for (; d < a.size(); ++d) result += a[d] * b[d];
    return result;
}

class Avx2Backend final : public Backend {
public:
    explicit Avx2Backend(std::shared_ptr<const Dataset> dataset) : dataset_(std::move(dataset)) {
        if (!dataset_) throw std::invalid_argument("Dataset must not be null");
    }
    SearchResult search(const Query& query) override {
        validate_query(*dataset_, query);
        detail::TopK top(query.top_k);
        for (std::size_t id = 0; id < dataset_->size(); ++id)
            top.add(id, dot_avx2(dataset_->row(id), query.values));
        return top.finish();
    }
    std::string_view name() const override { return "avx2"; }
private:
    std::shared_ptr<const Dataset> dataset_;
};
#endif
} // namespace

std::unique_ptr<Backend> make_avx2_backend(std::shared_ptr<const Dataset> dataset) {
    if (!avx2_available()) throw std::runtime_error("AVX2 is unavailable on this CPU/OS");
#ifdef FLOWSTATE_X86
    return std::make_unique<Avx2Backend>(std::move(dataset));
#else
    (void)dataset;
    throw std::runtime_error("This build does not target x86");
#endif
}
} // namespace flowstate
