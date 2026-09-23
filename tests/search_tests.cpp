#include "flowstate/backend.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace fs = flowstate;
namespace {
std::size_t checks = 0;
#define CHECK(condition) do { ++checks; if (!(condition)) throw std::runtime_error( \
    std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " #condition); } while (false)

template<class Exception, class Function> void throws(Function function) {
    ++checks;
    try { function(); } catch (const Exception&) { return; }
    throw std::runtime_error("Expected exception was not thrown");
}

bool close(double a, double b) { return std::abs(a - b) <= 2e-4 + 2e-5 * std::max(std::abs(a), std::abs(b)); }

// Independent double-precision full sort, rather than the production top-k heap.
void verify(const fs::Dataset& dataset, const fs::Query& query, const fs::SearchResult& result) {
    std::vector<double> scores(dataset.size());
    std::vector<std::size_t> order(dataset.size());
    std::iota(order.begin(), order.end(), 0);
    for (std::size_t id = 0; id < dataset.size(); ++id)
        for (std::size_t d = 0; d < dataset.dimension(); ++d)
            scores[id] += static_cast<double>(dataset.row(id)[d]) * query.values[d];
    std::sort(order.begin(), order.end(), [&](auto a, auto b) {
        return scores[a] > scores[b] || (scores[a] == scores[b] && a < b);
    });
    CHECK(result.ids.size() == query.top_k);
    CHECK(result.scores.size() == query.top_k);
    auto unique = result.ids;
    std::sort(unique.begin(), unique.end());
    CHECK(std::adjacent_find(unique.begin(), unique.end()) == unique.end());
    for (std::size_t k = 0; k < query.top_k; ++k) {
        CHECK(result.ids[k] < dataset.size());
        CHECK(close(result.scores[k], scores[result.ids[k]]));
        // Different IDs/order are permitted only when their reference scores tie within tolerance.
        CHECK(result.ids[k] == order[k] || close(scores[result.ids[k]], scores[order[k]]));
        if (k) CHECK(result.scores[k - 1] >= result.scores[k]);
    }
}

using Factory = std::unique_ptr<fs::Backend> (*)(std::shared_ptr<const fs::Dataset>);

void dataset_tests() {
    const auto a = fs::Dataset::generate(13, 9, 42);
    const auto b = fs::Dataset::generate(13, 9, 42);
    const auto c = fs::Dataset::generate(13, 9, 43);
    CHECK(std::equal(a->values().begin(), a->values().end(), b->values().begin()));
    CHECK(!std::equal(a->values().begin(), a->values().end(), c->values().begin()));
    CHECK(a->values().front() == -0.250919818878173828125f); // mt19937 seed 42 golden value
    throws<std::invalid_argument>([] { fs::Dataset::generate(0, 8); });
    throws<std::invalid_argument>([] { fs::Dataset::generate(2, 0); });
    throws<std::invalid_argument>([] { fs::Dataset(2, 8, {1, 2}); });
    throws<std::length_error>([] { fs::Dataset::generate(std::numeric_limits<std::size_t>::max(), 2); });
    throws<std::invalid_argument>([] { fs::Dataset(1, 1, {std::numeric_limits<float>::quiet_NaN()}); });
    throws<std::invalid_argument>([] { fs::Dataset(1, 1, {std::numeric_limits<float>::infinity()}); });
    throws<std::out_of_range>([&] { (void)a->row(a->size()); });
    if (!fs::avx2_available()) throws<std::runtime_error>([&] { fs::make_avx2_backend(a); });
    if (!fs::cuda_available()) throws<std::runtime_error>([&] { fs::make_cuda_backend(a); });
}

void backend_tests(Factory factory) {
    throws<std::invalid_argument>([&] { factory(nullptr); });
    auto hand = std::make_shared<const fs::Dataset>(4, 3,
        std::vector<float>{1, 2, 3, -1, -2, -3, 0, 1, 0, 1, 2, 3});
    auto backend = factory(hand);
    fs::Query query{{1, 0, 1}, 3};
    const auto result = backend->search(query);
    CHECK(result.ids == std::vector<std::size_t>({0, 3, 2}));
    CHECK(result.scores == std::vector<float>({4, 4, 0}));
    throws<std::invalid_argument>([&] { backend->search({{1, 2}, 1}); });
    throws<std::invalid_argument>([&] { backend->search({{}, 1}); });
    throws<std::invalid_argument>([&] { backend->search({{1, 2, 3}, 0}); });
    throws<std::invalid_argument>([&] { backend->search({{1, 2, 3}, 5}); });
    throws<std::invalid_argument>([&] { backend->search({{1, 2, INFINITY}, 1}); });
    throws<std::invalid_argument>([&] { backend->search({{1, 2, NAN}, 1}); });
    CHECK(backend->search_batch({}).results.empty());
    const std::vector<fs::Query> invalid_batch{query, {{1}, 1}};
    throws<std::invalid_argument>([&] { backend->search_batch(invalid_batch); });
    CHECK(backend->search(query).ids == result.ids); // reusable after validation errors

    const auto huge = std::make_shared<const fs::Dataset>(1, 1,
        std::vector<float>{std::numeric_limits<float>::max()});
    throws<std::overflow_error>([&] { factory(huge)->search({{2}, 1}); });

    for (std::size_t dimension : {1, 7, 8, 9, 31, 128, 384, 768}) {
        for (std::size_t count : {1, 5, 257}) {
            const auto dataset = fs::Dataset::generate(count, dimension, 42);
            auto candidate = factory(dataset);
            auto scalar = fs::make_scalar_backend(dataset);
            const auto generated = fs::Dataset::generate(5, dimension, 101);
            std::vector<fs::Query> queries;
            for (std::size_t i = 0; i < generated->size(); ++i) {
                const auto row = generated->row(i);
                queries.push_back({{row.begin(), row.end()}, i == 0 ? 1 : (i == 1 ? count : std::min(count, std::size_t{10}))});
                const auto found = candidate->search(queries.back());
                verify(*dataset, queries.back(), found);
                const auto baseline = scalar->search(queries.back());
                for (std::size_t k = 0; k < found.ids.size(); ++k) {
                    CHECK(close(found.scores[k], baseline.scores[k]));
                    CHECK(found.ids[k] == baseline.ids[k] || close(found.scores[k], baseline.scores[k]));
                }
            }
            // Grow, shrink, and grow the CUDA buffers; include heterogeneous top-k values.
            for (std::size_t batch_size : {1, 5, 2, 5}) {
                const auto batch = candidate->search_batch(std::span(queries).first(batch_size));
                CHECK(batch.results.size() == batch_size);
                for (std::size_t i = 0; i < batch_size; ++i) verify(*dataset, queries[i], batch.results[i]);
                if (candidate->name() == "cuda") {
                    CHECK(batch.device_timings.has_value());
                    CHECK(batch.device_timings->upload_ms >= 0);
                    CHECK(batch.device_timings->kernel_ms >= 0);
                    CHECK(batch.device_timings->download_ms >= 0);
                } else CHECK(!batch.device_timings.has_value());
            }
        }
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        const std::string backend = argc == 2 ? argv[1] : "scalar";
        if (backend == "scalar") { dataset_tests(); backend_tests(fs::make_scalar_backend); }
        else if (backend == "avx2") {
            if (!fs::avx2_available()) { std::cout << "SKIP: AVX2 unavailable\n"; return 77; }
            backend_tests(fs::make_avx2_backend);
        } else if (backend == "cuda") {
            if (!fs::cuda_available()) { std::cout << "SKIP: " << fs::cuda_status() << '\n'; return 77; }
            backend_tests(fs::make_cuda_backend);
        } else throw std::invalid_argument("Unknown test backend");
        std::cout << backend << ": " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
