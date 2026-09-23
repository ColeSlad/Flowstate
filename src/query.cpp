#include "flowstate/backend.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace flowstate {

std::size_t checked_product(std::size_t a, std::size_t b) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::length_error("Size multiplication overflow");
    return a * b;
}

Dataset::Dataset(std::size_t count, std::size_t dimension, std::vector<float> values)
    : count_(count), dimension_(dimension), values_(std::move(values)) {
    if (count == 0 || dimension == 0)
        throw std::invalid_argument("Dataset size and dimension must be positive");
    if (values_.size() != checked_product(count, dimension))
        throw std::invalid_argument("Dataset shape does not match its storage");
    if (!std::all_of(values_.begin(), values_.end(), [](float v) { return std::isfinite(v); }))
        throw std::invalid_argument("Dataset values must be finite");
}

std::shared_ptr<const Dataset> Dataset::generate(std::size_t count, std::size_t dimension,
                                                std::uint32_t seed) {
    if (count == 0 || dimension == 0)
        throw std::invalid_argument("Dataset size and dimension must be positive");
    std::vector<float> values(checked_product(count, dimension));
    std::mt19937 random(seed);
    // An explicit integer-to-float mapping avoids library-specific distribution algorithms.
    for (auto& value : values)
        value = static_cast<float>(random() >> 8) * 0x1p-23f - 1.0f;
    return std::make_shared<const Dataset>(count, dimension, std::move(values));
}

std::span<const float> Dataset::row(std::size_t index) const {
    if (index >= count_) throw std::out_of_range("Dataset row out of range");
    return std::span<const float>(values_).subspan(index * dimension_, dimension_);
}

void validate_query(const Dataset& dataset, const Query& query) {
    if (query.values.size() != dataset.dimension())
        throw std::invalid_argument("Query dimension must match the dataset");
    if (query.top_k == 0 || query.top_k > dataset.size())
        throw std::invalid_argument("top-k must be in [1, dataset size]");
    if (!std::all_of(query.values.begin(), query.values.end(), [](float v) { return std::isfinite(v); }))
        throw std::invalid_argument("Query values must be finite");
}

BatchResult Backend::search_batch(std::span<const Query> queries) {
    BatchResult batch;
    batch.results.reserve(queries.size());
    for (const auto& query : queries) batch.results.push_back(search(query));
    return batch;
}

} // namespace flowstate
