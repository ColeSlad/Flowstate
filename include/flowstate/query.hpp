#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace flowstate {

struct Query {
    std::vector<float> values;
    std::size_t top_k = 10;
};

struct SearchResult {
    std::vector<std::size_t> ids;
    std::vector<float> scores;
};

class Dataset {
public:
    Dataset(std::size_t count, std::size_t dimension, std::vector<float> values);
    static std::shared_ptr<const Dataset> generate(std::size_t count, std::size_t dimension,
                                                  std::uint32_t seed = 42);
    [[nodiscard]] std::size_t size() const { return count_; }
    [[nodiscard]] std::size_t dimension() const { return dimension_; }
    [[nodiscard]] std::span<const float> values() const { return values_; }
    [[nodiscard]] std::span<const float> row(std::size_t index) const;
private:
    std::size_t count_;
    std::size_t dimension_;
    std::vector<float> values_;
};

void validate_query(const Dataset& dataset, const Query& query);
std::size_t checked_product(std::size_t a, std::size_t b);

} // namespace flowstate
