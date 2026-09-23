#pragma once

#include "flowstate/backend.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace flowstate::detail {

struct Candidate { std::size_t id; float score; };
inline bool better(const Candidate& a, const Candidate& b) {
    return a.score > b.score || (a.score == b.score && a.id < b.id);
}

// Exact ordering stays a strict weak ordering; tolerances belong in equivalence tests.
class TopK {
public:
    explicit TopK(std::size_t k) : k_(k) { heap_.reserve(k); }
    void add(std::size_t id, float score) {
        if (!std::isfinite(score)) throw std::overflow_error("Non-finite dot product");
        Candidate candidate{id, score};
        if (heap_.size() < k_) {
            heap_.push_back(candidate);
            std::push_heap(heap_.begin(), heap_.end(), better);
        } else if (better(candidate, heap_.front())) {
            std::pop_heap(heap_.begin(), heap_.end(), better);
            heap_.back() = candidate;
            std::push_heap(heap_.begin(), heap_.end(), better);
        }
    }
    SearchResult finish() {
        std::sort(heap_.begin(), heap_.end(), better);
        SearchResult result;
        result.ids.reserve(heap_.size());
        result.scores.reserve(heap_.size());
        for (const auto& candidate : heap_) {
            result.ids.push_back(candidate.id);
            result.scores.push_back(candidate.score);
        }
        return result;
    }
private:
    std::size_t k_;
    std::vector<Candidate> heap_;
};

} // namespace flowstate::detail
