#include "search_detail.hpp"

namespace flowstate {
namespace {
class ScalarBackend final : public Backend {
public:
    explicit ScalarBackend(std::shared_ptr<const Dataset> dataset) : dataset_(std::move(dataset)) {
        if (!dataset_) throw std::invalid_argument("Dataset must not be null");
    }
    SearchResult search(const Query& query) override {
        validate_query(*dataset_, query);
        detail::TopK top(query.top_k);
        for (std::size_t id = 0; id < dataset_->size(); ++id) {
            const auto row = dataset_->row(id);
            float score = 0;
            for (std::size_t d = 0; d < row.size(); ++d) score += row[d] * query.values[d];
            top.add(id, score);
        }
        return top.finish();
    }
    std::string_view name() const override { return "scalar"; }
private:
    std::shared_ptr<const Dataset> dataset_;
};
} // namespace

std::unique_ptr<Backend> make_scalar_backend(std::shared_ptr<const Dataset> dataset) {
    return std::make_unique<ScalarBackend>(std::move(dataset));
}
} // namespace flowstate
