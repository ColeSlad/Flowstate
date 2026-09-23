#include "cli.hpp"
#include <chrono>
#include <iomanip>

int main(int argc, char** argv) {
    try {
        const auto options = flowstate::cli::parse(argc, argv, false);
        if (options.help) { flowstate::cli::help(false); return 0; }
        const auto dataset = flowstate::Dataset::generate(options.vectors, options.dimension, options.seed);
        auto backend = flowstate::cli::backend(options.backend, dataset);
        const auto queries = flowstate::cli::queries(options);
        const auto start = std::chrono::steady_clock::now();
        const auto result = backend->search(queries.front());
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        std::cout << "backend=" << backend->name() << " latency_ms=" << ms << '\n';
        std::cout << "id,score\n" << std::setprecision(9);
        for (std::size_t i = 0; i < result.ids.size(); ++i)
            std::cout << result.ids[i] << ',' << result.scores[i] << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "flowstate_search: " << error.what() << '\n';
        return 1;
    }
}
