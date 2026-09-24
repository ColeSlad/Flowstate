#pragma once

#include "flowstate/backend.hpp"
#include <charconv>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace flowstate::cli {
struct Options {
    std::size_t vectors = 100000;
    std::size_t dimension = 384;
    std::size_t top_k = 10;
    std::uint32_t seed = 42;
    std::string backend;
    std::size_t batch_size = 1;
    std::size_t iterations = 20;
    std::size_t warmup = 3;
    std::string label = "unspecified";
    std::size_t workers = 2;
    std::size_t requests = 512;
    std::size_t queue_capacity = 1024;
    std::size_t max_wait_us = 2000;
    bool require_all = false;
    bool help = false;
};

inline std::size_t number(std::string_view value) {
    std::size_t result = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
    if (error != std::errc{} || end != value.data() + value.size())
        throw std::invalid_argument("Expected an unsigned integer: " + std::string(value));
    return result;
}

inline Options parse(int argc, char** argv, bool benchmark, bool runtime = false) {
    Options options;
    if (runtime) options.batch_size = 32;
    options.backend = benchmark ? "all" : "auto";
    for (int i = 1; i < argc; ++i) {
        const std::string_view key(argv[i]);
        if (key == "--help") { options.help = true; continue; }
        if (key == "--require-all" && benchmark) { options.require_all = true; continue; }
        if (i + 1 == argc) throw std::invalid_argument("Missing value for " + std::string(key));
        const std::string_view value(argv[++i]);
        if (key == "--vectors") options.vectors = number(value);
        else if (key == "--dimension") options.dimension = number(value);
        else if (key == "--top-k") options.top_k = number(value);
        else if (key == "--seed") {
            const auto seed = number(value);
            if (seed > std::numeric_limits<std::uint32_t>::max()) throw std::invalid_argument("Seed exceeds uint32");
            options.seed = static_cast<std::uint32_t>(seed);
        } else if (key == "--backend") options.backend = value;
        else if (key == "--batch-size" && benchmark) options.batch_size = number(value);
        else if (key == "--iterations" && benchmark) options.iterations = number(value);
        else if (key == "--warmup" && benchmark) options.warmup = number(value);
        else if (key == "--label" && benchmark) options.label = value;
        else if (key == "--workers" && runtime) options.workers = number(value);
        else if (key == "--requests" && runtime) options.requests = number(value);
        else if (key == "--queue-capacity" && runtime) options.queue_capacity = number(value);
        else if (key == "--max-wait-us" && runtime) options.max_wait_us = number(value);
        else throw std::invalid_argument("Unknown option: " + std::string(key));
    }
    if (!options.vectors || !options.dimension || !options.top_k || options.top_k > options.vectors)
        throw std::invalid_argument("Positive vectors/dimension required; top-k must be in [1, vectors]");
    if (!options.batch_size || !options.iterations || !options.warmup)
        throw std::invalid_argument("Batch size, iterations, and warmup must be positive");
    if (options.backend != "auto" && options.backend != "scalar" && options.backend != "avx2" &&
        options.backend != "cuda" && !(benchmark && options.backend == "all"))
        throw std::invalid_argument("Unknown backend: " + options.backend);
    if (runtime && (!options.workers || !options.requests || !options.queue_capacity ||
                    options.requests > options.queue_capacity || options.max_wait_us > 60000000))
        throw std::invalid_argument("Positive workers/requests required; requests must fit the queue; max wait <= 60 seconds");
    if (options.require_all && options.backend != "all")
        throw std::invalid_argument("--require-all requires --backend all");
    return options;
}

inline void help(bool benchmark) {
    std::cout << "Options: --vectors N --dimension D --top-k K --seed S\n"
              << "  --backend auto|scalar|avx2|cuda" << (benchmark ? "|all" : "") << "\n";
    if (benchmark)
        std::cout << "  --batch-size N --iterations N --warmup N --label TEXT --require-all\n";
}

inline std::unique_ptr<Backend> backend(std::string_view name, std::shared_ptr<const Dataset> dataset) {
    if (name == "auto") name = avx2_available() ? "avx2" : "scalar";
    if (name == "scalar") return make_scalar_backend(std::move(dataset));
    if (name == "avx2") return make_avx2_backend(std::move(dataset));
    if (name == "cuda") return make_cuda_backend(std::move(dataset));
    throw std::invalid_argument("Unknown backend");
}

inline std::vector<Query> queries(const Options& options) {
    // Independent deterministic queries; not sampled from rows of the search dataset.
    const auto generated = Dataset::generate(options.batch_size, options.dimension, options.seed ^ 0x9e3779b9U);
    std::vector<Query> result;
    result.reserve(options.batch_size);
    for (std::size_t i = 0; i < options.batch_size; ++i) {
        const auto row = generated->row(i);
        result.push_back({std::vector<float>(row.begin(), row.end()), options.top_k});
    }
    return result;
}
} // namespace flowstate::cli
