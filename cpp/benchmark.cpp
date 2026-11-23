
#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

int main() {
    try {
        at::set_num_threads(1);

        auto model = torch::jit::load("models/fashion_cnn.pt");
        model.eval();

        std::vector<float> pixels(28 * 28);

        std::ifstream file("models/test_input.bin", std::ios::binary);

        if (!file) {
            throw std::runtime_error("Cannot open test_input.bin");
        }

        file.read(
            reinterpret_cast<char*>(pixels.data()),
            pixels.size() * sizeof(float)
        );

        if (!file) {
            throw std::runtime_error("Cannot read complete image");
        }

        auto input = torch::from_blob(
            pixels.data(),
            {1, 1, 28, 28},
            torch::kFloat32
        ).clone();

        std::vector<torch::jit::IValue> inputs{input};

        torch::InferenceMode guard;

        const int warmup_runs = 30;
        const int benchmark_runs = 500;

        // Warm-up
        for (int i = 0; i < warmup_runs; ++i) {
            model.forward(inputs);
        }

        std::vector<double> latencies;
        latencies.reserve(benchmark_runs);

        // Timed inference
        for (int i = 0; i < benchmark_runs; ++i) {
            auto start = std::chrono::steady_clock::now();

            auto output = model.forward(inputs);

            auto end = std::chrono::steady_clock::now();

            double elapsed_ms =
                std::chrono::duration<double, std::milli>(
                    end - start
                ).count();

            latencies.push_back(elapsed_ms);
        }

        std::sort(latencies.begin(), latencies.end());

        double sum = std::accumulate(
            latencies.begin(), latencies.end(), 0.0
        );

        double mean = sum / latencies.size();

        double median =
            (latencies[249] + latencies[250]) / 2.0;

        std::cout << std::fixed << std::setprecision(4);

        std::cout << "=== LibTorch C++ CPU Benchmark ===\n";
        std::cout << "CPU threads: 1\n";
        std::cout << "Warmup runs: " << warmup_runs << "\n";
        std::cout << "Measured runs: " << benchmark_runs << "\n";
        std::cout << "Mean latency: " << mean << " ms\n";
        std::cout << "Median latency: " << median << " ms\n";
        std::cout << "Min latency: " << latencies.front() << " ms\n";
        std::cout << "Max latency: " << latencies.back() << " ms\n";
        std::cout << "Estimated throughput: "
                  << 1000.0 / mean << " images/sec\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

