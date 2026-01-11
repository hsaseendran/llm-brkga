// benchmark_baseline.cu - Baseline performance measurement before BrkgaCuda 2.0 optimizations
// Records current performance for comparison after stream/coalescing/bb-segsort integration

#include "core/solver.hpp"
#include "configs/tsp_config.hpp"
#include "configs/knapsack_config.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <iomanip>
#include <sstream>

struct BenchmarkResult {
    std::string problem_type;
    std::string instance_name;
    int problem_size;
    int pop_size;
    int num_generations;
    double best_fitness;
    double avg_generation_time_ms;
    double total_time_seconds;
    double memory_bandwidth_gbs;
    int num_gpus;
    std::string optimization_mode;  // "baseline", "streams", "coalesced", "bbsegsort", "full"
};

void write_results_csv(const std::vector<BenchmarkResult>& results, const std::string& filename) {
    std::ofstream file(filename, std::ios::app);

    // Write header if file is empty
    file.seekp(0, std::ios::end);
    if (file.tellp() == 0) {
        file << "problem_type,instance_name,problem_size,pop_size,num_generations,best_fitness,"
             << "avg_gen_time_ms,total_time_s,memory_bandwidth_gbs,num_gpus,optimization_mode,timestamp\n";
    }

    // Get timestamp
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    char timestamp[64];
    std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", std::localtime(&time));

    for (const auto& r : results) {
        file << r.problem_type << ","
             << r.instance_name << ","
             << r.problem_size << ","
             << r.pop_size << ","
             << r.num_generations << ","
             << std::fixed << std::setprecision(6) << r.best_fitness << ","
             << std::setprecision(3) << r.avg_generation_time_ms << ","
             << std::setprecision(2) << r.total_time_seconds << ","
             << std::setprecision(2) << r.memory_bandwidth_gbs << ","
             << r.num_gpus << ","
             << r.optimization_mode << ","
             << timestamp << "\n";
    }

    file.close();
}

BenchmarkResult benchmark_tsp(const std::string& tsp_file, int pop_size, int max_gens,
                               int num_gpus, const std::string& opt_mode) {
    BenchmarkResult result;
    result.problem_type = "TSP";
    result.optimization_mode = opt_mode;
    result.num_gpus = num_gpus;

    std::cout << "\n=========================================" << std::endl;
    std::cout << "Benchmarking TSP: " << tsp_file << std::endl;
    std::cout << "Mode: " << opt_mode << std::endl;

    try {
        auto config = TSPConfig<float>::load_from_file(tsp_file);
        result.instance_name = config->get_instance_name();
        result.problem_size = config->get_num_cities();

        config->population_size = pop_size;
        config->elite_size = pop_size / 5;
        config->mutant_size = pop_size / 10;
        config->max_generations = max_gens;

        result.pop_size = pop_size;
        result.num_generations = max_gens;

        std::cout << "Cities: " << result.problem_size << std::endl;
        std::cout << "Population: " << pop_size << std::endl;
        std::cout << "Generations: " << max_gens << std::endl;
        std::cout << "GPUs: " << num_gpus << " (limiting to single GPU for baseline)" << std::endl;

        // Force single GPU mode for baseline benchmark
        // Note: Multi-GPU islands configuration will be added in future optimization phases
        Solver<float> solver(std::move(config), false, max_gens / 10);  // verbose=false to reduce output

        auto start = std::chrono::high_resolution_clock::now();
        solver.run();
        auto end = std::chrono::high_resolution_clock::now();

        result.total_time_seconds = std::chrono::duration<double>(end - start).count();
        result.avg_generation_time_ms = (result.total_time_seconds * 1000.0) / max_gens;
        result.best_fitness = solver.get_best_individual().fitness;

        // Estimate memory bandwidth (rough calculation)
        // Each generation: read population, write offspring, read fitness
        size_t bytes_per_gen = pop_size * result.problem_size * sizeof(float) * 3;
        result.memory_bandwidth_gbs = (bytes_per_gen * max_gens) / (result.total_time_seconds * 1e9);

        std::cout << "Best fitness: " << std::fixed << std::setprecision(2) << result.best_fitness << std::endl;
        std::cout << "Total time: " << std::setprecision(2) << result.total_time_seconds << " seconds" << std::endl;
        std::cout << "Avg generation: " << std::setprecision(3) << result.avg_generation_time_ms << " ms" << std::endl;
        std::cout << "Est. bandwidth: " << std::setprecision(2) << result.memory_bandwidth_gbs << " GB/s" << std::endl;
        std::cout << "=========================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        result.best_fitness = -1;
        result.total_time_seconds = -1;
    }

    return result;
}

BenchmarkResult benchmark_knapsack(int num_items, int pop_size, int max_gens,
                                    int num_gpus, const std::string& opt_mode) {
    BenchmarkResult result;
    result.problem_type = "Knapsack";
    result.instance_name = "synthetic_" + std::to_string(num_items);
    result.problem_size = num_items;
    result.optimization_mode = opt_mode;
    result.num_gpus = num_gpus;

    std::cout << "\n=========================================" << std::endl;
    std::cout << "Benchmarking Knapsack: " << num_items << " items" << std::endl;
    std::cout << "Mode: " << opt_mode << std::endl;

    try {
        auto config = KnapsackConfig<float>::create_random(num_items, 1000);

        config->population_size = pop_size;
        config->elite_size = pop_size / 5;
        config->mutant_size = pop_size / 10;
        config->max_generations = max_gens;

        result.pop_size = pop_size;
        result.num_generations = max_gens;

        std::cout << "Items: " << num_items << std::endl;
        std::cout << "Population: " << pop_size << std::endl;
        std::cout << "Generations: " << max_gens << std::endl;
        std::cout << "GPUs: " << num_gpus << " (limiting to single GPU for baseline)" << std::endl;

        // Force single GPU mode for baseline benchmark
        // Note: Multi-GPU islands configuration will be added in future optimization phases
        Solver<float> solver(std::move(config), false, max_gens / 10);  // verbose=false to reduce output

        auto start = std::chrono::high_resolution_clock::now();
        solver.run();
        auto end = std::chrono::high_resolution_clock::now();

        result.total_time_seconds = std::chrono::duration<double>(end - start).count();
        result.avg_generation_time_ms = (result.total_time_seconds * 1000.0) / max_gens;
        result.best_fitness = solver.get_best_individual().fitness;

        // Estimate memory bandwidth
        size_t bytes_per_gen = pop_size * num_items * sizeof(float) * 3;
        result.memory_bandwidth_gbs = (bytes_per_gen * max_gens) / (result.total_time_seconds * 1e9);

        std::cout << "Best fitness: " << std::fixed << std::setprecision(2) << result.best_fitness << std::endl;
        std::cout << "Total time: " << std::setprecision(2) << result.total_time_seconds << " seconds" << std::endl;
        std::cout << "Avg generation: " << std::setprecision(3) << result.avg_generation_time_ms << " ms" << std::endl;
        std::cout << "Est. bandwidth: " << std::setprecision(2) << result.memory_bandwidth_gbs << " GB/s" << std::endl;
        std::cout << "=========================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        result.best_fitness = -1;
        result.total_time_seconds = -1;
    }

    return result;
}

int main(int argc, char* argv[]) {
    std::string tsp_file = "";
    int pop_size = 8000;
    int max_gens = 500;
    int num_gpus = 1;
    std::string opt_mode = "baseline";
    std::string output_file = "benchmark_results.csv";
    bool run_tsp = true;
    bool run_knapsack = true;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--tsp" && i + 1 < argc) {
            tsp_file = argv[++i];
        } else if (arg == "--pop" && i + 1 < argc) {
            pop_size = std::stoi(argv[++i]);
        } else if (arg == "--gens" && i + 1 < argc) {
            max_gens = std::stoi(argv[++i]);
        } else if (arg == "--gpus" && i + 1 < argc) {
            num_gpus = std::stoi(argv[++i]);
        } else if (arg == "--mode" && i + 1 < argc) {
            opt_mode = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "--tsp-only") {
            run_knapsack = false;
        } else if (arg == "--knapsack-only") {
            run_tsp = false;
        } else if (arg == "--help") {
            std::cout << "Baseline Benchmark Tool\n"
                      << "Usage: " << argv[0] << " [options]\n\n"
                      << "Options:\n"
                      << "  --tsp <file>       TSP instance file (TSPLIB format)\n"
                      << "  --pop <size>       Population size (default: 8000)\n"
                      << "  --gens <n>         Number of generations (default: 500)\n"
                      << "  --gpus <n>         Number of GPUs (default: 1)\n"
                      << "  --mode <mode>      Optimization mode: baseline|streams|coalesced|bbsegsort|full\n"
                      << "  --output <file>    Output CSV file (default: benchmark_results.csv)\n"
                      << "  --tsp-only         Run only TSP benchmarks\n"
                      << "  --knapsack-only    Run only Knapsack benchmarks\n"
                      << "  --help             Show this help\n";
            return 0;
        }
    }

    std::cout << "=========================================" << std::endl;
    std::cout << "   Baseline Performance Benchmark        " << std::endl;
    std::cout << "   BrkgaCuda 2.0 Optimization Project    " << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "Population size: " << pop_size << std::endl;
    std::cout << "Generations: " << max_gens << std::endl;
    std::cout << "GPUs: " << num_gpus << std::endl;
    std::cout << "Optimization mode: " << opt_mode << std::endl;
    std::cout << "Output file: " << output_file << std::endl;
    std::cout << "=========================================" << std::endl;

    std::vector<BenchmarkResult> results;

    // TSP benchmarks
    if (run_tsp && !tsp_file.empty()) {
        results.push_back(benchmark_tsp(tsp_file, pop_size, max_gens, num_gpus, opt_mode));
    }

    // Knapsack benchmarks (different sizes)
    if (run_knapsack) {
        std::vector<int> knapsack_sizes = {100, 500, 1000};
        for (int size : knapsack_sizes) {
            results.push_back(benchmark_knapsack(size, pop_size, max_gens, num_gpus, opt_mode));
        }
    }

    // Write results
    write_results_csv(results, output_file);

    std::cout << "\n=========================================" << std::endl;
    std::cout << "Benchmark complete!" << std::endl;
    std::cout << "Results appended to: " << output_file << std::endl;
    std::cout << "=========================================" << std::endl;

    return 0;
}
