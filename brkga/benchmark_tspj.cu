// benchmark_tspj.cu - Comprehensive TSPJ Benchmark for Medium and Large Problems
// Uses 8 L40S GPUs with island model for optimal performance

#include "core/solver.hpp"
#include "configs/tspj_config.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <filesystem>
#include <iomanip>
#include <algorithm>

namespace fs = std::filesystem;

struct BenchmarkResult {
    std::string instance_name;
    std::string problem_size;  // "Medium" or "Large"
    std::string batch;
    int num_cities;
    double best_fitness;
    double execution_time_seconds;
    int generations_run;
    int num_gpus_used;
};

std::vector<std::string> find_cost_files(const std::string& directory) {
    std::vector<std::string> files;
    for (const auto& entry : fs::directory_iterator(directory)) {
        std::string filename = entry.path().filename().string();
        if (filename.find("_cost_table_by_coordinates.csv") != std::string::npos) {
            files.push_back(entry.path().string());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

BenchmarkResult run_benchmark(const std::string& data_file, const std::string& problem_size,
                               const std::string& batch, int num_gpus, int max_gens_override = -1,
                               int pop_size_override = -1) {
    BenchmarkResult result;
    result.problem_size = problem_size;
    result.batch = batch;
    result.num_gpus_used = num_gpus;

    try {
        std::cout << "\n=========================================" << std::endl;
        std::cout << "Loading: " << data_file << std::endl;

        auto config = TSPJConfig<float>::load_from_file(data_file);
        result.instance_name = config->get_instance_name();
        result.num_cities = config->get_num_cities();

        std::cout << "Instance: " << result.instance_name << std::endl;
        std::cout << "Cities: " << result.num_cities << std::endl;

        // Use multi-GPU optimized configuration
        TSPJConfig<float>::configure_for_multi_gpu(config.get(), result.num_cities, num_gpus);

        // Override max generations if specified
        if (max_gens_override > 0) {
            config->max_generations = max_gens_override;
        }

        // Override population size if specified
        if (pop_size_override > 0) {
            config->population_size = pop_size_override;
            config->elite_size = pop_size_override / 5;  // 20% elite
            config->mutant_size = pop_size_override / 10;  // 10% mutants
        }

        result.generations_run = config->max_generations;

        std::cout << "Population: " << config->population_size << std::endl;
        std::cout << "Generations: " << config->max_generations << std::endl;
        std::cout << "GPUs: " << num_gpus << " (island model)" << std::endl;
        std::cout << "Creating solver..." << std::flush;

        // Create solver (verbose=true, print every 10 generations)
        Solver<float> solver(std::move(config), true, 10);
        std::cout << " done." << std::endl;

        auto start = std::chrono::high_resolution_clock::now();
        solver.run();
        auto end = std::chrono::high_resolution_clock::now();

        result.execution_time_seconds = std::chrono::duration<double>(end - start).count();
        result.best_fitness = solver.get_best_individual().fitness;

        std::cout << "Best fitness: " << std::fixed << std::setprecision(2)
                  << result.best_fitness << std::endl;
        std::cout << "Time: " << std::setprecision(2)
                  << result.execution_time_seconds << " seconds" << std::endl;
        std::cout << "=========================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error processing " << data_file << ": " << e.what() << std::endl;
        result.best_fitness = -1;
        result.execution_time_seconds = -1;
    }

    return result;
}

void write_results_csv(const std::vector<BenchmarkResult>& results, const std::string& filename) {
    std::ofstream file(filename);
    file << "instance_name,problem_size,batch,num_cities,best_fitness,execution_time_seconds,generations,num_gpus" << std::endl;

    for (const auto& r : results) {
        file << r.instance_name << ","
             << r.problem_size << ","
             << r.batch << ","
             << r.num_cities << ","
             << std::fixed << std::setprecision(6) << r.best_fitness << ","
             << std::setprecision(2) << r.execution_time_seconds << ","
             << r.generations_run << ","
             << r.num_gpus_used << std::endl;
    }

    file.close();
    std::cout << "\nResults saved to: " << filename << std::endl;
}

void print_summary(const std::vector<BenchmarkResult>& results) {
    std::cout << "\n=================================================" << std::endl;
    std::cout << "                 BENCHMARK SUMMARY                 " << std::endl;
    std::cout << "=================================================" << std::endl;

    double total_time = 0;
    int medium_count = 0, large_count = 0;

    std::cout << std::left << std::setw(20) << "Instance"
              << std::setw(10) << "Size"
              << std::setw(8) << "Cities"
              << std::setw(15) << "Best Fitness"
              << std::setw(12) << "Time (s)" << std::endl;
    std::cout << std::string(65, '-') << std::endl;

    for (const auto& r : results) {
        if (r.best_fitness > 0) {
            std::cout << std::left << std::setw(20) << r.instance_name
                      << std::setw(10) << r.problem_size
                      << std::setw(8) << r.num_cities
                      << std::setw(15) << std::fixed << std::setprecision(2) << r.best_fitness
                      << std::setw(12) << r.execution_time_seconds << std::endl;

            total_time += r.execution_time_seconds;
            if (r.problem_size == "Medium") medium_count++;
            else large_count++;
        }
    }

    std::cout << std::string(65, '-') << std::endl;
    std::cout << "Medium problems: " << medium_count << std::endl;
    std::cout << "Large problems: " << large_count << std::endl;
    std::cout << "Total time: " << std::setprecision(2) << total_time << " seconds" << std::endl;
    std::cout << "            (" << std::setprecision(2) << (total_time / 60.0) << " minutes)" << std::endl;
    std::cout << "=================================================" << std::endl;
}

int main(int argc, char* argv[]) {
    std::string base_path = "data/";
    int num_gpus = 8;
    int max_gens = -1;  // Use default based on problem size
    int pop_size = -1;  // Population size override (-1 = auto)
    bool run_medium = true;
    bool run_large = true;
    std::string specific_batch = "";  // Run specific batch only
    int limit_per_batch = -1;  // Limit instances per batch (-1 = all)

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--base-path" && i + 1 < argc) {
            base_path = argv[++i];
        } else if (arg == "--gpus" && i + 1 < argc) {
            num_gpus = std::stoi(argv[++i]);
        } else if (arg == "--max-gens" && i + 1 < argc) {
            max_gens = std::stoi(argv[++i]);
        } else if (arg == "--medium-only") {
            run_large = false;
        } else if (arg == "--large-only") {
            run_medium = false;
        } else if (arg == "--batch" && i + 1 < argc) {
            specific_batch = argv[++i];
        } else if (arg == "--limit" && i + 1 < argc) {
            limit_per_batch = std::stoi(argv[++i]);
        } else if (arg == "--pop" && i + 1 < argc) {
            pop_size = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "TSPJ Benchmark Usage:" << std::endl;
            std::cout << "  --base-path <path>  Base path to data directory (default: data/)" << std::endl;
            std::cout << "  --gpus <n>          Number of GPUs to use (default: 8)" << std::endl;
            std::cout << "  --max-gens <n>      Override max generations (default: auto)" << std::endl;
            std::cout << "  --pop <n>           Override population size (default: auto)" << std::endl;
            std::cout << "  --medium-only       Run only Medium problems" << std::endl;
            std::cout << "  --large-only        Run only Large problems" << std::endl;
            std::cout << "  --batch <name>      Run specific batch only (e.g., Batch_01)" << std::endl;
            std::cout << "  --limit <n>         Limit instances per batch" << std::endl;
            return 0;
        }
    }

    std::cout << "=========================================" << std::endl;
    std::cout << "   TSPJ BRKGA Benchmark Suite           " << std::endl;
    std::cout << "   8 x L40S GPU Island Model            " << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "GPUs: " << num_gpus << std::endl;
    std::cout << "Base path: " << base_path << std::endl;
    std::cout << "Medium problems: " << (run_medium ? "Yes" : "No") << std::endl;
    std::cout << "Large problems: " << (run_large ? "Yes" : "No") << std::endl;
    if (!specific_batch.empty()) {
        std::cout << "Specific batch: " << specific_batch << std::endl;
    }
    if (limit_per_batch > 0) {
        std::cout << "Limit per batch: " << limit_per_batch << std::endl;
    }
    std::cout << "=========================================" << std::endl;

    std::vector<BenchmarkResult> all_results;
    std::vector<std::string> batches = {"Batch_01", "Batch_02", "Batch_03", "Batch_04"};

    // Filter batches if specific one requested
    if (!specific_batch.empty()) {
        batches = {specific_batch};
    }

    // Run Medium problems
    if (run_medium) {
        std::cout << "\n*** MEDIUM PROBLEMS ***" << std::endl;
        for (const auto& batch : batches) {
            std::string dir = base_path + "Medium_problems/" + batch + "/";
            if (!fs::exists(dir)) {
                std::cout << "Directory not found: " << dir << std::endl;
                continue;
            }

            auto files = find_cost_files(dir);
            std::cout << "\n[" << batch << "] Found " << files.size() << " instances" << std::endl;

            int count = 0;
            for (const auto& file : files) {
                if (limit_per_batch > 0 && count >= limit_per_batch) break;
                auto result = run_benchmark(file, "Medium", batch, num_gpus, max_gens, pop_size);
                all_results.push_back(result);
                count++;
            }
        }
    }

    // Run Large problems
    if (run_large) {
        std::cout << "\n*** LARGE PROBLEMS ***" << std::endl;
        for (const auto& batch : batches) {
            std::string dir = base_path + "Large_problems/" + batch + "/";
            if (!fs::exists(dir)) {
                std::cout << "Directory not found: " << dir << std::endl;
                continue;
            }

            auto files = find_cost_files(dir);
            std::cout << "\n[" << batch << "] Found " << files.size() << " instances" << std::endl;

            int count = 0;
            for (const auto& file : files) {
                if (limit_per_batch > 0 && count >= limit_per_batch) break;
                auto result = run_benchmark(file, "Large", batch, num_gpus, max_gens, pop_size);
                all_results.push_back(result);
                count++;
            }
        }
    }

    // Generate timestamp for output filename
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    char timestamp[64];
    std::strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", std::localtime(&time));

    std::string output_file = "tspj_benchmark_results_" + std::string(timestamp) + ".csv";
    write_results_csv(all_results, output_file);
    print_summary(all_results);

    return 0;
}
