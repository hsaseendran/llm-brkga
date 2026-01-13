// examples/rcmadp_example.cpp
// Example usage of the Resource-Constrained Multi-Agent Drop-and-Pick Problem solver
//
// Compile:
//   nvcc -O3 -std=c++17 -I../core -I../configs -I../utils rcmadp_example.cpp -o rcmadp_solver
//
// Run:
//   ./rcmadp_solver <tsp_file> <processing_times_file> [num_agents] [resources_per_agent]
//
// Example:
//   ./rcmadp_solver ../data/berlin52.tsp processing_times.txt 3 2

#include "configs/rcmadp_config.hpp"
#include "core/solver.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

using Float = float;

void print_usage(const char* program) {
    std::cout << "Usage:\n";
    std::cout << "  Mode 1 (TSPJ format): " << program << " --tspj <base_or_TT_file> [num_agents] [resources_per_agent] [options]\n";
    std::cout << "  Mode 2 (separate files): " << program << " <tsp_file> <processing_times_file> [num_agents] [resources_per_agent] [options]\n";
    std::cout << "\nArguments:\n";
    std::cout << "  --tspj                - Use TSPJ format (_TT.csv and _JT.csv paired files)\n";
    std::cout << "  tsp_file              - TSP file with locations (TSPLIB format)\n";
    std::cout << "  processing_times_file - File with processing times for each customer\n";
    std::cout << "  num_agents            - Number of agents (default: 3)\n";
    std::cout << "  resources_per_agent   - Resources per agent (default: 2)\n";
    std::cout << "\nOptions:\n";
    std::cout << "  --pop <size>          - Population size (default: auto-configured)\n";
    std::cout << "  --gen <count>         - Number of generations (default: auto-configured)\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << program << " --tspj ../uploads/berlin52_TSPJ_TT.csv 4 3\n";
    std::cout << "  " << program << " --tspj data/berlin52_TSPJ_TT.csv 4 3 --pop 1000 --gen 500\n";
    std::cout << "  " << program << " berlin52.tsp processing_times.txt 3 2 --pop 2000 --gen 1000\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::unique_ptr<RCMADPConfig<Float>> config;
    int num_agents = 3;
    int resources_per_agent = 2;
    int pop_size = -1;  // -1 means auto-configure
    int num_generations = -1;

    // Helper to parse optional --pop and --gen flags
    auto parse_options = [&](int start_idx) {
        for (int i = start_idx; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--pop" && i + 1 < argc) {
                pop_size = std::stoi(argv[++i]);
            } else if (arg == "--gen" && i + 1 < argc) {
                num_generations = std::stoi(argv[++i]);
            }
        }
    };

    try {
        std::cout << "Loading RCMADP instance...\n";

        // Check for TSPJ mode
        if (std::string(argv[1]) == "--tspj") {
            if (argc < 3) {
                std::cerr << "Error: --tspj requires a file argument\n";
                print_usage(argv[0]);
                return 1;
            }
            std::string tspj_file = argv[2];
            num_agents = (argc > 3 && argv[3][0] != '-') ? std::stoi(argv[3]) : 3;
            resources_per_agent = (argc > 4 && argv[4][0] != '-') ? std::stoi(argv[4]) : 2;
            parse_options(3);

            // Load from TSPJ format
            config = RCMADPConfig<Float>::load_from_tspj(
                tspj_file, num_agents, resources_per_agent);
        } else {
            // Standard mode: TSP file + processing times file
            if (argc < 3) {
                print_usage(argv[0]);
                return 1;
            }
            std::string tsp_file = argv[1];
            std::string proc_file = argv[2];
            num_agents = (argc > 3 && argv[3][0] != '-') ? std::stoi(argv[3]) : 3;
            resources_per_agent = (argc > 4 && argv[4][0] != '-') ? std::stoi(argv[4]) : 2;
            parse_options(3);

            // Load from separate files
            config = RCMADPConfig<Float>::load_from_files(
                tsp_file, proc_file, num_agents, resources_per_agent);
        }

        // Print instance info
        config->print_instance_info();

        // Configure for problem size (auto-configure first)
        RCMADPConfig<Float>::configure_for_size(
            config.get(), config->get_num_customers(), config->get_num_agents());

        // Override with user-specified values if provided
        if (pop_size > 0) {
            config->population_size = pop_size;
            config->elite_size = static_cast<int>(pop_size * 0.15);
            config->mutant_size = static_cast<int>(pop_size * 0.10);
        }
        if (num_generations > 0) {
            config->max_generations = num_generations;
        }

        std::cout << "\nBRKGA Parameters:\n";
        std::cout << "  Population: " << config->population_size << "\n";
        std::cout << "  Generations: " << config->max_generations << "\n";
        std::cout << "  Elite: " << config->elite_size << "\n";
        std::cout << "  Mutants: " << config->mutant_size << "\n";

        // Keep a raw pointer to config for later use (before move)
        RCMADPConfig<Float>* config_ptr = config.get();

        // Create and run solver
        std::cout << "\nStarting optimization...\n";
        auto start = std::chrono::high_resolution_clock::now();

        Solver<Float> solver(std::move(config));
        solver.run();

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();

        // Get best solution
        const auto& best = solver.get_best_individual();

        std::cout << "\n=== Optimization Complete ===\n";
        std::cout << "Time: " << std::fixed << std::setprecision(2) << elapsed << " seconds\n";
        std::cout << "Best Fitness (Total Travel Cost): " << best.fitness << "\n";

        // Decode and print solution details
        try {
            auto solution = config_ptr->decode_solution(best);
            std::cout << "\nSolution Details:\n";
            std::cout << "  Total Travel Cost: " << solution.total_travel_cost << "\n";
            std::cout << "  Makespan: " << solution.makespan << "\n";
            std::cout << "  Unserviced Customers: " << solution.unserviced_customers << "\n";

            for (const auto& schedule : solution.agent_schedules) {
                std::cout << "\n  Agent " << schedule.agent_id << ":\n";
                std::cout << "    Travel Time: " << schedule.total_travel_time << "\n";
                std::cout << "    Dropoffs: " << schedule.dropoffs.size() << " customers\n";
                std::cout << "    Pickups: " << schedule.pickups.size() << " customers\n";
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: Could not decode solution: " << e.what() << "\n";
        }

        std::cout << "\nOptimization finished successfully!\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
