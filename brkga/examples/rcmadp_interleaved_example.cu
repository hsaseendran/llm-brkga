// examples/rcmadp_interleaved_example.cu
// Example usage of the Interleaved RCMADP solver
//
// Compile:
//   nvcc -O3 -std=c++17 -I.. rcmadp_interleaved_example.cu -o rcmadp_interleaved_solver
//
// Run:
//   ./rcmadp_interleaved_solver --tspj <TT_file> [num_agents] [resources_per_agent] [options]
//
// Example:
//   ./rcmadp_interleaved_solver --tspj ../data/berlin52_TSPJ_TT.csv 4 3 --pop 1000 --gen 500

#include "configs/rcmadp_interleaved_config.hpp"
#include "core/solver.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

using Float = float;

void print_usage(const char* program) {
    std::cout << "RCMADP Interleaved Solver - Fully interleaved dropoff/pickup operations\n\n";
    std::cout << "Usage:\n";
    std::cout << "  " << program << " --tspj <TT_file> [num_agents] [resources_per_agent] [options]\n";
    std::cout << "  " << program << " <tsp_file> <processing_times_file> [num_agents] [resources_per_agent] [options]\n";
    std::cout << "\nArguments:\n";
    std::cout << "  --tspj                - Use TSPJ format (_TT.csv and _JT.csv paired files)\n";
    std::cout << "  num_agents            - Number of agents (default: 3)\n";
    std::cout << "  resources_per_agent   - Resources per agent (default: 2)\n";
    std::cout << "\nOptions:\n";
    std::cout << "  --pop <size>          - Population size (default: auto-configured)\n";
    std::cout << "  --gen <count>         - Number of generations (default: auto-configured)\n";
    std::cout << "  --output <file>       - Output JSON solution file (for generate_html_report.py)\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << program << " --tspj ../data/berlin52_TSPJ_TT.csv 4 3\n";
    std::cout << "  " << program << " --tspj ../data/berlin52_TSPJ_TT.csv 4 3 --pop 1000 --gen 500\n";
    std::cout << "  " << program << " data/berlin52.tsp data/proc_times.txt 6 4 --output soln.json\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::unique_ptr<RCMADPInterleavedConfig<Float>> config;
    int num_agents = 3;
    int resources_per_agent = 2;
    int pop_size = -1;
    int num_generations = -1;
    std::string output_json_file;

    auto parse_options = [&](int start_idx) {
        for (int i = start_idx; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--pop" && i + 1 < argc) {
                pop_size = std::stoi(argv[++i]);
            } else if (arg == "--gen" && i + 1 < argc) {
                num_generations = std::stoi(argv[++i]);
            } else if (arg == "--output" && i + 1 < argc) {
                output_json_file = argv[++i];
            }
        }
    };

    try {
        std::cout << "Loading RCMADP Interleaved instance...\n";

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

            config = RCMADPInterleavedConfig<Float>::load_from_tspj(
                tspj_file, num_agents, resources_per_agent);
        } else {
            if (argc < 3) {
                print_usage(argv[0]);
                return 1;
            }
            std::string tsp_file = argv[1];
            std::string proc_file = argv[2];
            num_agents = (argc > 3 && argv[3][0] != '-') ? std::stoi(argv[3]) : 3;
            resources_per_agent = (argc > 4 && argv[4][0] != '-') ? std::stoi(argv[4]) : 2;
            parse_options(3);

            config = RCMADPInterleavedConfig<Float>::load_from_files(
                tsp_file, proc_file, num_agents, resources_per_agent);
        }

        config->print_instance_info();

        RCMADPInterleavedConfig<Float>::configure_for_size(
            config.get(), config->get_num_customers(), config->get_num_agents());

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

        RCMADPInterleavedConfig<Float>* config_ptr = config.get();

        std::cout << "\nStarting optimization (interleaved mode)...\n";
        auto start = std::chrono::high_resolution_clock::now();

        Solver<Float> solver(std::move(config));
        solver.run();

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();

        const auto& best = solver.get_best_individual();

        std::cout << "\n=== Optimization Complete ===\n";
        std::cout << "Time: " << std::fixed << std::setprecision(2) << elapsed << " seconds\n";
        std::cout << "Best Fitness: " << best.fitness << "\n";

        // Print detailed solution
        config_ptr->print_solution(best);

        // Export JSON if output file specified
        if (!output_json_file.empty()) {
            auto convergence = solver.get_convergence_history();
            config_ptr->export_solution_json(best, output_json_file, elapsed, convergence);
            std::cout << "\nJSON solution exported to: " << output_json_file << "\n";
        }

        std::cout << "\nOptimization finished successfully!\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
