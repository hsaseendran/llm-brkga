// main.cu - Updated with multi-objective NSGA-II support and complete solution export
#include <iostream>
#include <string>
#include <fstream>
#include <cstdlib>
#include <map>
#include <vector>
#include <algorithm>
#include <iomanip>

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " --data <file> --config <config_name> [options]" << std::endl;
    std::cout << "\nRequired:" << std::endl;
    std::cout << "  --data <file>        Input data file" << std::endl;
    std::cout << "  --config <name>      Config name" << std::endl;
    std::cout << "\nOptional:" << std::endl;
    std::cout << "  --verbose            Enable verbose output" << std::endl;
    std::cout << "  --benchmark          Run performance benchmark" << std::endl;
    std::cout << "  --pop <size>         Override population size" << std::endl;
    std::cout << "  --gen <count>        Override generation count" << std::endl;
    std::cout << "\nAvailable configs:" << std::endl;
    std::cout << "  Single-Objective (BRKGA):" << std::endl;
    std::cout << "    tsp_config        - Traveling Salesman Problem" << std::endl;
    std::cout << "    tspj_config       - TSP with Job assignment" << std::endl;
    std::cout << "    knapsack_config   - 0/1 Knapsack Problem" << std::endl;
    std::cout << "\n  Multi-Objective (NSGA-II):" << std::endl;
    std::cout << "    multi_tsp_config  - Multi-objective TSP (distance + time)" << std::endl;
    std::cout << "    multi_objective_vrp_config - Multi-objective VRP (distance + max route)" << std::endl;
    std::cout << "\nExamples:" << std::endl;
    std::cout << "  " << program_name << " --data data/berlin52.tsp --config tsp_config" << std::endl;
    std::cout << "  " << program_name << " --data data/berlin52.tsp --config multi_tsp_config --pop 1000" << std::endl;
    std::cout << "  " << program_name << " --data data/A-n32-k5.vrp --config multi_objective_vrp_config" << std::endl;
    std::cout << "  " << program_name << " --data data/knapsack.txt --config knapsack_config --benchmark" << std::endl;
}

bool file_exists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}

// Check if config is multi-objective
bool is_multi_objective_config(const std::string& config_name) {
    return config_name == "multi_tsp_config" || config_name == "multi_objective_vrp_config";
}

bool compile_and_run(const std::string& config_name, const std::string& data_file, 
                     bool verbose = false, bool benchmark = false,
                     int custom_pop = -1, int custom_gen = -1) {
    
    std::string header_file = "configs/" + config_name + ".hpp";
    
    if (!file_exists(header_file)) {
        std::cerr << "Error: Config file not found: " << header_file << std::endl;
        return false;
    }
    
    if (!file_exists(data_file)) {
        std::cerr << "Error: Data file not found: " << data_file << std::endl;
        return false;
    }
    
    std::cout << "✓ Found config file: " << header_file << std::endl;
    std::cout << "✓ Found data file: " << data_file << std::endl;
    
    bool is_multi_obj = is_multi_objective_config(config_name);
    std::cout << "✓ Optimization mode: " << (is_multi_obj ? "Multi-Objective (NSGA-II)" : "Single-Objective (BRKGA)") << std::endl;
    
    // Create temporary main file
    std::string temp_main = "temp_main_" + config_name + ".cu";
    std::string executable = "brkga_unified";
    
    std::ofstream temp_file(temp_main);
    if (!temp_file.is_open()) {
        std::cerr << "Error: Cannot create temporary file: " << temp_main << std::endl;
        return false;
    }
    
    // Write the unified main file
    temp_file << "#include \"core/solver.hpp\"\n";
    temp_file << "#include \"configs/" << config_name << ".hpp\"\n";
    temp_file << "#include <iostream>\n";
    temp_file << "#include <memory>\n";
    temp_file << "#include <iomanip>\n\n";
    
    temp_file << "int main() {\n";
    temp_file << "    try {\n";
    temp_file << "        std::cout << \"Loading data file: " << data_file << "\" << std::endl;\n";
    
    // Generate config loading code
    if (config_name == "tsp_config") {
        temp_file << "        auto config = TSPConfig<float>::load_from_file(\"" << data_file << "\");\n";
        temp_file << "        config->print_instance_info();\n";
        temp_file << "        TSPConfig<float>::configure_for_size(config.get(), config->get_num_cities());\n";
    } else if (config_name == "tspj_config") {
        temp_file << "        auto config = TSPJConfig<float>::load_from_file(\"" << data_file << "\");\n";
        temp_file << "        config->print_instance_info();\n";
        temp_file << "        TSPJConfig<float>::configure_for_size(config.get(), config->get_num_cities());\n";
    } else if (config_name == "knapsack_config") {
        temp_file << "        auto config = KnapsackConfig<float>::load_from_file(\"" << data_file << "\");\n";
        temp_file << "        config->print_instance_info();\n";
        temp_file << "        KnapsackConfig<float>::configure_for_size(config.get(), config->get_weights().size());\n";
    } else if (config_name == "multi_tsp_config") {
        temp_file << "        auto config = MultiTSPConfig<float>::load_from_file(\"" << data_file << "\");\n";
        temp_file << "        config->print_instance_info();\n";
        temp_file << "        MultiTSPConfig<float>::configure_for_size(config.get(), config->get_num_cities());\n";
    } else if (config_name == "multi_objective_vrp_config") {
        temp_file << "        auto config = MultiObjectiveVRPConfig<float>::load_from_file(\"" << data_file << "\");\n";
        temp_file << "        config->print_instance_info();\n";
        temp_file << "        MultiObjectiveVRPConfig<float>::configure_for_size(config.get(), config->get_num_customers());\n";
    } else {
        std::cerr << "Error: Unknown config type: " << config_name << std::endl;
        temp_file.close();
        std::remove(temp_main.c_str());
        return false;
    }
    
    // Override parameters if specified
    if (custom_pop > 0) {
        temp_file << "        \n        // Override population size\n";
        temp_file << "        config->population_size = " << custom_pop << ";\n";
        temp_file << "        config->elite_size = config->population_size / 5;\n";
        temp_file << "        config->mutant_size = config->population_size / 10;\n";
        temp_file << "        config->update_cuda_grid_size();\n";
        temp_file << "        std::cout << \"Population size overridden to: " << custom_pop << "\" << std::endl;\n";
    }
    
    if (custom_gen > 0) {
        temp_file << "        \n        // Override generation count\n";
        temp_file << "        config->max_generations = " << custom_gen << ";\n";
        temp_file << "        std::cout << \"Max generations overridden to: " << custom_gen << "\" << std::endl;\n";
    }
    
    temp_file << "        \n";
    temp_file << "        std::cout << \"Configuration loaded successfully\" << std::endl;\n";
    temp_file << "        config->print_config();\n";
    temp_file << "        \n";
    
    // Create solver and run
    temp_file << "        // Create and run solver\n";
    temp_file << "        Solver<float> solver(std::move(config), " << (verbose ? "true" : "true") << ", 50);\n";
    
    // Benchmark mode for multi-objective
    if (benchmark && is_multi_obj) {
        temp_file << "        \n        std::cout << \"\\n=== Running Benchmark Mode ===\" << std::endl;\n";
        temp_file << "        solver.benchmark_modes();\n";
        temp_file << "        std::cout << \"================================\" << std::endl;\n";
    }
    
    temp_file << "        \n        solver.run();\n";
    temp_file << "        \n";
    
    // Handle results based on problem type
    if (is_multi_obj) {
        temp_file << "        // Export Pareto front for multi-objective problem\n";
        temp_file << "        solver.export_pareto_front(\"" << config_name << "_pareto_front.txt\");\n";
        temp_file << "        std::cout << \"\\n✓ Pareto front exported to: " << config_name << "_pareto_front.txt\" << std::endl;\n";
        temp_file << "        \n";
        
        // Export all individual solutions for VRP
        if (config_name == "multi_objective_vrp_config") {
            temp_file << "        // Export all individual VRP solutions\n";
            temp_file << "        {\n";
            temp_file << "            auto pareto_solutions = solver.get_pareto_front();\n";
            temp_file << "            if (!pareto_solutions.empty()) {\n";
            temp_file << "                auto vrp_config = dynamic_cast<MultiObjectiveVRPConfig<float>*>(solver.get_config());\n";
            temp_file << "                if (vrp_config) {\n";
            temp_file << "                    vrp_config->export_all_pareto_solutions(pareto_solutions, \"pareto_solutions\");\n";
            temp_file << "                    std::cout << \"\\\\n✓ All \" << pareto_solutions.size() << \" solutions exported to: pareto_solutions/\" << std::endl;\n";
            temp_file << "                }\n";
            temp_file << "            }\n";
            temp_file << "        }\n";
            temp_file << "        \n";
        }
        
        temp_file << "        // Print Pareto front analysis\n";
        temp_file << "        auto pareto = solver.get_pareto_front();\n";
        temp_file << "        std::cout << \"\\n=== Pareto Front Analysis ===\" << std::endl;\n";
        temp_file << "        std::cout << \"Solutions in Pareto front: \" << pareto.size() << std::endl;\n";
        temp_file << "        \n";
        temp_file << "        if (!pareto.empty()) {\n";
        temp_file << "            std::cout << \"\\nFirst 5 solutions:\" << std::endl;\n";
        temp_file << "            std::cout << \"  Obj1      Obj2\" << std::endl;\n";
        temp_file << "            std::cout << \"  ----------------------\" << std::endl;\n";
        temp_file << "            int count = std::min(5, static_cast<int>(pareto.size()));\n";
        temp_file << "            for (int i = 0; i < count; i++) {\n";
        temp_file << "                std::cout << \"  \" << std::fixed << std::setprecision(4) \n";
        temp_file << "                          << pareto[i].objectives[0] << \"  \" \n";
        temp_file << "                          << pareto[i].objectives[1] << std::endl;\n";
        temp_file << "            }\n";
        temp_file << "        }\n";
        temp_file << "        \n";
        temp_file << "        std::cout << \"\\nVisualize with:\" << std::endl;\n";
        
        if (config_name == "multi_objective_vrp_config") {
            temp_file << "        std::cout << \"  python3 visualize_vrp_solutions.py\" << std::endl;\n";
        } else {
            temp_file << "        std::cout << \"  make visualize-pareto\" << std::endl;\n";
        }
        
        temp_file << "        std::cout << \"Or manually:\" << std::endl;\n";
        temp_file << "        std::cout << \"  python3 -c \\\"import matplotlib.pyplot as plt; import numpy as np; \"\n";
        temp_file << "                  << \"data=np.loadtxt('" << config_name << "_pareto_front.txt'); \"\n";
        temp_file << "                  << \"plt.scatter(data[:,0],data[:,1]); \"\n";
        temp_file << "                  << \"plt.xlabel('Objective 1'); plt.ylabel('Objective 2'); \"\n";
        temp_file << "                  << \"plt.title('Pareto Front'); plt.savefig('pareto.png'); \"\n";
        temp_file << "                  << \"print('Saved to pareto.png')\\\"\" << std::endl;\n";
    } else {
        temp_file << "        // Single-objective: export best solution\n";
        temp_file << "        const auto& best = solver.get_best_individual();\n";
        temp_file << "        std::cout << \"\\n=== Best Solution ===\" << std::endl;\n";
        temp_file << "        std::cout << \"Fitness: \" << std::fixed << std::setprecision(6) << best.fitness << std::endl;\n";
        temp_file << "        \n";
        temp_file << "        // Export solution\n";
        temp_file << "        std::string solution_file = \"solutions/" << config_name << "_solution.txt\";\n";
        temp_file << "        solver.get_config()->export_solution(best, solution_file);\n";
        temp_file << "        std::cout << \"\\n✓ Solution exported to: \" << solution_file << std::endl;\n";
    }
    
    temp_file << "        \n";
    temp_file << "        std::cout << \"\\n✓ Optimization completed successfully!\" << std::endl;\n";
    temp_file << "        \n";
    temp_file << "    } catch (const std::exception& e) {\n";
    temp_file << "        std::cerr << \"Error: \" << e.what() << std::endl;\n";
    temp_file << "        return 1;\n";
    temp_file << "    }\n";
    temp_file << "    return 0;\n";
    temp_file << "}\n";
    
    temp_file.close();
    
    // Create solutions directory
    (void)system("mkdir -p solutions");
    
    // Compile
    std::string compile_cmd = "nvcc -std=c++17 -arch=sm_75 -O3 -Xcompiler -fopenmp -I. -Icore -Iconfigs -Iutils " + 
                             temp_main + " -o " + executable + " -lcurand -lcudart -lpthread 2>&1";
    
    std::cout << "\nCompiling unified BRKGA solver..." << std::endl;
    
    int result = system(compile_cmd.c_str());
    
    if (result != 0) {
        std::cerr << "Compilation failed!" << std::endl;
        return false;
    }
    
    std::cout << "✓ Compilation successful!" << std::endl;
    
    // Run
    std::cout << "\n==========================================" << std::endl;
    std::cout << "Running solver..." << std::endl;
    std::cout << "==========================================" << std::endl;
    std::string run_cmd = "./" + executable;
    result = system(run_cmd.c_str());
    
    // Clean up
    std::remove(temp_main.c_str());
    std::remove(executable.c_str());
    
    return result == 0;
}

int main(int argc, char* argv[]) {
    std::string data_file;
    std::string config_name;
    bool verbose = false;
    bool benchmark = false;
    int custom_pop = -1;
    int custom_gen = -1;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--data" && i + 1 < argc) {
            data_file = argv[++i];
        } else if (arg == "--config" && i + 1 < argc) {
            config_name = argv[++i];
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--benchmark") {
            benchmark = true;
        } else if (arg == "--pop" && i + 1 < argc) {
            custom_pop = std::stoi(argv[++i]);
        } else if (arg == "--gen" && i + 1 < argc) {
            custom_gen = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--create-samples") {
            // Create sample data files
            std::cout << "Creating sample data files..." << std::endl;
            (void)system("mkdir -p data");
            
            // Create simple TSP instance (will be created by the framework)
            std::cout << "✓ Data directory created" << std::endl;
            std::cout << "Note: Load actual TSPLIB instances or use the framework's generators" << std::endl;
            return 0;
        }
    }
    
    if (data_file.empty() || config_name.empty()) {
        std::cerr << "Error: Missing required arguments" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    std::cout << "=========================================" << std::endl;
    std::cout << "   Unified BRKGA Framework v2.5" << std::endl;
    std::cout << "   With Multi-GPU NSGA-II Support" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "Data file: " << data_file << std::endl;
    std::cout << "Config: " << config_name << std::endl;
    
    if (custom_pop > 0) {
        std::cout << "Population: " << custom_pop << " (custom)" << std::endl;
    }
    if (custom_gen > 0) {
        std::cout << "Generations: " << custom_gen << " (custom)" << std::endl;
    }
    
    std::cout << "Benchmark mode: " << (benchmark ? "enabled" : "disabled") << std::endl;
    std::cout << "Execution mode: auto-select" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    // Validate config name
    std::vector<std::string> valid_configs;
    valid_configs.push_back("tsp_config");
    valid_configs.push_back("tspj_config");
    valid_configs.push_back("knapsack_config");
    valid_configs.push_back("multi_tsp_config");
    valid_configs.push_back("multi_objective_vrp_config");
    
    bool valid = false;
    for (size_t i = 0; i < valid_configs.size(); i++) {
        if (config_name == valid_configs[i]) {
            valid = true;
            break;
        }
    }
    
    if (!valid) {
        std::cerr << "Error: Invalid config name: " << config_name << std::endl;
        std::cerr << "Valid options: ";
        for (size_t i = 0; i < valid_configs.size(); i++) {
            std::cerr << valid_configs[i];
            if (i < valid_configs.size() - 1) std::cerr << ", ";
        }
        std::cerr << std::endl;
        return 1;
    }
    
    // Run
    if (!compile_and_run(config_name, data_file, verbose, benchmark, custom_pop, custom_gen)) {
        std::cerr << "\n✗ Execution failed!" << std::endl;
        return 1;
    }
    
    std::cout << "\n=========================================" << std::endl;
    std::cout << "   Execution completed successfully!" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    // Show next steps
    if (is_multi_objective_config(config_name)) {
        std::cout << "\nNext steps:" << std::endl;
        std::cout << "  1. View Pareto front: cat " << config_name << "_pareto_front.txt" << std::endl;
        
        if (config_name == "multi_objective_vrp_config") {
            std::cout << "  2. View individual solutions: ls pareto_solutions/" << std::endl;
            std::cout << "  3. Visualize routes: python3 visualize_vrp_solutions.py" << std::endl;
            std::cout << "  4. Analyze quality: python3 analyze_vrp_pareto.py" << std::endl;
        } else {
            std::cout << "  2. Visualize: make visualize-pareto" << std::endl;
            std::cout << "  3. Analyze quality: make analyze-pareto" << std::endl;
        }
        
        if (!benchmark) {
            std::cout << "  " << (config_name == "multi_objective_vrp_config" ? "5" : "4") 
                     << ". Benchmark: Add --benchmark flag to compare GPU modes" << std::endl;
        }
    } else {
        std::cout << "\nNext steps:" << std::endl;
        std::cout << "  1. View solution: cat solutions/" << config_name << "_solution.txt" << std::endl;
        std::cout << "  2. Run with different parameters: --pop <size> --gen <count>" << std::endl;
    }
    
    return 0;
}