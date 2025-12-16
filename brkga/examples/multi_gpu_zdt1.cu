// examples/multi_gpu_zdt1.cu - CORRECTED with diversity checks
#include "../core/solver.hpp"
#include "../core/config.hpp"
#include <iostream>
#include <cmath>
#include <memory>
#include <set>

template<typename T>
class ZDT1MultiGPUConfig : public BRKGAConfig<T> {
private:
    int num_variables;
    
public:
    ZDT1MultiGPUConfig(int n_vars = 30, int pop_size = 1000) 
        : BRKGAConfig<T>({n_vars}, 2),
          num_variables(n_vars) {
        
        this->population_size = pop_size;
        this->elite_size = 0;
        this->mutant_size = 0;
        this->max_generations = 500;
        this->elite_prob = 0.7;  // Crossover probability
        
        this->objective_functions[0] = [this](const Individual<T>& ind) {
            return objective_f1(ind);
        };
        
        this->objective_functions[1] = [this](const Individual<T>& ind) {
            return objective_f2(ind);
        };
        
        this->update_cuda_grid_size();
    }
    
    T objective_f1(const Individual<T>& individual) const {
        return individual.get_chromosome()[0];
    }
    
    T objective_f2(const Individual<T>& individual) const {
        const auto& x = individual.get_chromosome();
        
        T sum = 0;
        for (int i = 1; i < num_variables; i++) {
            sum += x[i];
        }
        T g = 1.0 + 9.0 * sum / (num_variables - 1);
        
        T f1 = x[0];
        T f2 = g * (1.0 - std::sqrt(f1 / g));
        
        return f2;
    }
    
    void print_solution(const Individual<T>& individual) override {
        std::cout << "\n=== Solution ===" << std::endl;
        std::cout << "Objective 1 (f1): " << std::fixed << std::setprecision(6) 
                  << individual.objectives[0] << std::endl;
        std::cout << "Objective 2 (f2): " << std::setprecision(6) 
                  << individual.objectives[1] << std::endl;
        std::cout << "Rank: " << individual.rank << std::endl;
        std::cout << "Crowding distance: " << std::setprecision(4) 
                  << individual.crowding_distance << std::endl;
        std::cout << "================" << std::endl;
    }
    
    bool validate_solution(const Individual<T>& individual) override {
        const auto& x = individual.get_chromosome();
        for (const auto& val : x) {
            if (val < 0.0 || val > 1.0) return false;
        }
        return true;
    }
    
    void print_instance_info() const {
        std::cout << "=== ZDT1 Multi-GPU Benchmark ===" << std::endl;
        std::cout << "Variables: " << num_variables << std::endl;
        std::cout << "Objectives: 2 (minimize)" << std::endl;
        std::cout << "Population: " << this->population_size << std::endl;
        std::cout << "True Pareto front: f2 = 1 - sqrt(f1)" << std::endl;
        std::cout << "=================================" << std::endl;
    }
    
    T calculate_hypervolume(const std::vector<Individual<T>>& pareto_front, 
                           T ref_point_x = 1.0, T ref_point_y = 1.0) const {
        if (pareto_front.empty()) return T(0);
        
        std::vector<std::pair<T, T>> points;
        for (const auto& ind : pareto_front) {
            if (ind.objectives[0] <= ref_point_x && ind.objectives[1] <= ref_point_y) {
                points.emplace_back(ind.objectives[0], ind.objectives[1]);
            }
        }
        
        if (points.empty()) return T(0);
        
        std::sort(points.begin(), points.end());
        
        T volume = 0;
        T prev_x = 0;
        
        for (const auto& point : points) {
            volume += (point.first - prev_x) * (ref_point_y - point.second);
            prev_x = point.first;
        }
        
        return volume;
    }
};

int main(int argc, char* argv[]) {
    try {
        std::cout << "===============================================" << std::endl;
        std::cout << "  Multi-GPU NSGA-II on ZDT1 Benchmark" << std::endl;
        std::cout << "  With Diversity Preservation" << std::endl;
        std::cout << "===============================================" << std::endl;
        
        int pop_size = 1000;
        int num_vars = 30;
        int max_gen = 500;
        bool benchmark_mode = false;
        
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--pop" && i + 1 < argc) {
                pop_size = std::stoi(argv[++i]);
            } else if (arg == "--vars" && i + 1 < argc) {
                num_vars = std::stoi(argv[++i]);
            } else if (arg == "--gen" && i + 1 < argc) {
                max_gen = std::stoi(argv[++i]);
            } else if (arg == "--benchmark") {
                benchmark_mode = true;
            } else if (arg == "--help") {
                std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
                std::cout << "Options:" << std::endl;
                std::cout << "  --pop <size>     Population size (default: 1000)" << std::endl;
                std::cout << "  --vars <n>       Number of variables (default: 30)" << std::endl;
                std::cout << "  --gen <n>        Max generations (default: 500)" << std::endl;
                std::cout << "  --benchmark      Run performance benchmark" << std::endl;
                std::cout << "  --help           Show this help message" << std::endl;
                return 0;
            }
        }
        
        auto config = std::make_unique<ZDT1MultiGPUConfig<float>>(num_vars, pop_size);
        config->max_generations = max_gen;
        config->print_instance_info();
        config->print_config();
        
        Solver<float> solver(std::move(config), true, 50);
        
        if (benchmark_mode) {
            std::cout << "\n=== Running Benchmark Mode ===" << std::endl;
            solver.benchmark_modes();
            std::cout << "================================" << std::endl;
        }
        
        std::cout << "\n=== Starting Multi-GPU Optimization ===" << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        solver.run();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();
        
        std::cout << "\n=== Results ===" << std::endl;
        std::cout << "Total execution time: " << (duration / 1000.0) << " seconds" << std::endl;
        
        solver.export_pareto_front("zdt1_multi_gpu_pareto.txt");
        
        auto pareto = solver.get_pareto_front();
        std::cout << "\nPareto front size: " << pareto.size() << std::endl;
        
        if (!pareto.empty()) {
            // Count UNIQUE solutions
            std::set<std::pair<float, float>> unique_solutions;
            for (const auto& ind : pareto) {
                unique_solutions.insert({ind.objectives[0], ind.objectives[1]});
            }
            
            std::cout << "Unique solutions: " << unique_solutions.size() << std::endl;
            
            double diversity_ratio = (double)unique_solutions.size() / pareto.size();
            std::cout << "Diversity ratio: " << std::fixed << std::setprecision(2) 
                      << diversity_ratio * 100 << "%" << std::endl;
            
            if (diversity_ratio < 0.70) {
                std::cout << "\n❌ WARNING: Low diversity detected!" << std::endl;
                std::cout << "   Consider: Increasing mutation rate, population size, or generations" << std::endl;
            } else if (diversity_ratio > 0.95) {
                std::cout << "\n✓ Excellent diversity maintained!" << std::endl;
            } else {
                std::cout << "\n✓ Good diversity maintained" << std::endl;
            }
            
            // Calculate hypervolume
            ZDT1MultiGPUConfig<float> temp_config(num_vars, pop_size);
            float hv = temp_config.calculate_hypervolume(pareto, 1.0, 1.0);
            std::cout << "\nHypervolume: " << std::fixed << std::setprecision(6) << hv << std::endl;
            
            // Show some solutions
            std::cout << "\nFirst 10 Pareto solutions:" << std::endl;
            std::cout << "     f1        f2      Error" << std::endl;
            std::cout << "----------------------------------------" << std::endl;
            
            int count = std::min(10, static_cast<int>(pareto.size()));
            for (int i = 0; i < count; i++) {
                float f1 = pareto[i].objectives[0];
                float f2 = pareto[i].objectives[1];
                float true_f2 = 1.0 - std::sqrt(f1);
                float error = std::abs(f2 - true_f2);
                
                std::cout << std::setw(10) << std::setprecision(6) << f1 
                          << "  " << std::setw(10) << f2 
                          << "  " << std::setw(10) << std::setprecision(8) << error << std::endl;
            }
            
            // Calculate error from true Pareto front
            float total_error = 0.0;
            for (const auto& ind : pareto) {
                float f1 = ind.objectives[0];
                float f2 = ind.objectives[1];
                float true_f2 = 1.0 - std::sqrt(f1);
                float error = std::abs(f2 - true_f2);
                total_error += error;
            }
            float avg_error = total_error / pareto.size();
            
            std::cout << "\nAverage error from true Pareto front: " 
                      << std::setprecision(8) << avg_error << std::endl;
            
            // Calculate coverage of Pareto front
            float min_f1 = pareto[0].objectives[0];
            float max_f1 = pareto[0].objectives[0];
            for (const auto& ind : pareto) {
                min_f1 = std::min(min_f1, ind.objectives[0]);
                max_f1 = std::max(max_f1, ind.objectives[0]);
            }
            
            std::cout << "Pareto front coverage:" << std::endl;
            std::cout << "  f1 range: [" << std::setprecision(4) << min_f1 
                      << ", " << max_f1 << "]" << std::endl;
            std::cout << "  Coverage: " << std::setprecision(1) 
                      << (max_f1 - min_f1) * 100 << "% of [0,1]" << std::endl;
        }
        
        // Visualization instructions
        std::cout << "\n=== Visualization ===" << std::endl;
        std::cout << "Plot Pareto front with:" << std::endl;
        std::cout << "\n  python3 << 'EOF'" << std::endl;
        std::cout << "import matplotlib.pyplot as plt" << std::endl;
        std::cout << "import numpy as np" << std::endl;
        std::cout << "" << std::endl;
        std::cout << "# Load data" << std::endl;
        std::cout << "data = np.loadtxt('zdt1_multi_gpu_pareto.txt')" << std::endl;
        std::cout << "" << std::endl;
        std::cout << "# Create figure" << std::endl;
        std::cout << "plt.figure(figsize=(10, 6))" << std::endl;
        std::cout << "" << std::endl;
        std::cout << "# Plot found solutions" << std::endl;
        std::cout << "plt.scatter(data[:,0], data[:,1], alpha=0.6, s=30, " << std::endl;
        std::cout << "           c='blue', label='Found solutions')" << std::endl;
        std::cout << "" << std::endl;
        std::cout << "# Plot true Pareto front" << std::endl;
        std::cout << "x_true = np.linspace(0, 1, 200)" << std::endl;
        std::cout << "y_true = 1 - np.sqrt(x_true)" << std::endl;
        std::cout << "plt.plot(x_true, y_true, 'r--', linewidth=2, " << std::endl;
        std::cout << "        label='True Pareto front')" << std::endl;
        std::cout << "" << std::endl;
        std::cout << "# Formatting" << std::endl;
        std::cout << "plt.xlabel('Objective 1 (f1)', fontsize=12)" << std::endl;
        std::cout << "plt.ylabel('Objective 2 (f2)', fontsize=12)" << std::endl;
        std::cout << "plt.title('Multi-GPU NSGA-II: ZDT1 Results', fontsize=14)" << std::endl;
        std::cout << "plt.legend(fontsize=10)" << std::endl;
        std::cout << "plt.grid(True, alpha=0.3)" << std::endl;
        std::cout << "plt.tight_layout()" << std::endl;
        std::cout << "plt.savefig('zdt1_multi_gpu_result.png', dpi=300)" << std::endl;
        std::cout << "print('✓ Saved to: zdt1_multi_gpu_result.png')" << std::endl;
        std::cout << "EOF" << std::endl;
        
        std::cout << "\n✓ Multi-GPU optimization completed successfully!" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}