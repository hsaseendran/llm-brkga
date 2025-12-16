// examples/zdt1.cu - Simple multi-objective optimization example
#include "../core/solver.hpp"
#include "../core/config.hpp"
#include <iostream>
#include <cmath>
#include <memory>

// ZDT1 benchmark problem configuration
// Two objectives to minimize:
// f1(x) = x1
// f2(x) = g(x) * [1 - sqrt(x1/g(x))]
// where g(x) = 1 + 9 * sum(x2...xn) / (n-1)
// True Pareto front: f2 = 1 - sqrt(f1), f1 in [0,1]

template<typename T>
class ZDT1Config : public BRKGAConfig<T> {
private:
    int num_variables;
    
public:
    ZDT1Config(int n_vars = 30) 
        : BRKGAConfig<T>({n_vars}, 2),  // 2 objectives
          num_variables(n_vars) {
        
        // Configure for multi-objective
        this->population_size = 100;
        this->elite_size = 0;      // Not used in NSGA-II
        this->mutant_size = 0;     // Not used in NSGA-II
        this->max_generations = 5;
        this->elite_prob = 0.5;    // Crossover probability
        
        // Define objective functions
        this->objective_functions[0] = [this](const Individual<T>& ind) {
            return objective_f1(ind);
        };
        
        this->objective_functions[1] = [this](const Individual<T>& ind) {
            return objective_f2(ind);
        };
        
        this->update_cuda_grid_size();
    }
    
    T objective_f1(const Individual<T>& individual) const {
        // f1(x) = x1
        return individual.get_chromosome()[0];
    }
    
    T objective_f2(const Individual<T>& individual) const {
        const auto& x = individual.get_chromosome();
        
        // Calculate g(x) = 1 + 9 * sum(x2...xn) / (n-1)
        T sum = 0;
        for (int i = 1; i < num_variables; i++) {
            sum += x[i];
        }
        T g = 1.0 + 9.0 * sum / (num_variables - 1);
        
        // Calculate f2(x) = g(x) * [1 - sqrt(x1/g(x))]
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
        std::cout << "================" << std::endl;
    }
    
    bool validate_solution(const Individual<T>& individual) override {
        // Check if all variables are in [0, 1]
        const auto& x = individual.get_chromosome();
        for (const auto& val : x) {
            if (val < 0.0 || val > 1.0) return false;
        }
        return true;
    }
    
    void print_instance_info() const {
        std::cout << "=== ZDT1 Benchmark Problem ===" << std::endl;
        std::cout << "Variables: " << num_variables << std::endl;
        std::cout << "Objectives: 2 (minimize)" << std::endl;
        std::cout << "True Pareto front: f2 = 1 - sqrt(f1)" << std::endl;
        std::cout << "===============================" << std::endl;
    }
};

int main() {
    try {
        std::cout << "Multi-Objective Optimization with NSGA-II" << std::endl;
        std::cout << "==========================================" << std::endl;
        
        // Create ZDT1 configuration
        auto config = std::make_unique<ZDT1Config<float>>(30);
        config->print_instance_info();
        config->print_config();
        
        // Create solver
        Solver<float> solver(std::move(config), true, 50);
        
        // Run optimization
        solver.run();
        
        // Export Pareto front
        solver.export_pareto_front("zdt1_pareto_front.txt");
        
        // Print some solutions from Pareto front
        auto pareto = solver.get_pareto_front();
        std::cout << "\n=== Pareto Front Solutions ===" << std::endl;
        std::cout << "Total solutions: " << pareto.size() << std::endl;
        
        if (!pareto.empty()) {
            std::cout << "\nFirst 5 solutions:" << std::endl;
            std::cout << "f1\t\tf2" << std::endl;
            std::cout << "------------------------" << std::endl;
            
            int count = std::min(5, static_cast<int>(pareto.size()));
            for (int i = 0; i < count; i++) {
                std::cout << std::fixed << std::setprecision(6) 
                          << pareto[i].objectives[0] << "\t" 
                          << pareto[i].objectives[1] << std::endl;
            }
        }
        
        std::cout << "\n✓ Optimization completed successfully!" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}