// core/config.hpp - EXTENDED for multi-objective support
#ifndef CONFIG_HPP
#define CONFIG_HPP

#include "individual.hpp"
#include <functional>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <memory>
#include <vector>

template<typename T>
class BRKGAConfig {
public:
    // Problem dimensions
    std::vector<int> component_lengths;
    int num_components;
    
    // GA parameters
    int population_size;
    int elite_size;
    int mutant_size;
    double elite_prob;
    int max_generations;
    
    // CUDA parameters
    int threads_per_block;
    int blocks_per_grid;
    
    // Single-objective support (backward compatible)
    std::function<T(const Individual<T>&)> fitness_function;
    std::function<std::vector<std::vector<T>>(const Individual<T>&)> decoder;
    std::function<bool(T, T)> comparator;
    
    // Multi-objective support (NEW)
    int num_objectives;
    std::vector<std::function<T(const Individual<T>&)>> objective_functions;
    
    // Device function pointers
    void* d_fitness_func;
    void* d_decoder_func;
    
    // GPU evaluation interface
    virtual bool has_gpu_evaluation() const {
        return false;
    }

    virtual void evaluate_population_gpu(T* d_population, T* d_fitness,
                                        int pop_size, int chrom_len) {
        // Override in subclasses for GPU support
    }

    // Local search interface (problem-specific, override in subclasses)
    virtual bool has_local_search() const {
        return false;  // Default: no local search
    }

    // Apply local search to elite individuals
    // Returns number of improvements found
    virtual int apply_local_search_gpu(
        T* d_population,           // Population on GPU
        T* d_backup,               // Backup buffer for restoration
        T* d_fitness,              // Fitness values
        T* d_fitness_backup,       // Backup fitness for comparison
        void* d_rng_states,        // curandState* for random numbers
        int pop_size,              // Population size
        int chrom_len,             // Chromosome length
        int num_to_improve,        // Number of individuals to apply LS to
        int num_moves              // Number of moves per individual
    ) {
        // Default: do nothing
        return 0;
    }
    
    // Single component constructor (backward compatibility)
    BRKGAConfig(int chrom_len) 
        : component_lengths({chrom_len}), num_components(1),
          population_size(100), elite_size(20), mutant_size(10),
          elite_prob(0.7), max_generations(1000),
          threads_per_block(256), blocks_per_grid(32),
          num_objectives(1),  // Default: single objective
          d_fitness_func(nullptr), d_decoder_func(nullptr) {}
    
    // Multi-component constructor
    BRKGAConfig(const std::vector<int>& comp_lengths) 
        : component_lengths(comp_lengths), num_components(comp_lengths.size()),
          population_size(100), elite_size(20), mutant_size(10),
          elite_prob(0.7), max_generations(1000),
          threads_per_block(256), blocks_per_grid(32),
          num_objectives(1),  // Default: single objective
          d_fitness_func(nullptr), d_decoder_func(nullptr) {}
    
    // Multi-objective constructor (NEW)
    BRKGAConfig(const std::vector<int>& comp_lengths, int n_objectives)
        : component_lengths(comp_lengths), num_components(comp_lengths.size()),
          population_size(100), elite_size(20), mutant_size(10),
          elite_prob(0.7), max_generations(1000),
          threads_per_block(256), blocks_per_grid(32),
          num_objectives(n_objectives),
          d_fitness_func(nullptr), d_decoder_func(nullptr) {
        objective_functions.resize(n_objectives);
    }
    
    virtual ~BRKGAConfig() = default;
    
    // Device memory management
    virtual void initialize_device_functions() {}
    virtual void cleanup_device_functions() {}
    
    // Problem-specific behavior (to be overridden)
    virtual void print_solution(const Individual<T>& individual) {
        if (num_objectives > 1) {
            std::cout << "Objectives: ";
            for (size_t i = 0; i < individual.objectives.size(); i++) {
                std::cout << individual.objectives[i];
                if (i < individual.objectives.size() - 1) std::cout << ", ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "Best fitness: " << individual.fitness << std::endl;
        }
    }
    
    virtual bool validate_solution(const Individual<T>& individual) { 
        return true; 
    }
    
    virtual void export_solution(const Individual<T>& individual, const std::string& filename) {
        std::ofstream file(filename);
        
        if (num_objectives > 1) {
            file << "Objectives: ";
            for (size_t i = 0; i < individual.objectives.size(); i++) {
                file << individual.objectives[i];
                if (i < individual.objectives.size() - 1) file << " ";
            }
            file << std::endl;
        } else {
            file << "Fitness: " << individual.fitness << std::endl;
        }
        
        file << "Chromosome: ";
        for (int comp = 0; comp < num_components; comp++) {
            const auto& component = individual.get_component(comp);
            file << "Component " << comp << ": ";
            for (size_t i = 0; i < component.size(); i++) {
                file << component[i];
                if (i < component.size() - 1) file << " ";
            }
            if (comp < num_components - 1) file << " | ";
        }
        file << std::endl;
        file.close();
    }
    
    // Configuration validation
    virtual bool is_valid() const {
        bool basic_valid = !component_lengths.empty() && 
                          std::all_of(component_lengths.begin(), component_lengths.end(), 
                                     [](int len) { return len > 0; }) &&
                          population_size > 0 && 
                          elite_prob >= 0.0 && elite_prob <= 1.0 &&
                          max_generations > 0 &&
                          num_objectives > 0;
        
        if (!basic_valid) return false;
        
        // For single-objective (BRKGA), check elite/mutant constraints
        if (num_objectives == 1) {
            return elite_size > 0 && 
                   mutant_size >= 0 &&
                   elite_size + mutant_size <= population_size;
        }
        
        // For multi-objective (NSGA-II), elite/mutant sizes don't matter
        return true;
    }
    
    // Helper methods
    int get_total_chromosome_length() const {
        int total = 0;
        for (int len : component_lengths) {
            total += len;
        }
        return total;
    }
    
    int chromosome_length() const {
        return get_total_chromosome_length();
    }
    
    int get_offspring_size() const {
        return population_size - elite_size - mutant_size;
    }
    
    void update_cuda_grid_size() {
        blocks_per_grid = (population_size + threads_per_block - 1) / threads_per_block;
    }
    
    bool is_multi_objective() const {
        return num_objectives > 1;
    }
    
    void print_config() const {
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Mode: " << (is_multi_objective() ? "Multi-objective" : "Single-objective") << std::endl;
        if (is_multi_objective()) {
            std::cout << "  Objectives: " << num_objectives << std::endl;
        }
        std::cout << "  Components: " << num_components << std::endl;
        std::cout << "  Component lengths: ";
        for (size_t i = 0; i < component_lengths.size(); i++) {
            std::cout << component_lengths[i];
            if (i < component_lengths.size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;
        std::cout << "  Total chromosome length: " << get_total_chromosome_length() << std::endl;
        std::cout << "  Population size: " << population_size << std::endl;
        std::cout << "  Elite size: " << elite_size << std::endl;
        std::cout << "  Mutant size: " << mutant_size << std::endl;
        std::cout << "  Offspring size: " << get_offspring_size() << std::endl;
        std::cout << "  Elite probability: " << elite_prob << std::endl;
        std::cout << "  Max generations: " << max_generations << std::endl;
        std::cout << "  CUDA threads per block: " << threads_per_block << std::endl;
        std::cout << "  CUDA blocks per grid: " << blocks_per_grid << std::endl;
    }
};

#endif // CONFIG_HPP