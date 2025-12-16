#ifndef GENETIC_OPERATORS_HPP
#define GENETIC_OPERATORS_HPP

#include "individual.hpp"
#include "config.hpp"
#include "cuda_kernels.cuh"
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

template<typename T>
class GeneticOperators {
private:
    BRKGAConfig<T>* config;
    std::mt19937 rng;
    
    // Device memory for CUDA operations
    T* d_population;
    T* d_elite_pop;
    T* d_non_elite_pop;
    T* d_offspring;
    T* d_mutants;
    curandState* d_states;
    
    bool device_memory_allocated;
    
public:
    GeneticOperators(BRKGAConfig<T>* cfg) 
        : config(cfg), device_memory_allocated(false),
          rng(std::chrono::steady_clock::now().time_since_epoch().count()) {
        allocate_device_memory();
    }
    
    ~GeneticOperators() {
        cleanup_device_memory();
    }
    
    void allocate_device_memory() {
        if (device_memory_allocated) return;
        
        int pop_size = config->population_size;
        int total_chrom_len = config->get_total_chromosome_length(); // FIXED: use correct method
        int elite_size = config->elite_size;
        int non_elite_size = pop_size - elite_size;
        int offspring_size = config->get_offspring_size();
        
        cudaMalloc(&d_population, pop_size * total_chrom_len * sizeof(T));
        cudaMalloc(&d_elite_pop, elite_size * total_chrom_len * sizeof(T));
        cudaMalloc(&d_non_elite_pop, non_elite_size * total_chrom_len * sizeof(T));
        cudaMalloc(&d_offspring, offspring_size * total_chrom_len * sizeof(T));
        cudaMalloc(&d_mutants, config->mutant_size * total_chrom_len * sizeof(T));
        cudaMalloc(&d_states, pop_size * sizeof(curandState));
        
        CudaUtils::check_cuda_error(cudaGetLastError(), "Device memory allocation");
        device_memory_allocated = true;
    }
    
    void cleanup_device_memory() {
        if (!device_memory_allocated) return;
        
        cudaFree(d_population);
        cudaFree(d_elite_pop);
        cudaFree(d_non_elite_pop);
        cudaFree(d_offspring);
        cudaFree(d_mutants);
        cudaFree(d_states);
        device_memory_allocated = false;
    }
    
    // FIXED: Helper methods for flattening/unflattening multi-component individuals
private:
    std::vector<T> flatten_population(const std::vector<Individual<T>>& population) {
        std::vector<T> flat;
        int total_len = config->get_total_chromosome_length();
        flat.reserve(population.size() * total_len);
        
        for (const auto& individual : population) {
            auto individual_flat = individual.flatten();
            flat.insert(flat.end(), individual_flat.begin(), individual_flat.end());
        }
        return flat;
    }
    
    void unflatten_to_population(const std::vector<T>& flat, std::vector<Individual<T>>& population) {
        int total_len = config->get_total_chromosome_length();
        
        for (size_t i = 0; i < population.size(); i++) {
            std::vector<T> individual_data(flat.begin() + i * total_len, 
                                         flat.begin() + (i + 1) * total_len);
            population[i].unflatten(individual_data);
        }
    }

public:
    // Host-based crossover operation
    void crossover_host(const std::vector<Individual<T>>& elite,
                       const std::vector<Individual<T>>& non_elite,
                       std::vector<Individual<T>>& offspring) {
        
        if (elite.empty() || non_elite.empty()) return;
        
        std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
        std::uniform_int_distribution<int> elite_dist(0, elite.size() - 1);
        std::uniform_int_distribution<int> non_elite_dist(0, non_elite.size() - 1);
        
        for (auto& child : offspring) {
            int elite_parent = elite_dist(rng);
            int non_elite_parent = non_elite_dist(rng);
            
            // Crossover each component separately
            for (int comp = 0; comp < config->num_components; comp++) {
                for (int i = 0; i < config->component_lengths[comp]; i++) {
                    if (prob_dist(rng) < config->elite_prob) {
                        child.get_component(comp)[i] = elite[elite_parent].get_component(comp)[i];
                    } else {
                        child.get_component(comp)[i] = non_elite[non_elite_parent].get_component(comp)[i];
                    }
                }
            }
            child.reset_evaluation();
        }
    }
    
    // Device-based crossover operation - FIXED for multi-component
    void crossover_device(const std::vector<Individual<T>>& elite,
                         const std::vector<Individual<T>>& non_elite,
                         std::vector<Individual<T>>& offspring) {
        
        if (elite.empty() || non_elite.empty() || offspring.empty()) return;
        
        int total_chrom_len = config->get_total_chromosome_length();
        int elite_size = elite.size();
        int non_elite_size = non_elite.size();
        int offspring_size = offspring.size();
        
        // Flatten and copy elite population to device
        std::vector<T> elite_flat = flatten_population(elite);
        cudaMemcpy(d_elite_pop, elite_flat.data(), elite_size * total_chrom_len * sizeof(T), cudaMemcpyHostToDevice);
        
        // Flatten and copy non-elite population to device
        std::vector<T> non_elite_flat = flatten_population(non_elite);
        cudaMemcpy(d_non_elite_pop, non_elite_flat.data(), non_elite_size * total_chrom_len * sizeof(T), cudaMemcpyHostToDevice);
        
        // Launch crossover kernel
        dim3 block(config->threads_per_block);
        dim3 grid = CudaUtils::calculate_grid_size(offspring_size, config->threads_per_block);
        
        crossover_kernel<<<grid, block>>>(
            d_elite_pop, d_non_elite_pop, d_offspring, d_states,
            offspring_size, total_chrom_len, config->elite_prob, elite_size, non_elite_size
        );
        
        CudaUtils::sync_and_check("Crossover kernel execution");
        
        // Copy offspring back to host and unflatten
        std::vector<T> offspring_flat(offspring_size * total_chrom_len);
        cudaMemcpy(offspring_flat.data(), d_offspring, offspring_size * total_chrom_len * sizeof(T), cudaMemcpyDeviceToHost);
        
        unflatten_to_population(offspring_flat, offspring);
    }
    
    // Host-based mutation operation
    void mutate_host(std::vector<Individual<T>>& mutants) {
        std::uniform_real_distribution<T> dist(0.0, 1.0);
        
        for (auto& mutant : mutants) {
            for (int comp = 0; comp < config->num_components; comp++) {
                for (int i = 0; i < config->component_lengths[comp]; i++) {
                    mutant.get_component(comp)[i] = dist(rng);
                }
            }
            mutant.reset_evaluation();
        }
    }
    
    // Device-based mutation operation - FIXED for multi-component
    void mutate_device(std::vector<Individual<T>>& mutants) {
        if (mutants.empty()) return;
        
        int total_chrom_len = config->get_total_chromosome_length();
        int mutant_count = mutants.size();
        
        // Launch mutation kernel
        dim3 block(config->threads_per_block);
        dim3 grid = CudaUtils::calculate_grid_size(mutant_count, config->threads_per_block);
        
        mutation_kernel<<<grid, block>>>(d_mutants, d_states, mutant_count, total_chrom_len);
        
        CudaUtils::sync_and_check("Mutation kernel execution");
        
        // Copy mutants back to host and unflatten
        std::vector<T> mutants_flat(mutant_count * total_chrom_len);
        cudaMemcpy(mutants_flat.data(), d_mutants, mutant_count * total_chrom_len * sizeof(T), cudaMemcpyDeviceToHost);
        
        unflatten_to_population(mutants_flat, mutants);
    }
    
    // Initialize device random states
    void initialize_device_random_states() {
        dim3 block(config->threads_per_block);
        dim3 grid = CudaUtils::calculate_grid_size(config->population_size, config->threads_per_block);
        
        int total_chrom_len = config->get_total_chromosome_length();
        
        initialize_population_kernel<<<grid, block>>>(
            d_population, d_states, config->population_size, total_chrom_len
        );
        
        CudaUtils::sync_and_check("Random state initialization");
    }
    
    // Selection operation (tournament selection)
    Individual<T> tournament_selection(const std::vector<Individual<T>>& population, int tournament_size = 3) {
        if (population.empty()) {
            throw std::runtime_error("Cannot select from empty population");
        }
        
        std::uniform_int_distribution<int> dist(0, population.size() - 1);
        Individual<T> best = population[dist(rng)];
        
        for (int i = 1; i < tournament_size; i++) {
            Individual<T> candidate = population[dist(rng)];
            if (config->comparator) {
                if (config->comparator(candidate.fitness, best.fitness)) {
                    best = candidate;
                }
            } else {
                if (candidate.fitness < best.fitness) {
                    best = candidate;
                }
            }
        }
        
        return best;
    }
    
    // Roulette wheel selection
    Individual<T> roulette_selection(const std::vector<Individual<T>>& population) {
        if (population.empty()) {
            throw std::runtime_error("Cannot select from empty population");
        }
        
        // Calculate total fitness (for maximization problems)
        T total_fitness = T(0);
        T min_fitness = population[0].fitness;
        
        // Find minimum fitness for offset in case of negative values
        for (const auto& individual : population) {
            min_fitness = std::min(min_fitness, individual.fitness);
        }
        
        // Offset fitness values to make them positive
        T offset = (min_fitness < T(0)) ? -min_fitness + T(1) : T(0);
        
        for (const auto& individual : population) {
            total_fitness += individual.fitness + offset;
        }
        
        if (total_fitness <= T(0)) {
            // Fallback to uniform selection
            std::uniform_int_distribution<int> dist(0, population.size() - 1);
            return population[dist(rng)];
        }
        
        // Generate random number
        std::uniform_real_distribution<T> dist(T(0), total_fitness);
        T random_value = dist(rng);
        
        // Select individual based on fitness proportion
        T cumulative_fitness = T(0);
        for (const auto& individual : population) {
            cumulative_fitness += individual.fitness + offset;
            if (cumulative_fitness >= random_value) {
                return individual;
            }
        }
        
        // Fallback (should not reach here)
        return population.back();
    }
    
    // Uniform crossover (alternative to biased crossover)
    void uniform_crossover(const Individual<T>& parent1, const Individual<T>& parent2,
                          Individual<T>& child, double crossover_rate = 0.5) {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        
        for (int comp = 0; comp < config->num_components; comp++) {
            for (int i = 0; i < config->component_lengths[comp]; i++) {
                if (dist(rng) < crossover_rate) {
                    child.get_component(comp)[i] = parent1.get_component(comp)[i];
                } else {
                    child.get_component(comp)[i] = parent2.get_component(comp)[i];
                }
            }
        }
        child.reset_evaluation();
    }
    
    // Single-point crossover (for single component problems)
    void single_point_crossover(const Individual<T>& parent1, const Individual<T>& parent2,
                               Individual<T>& child) {
        if (config->num_components > 1) {
            // For multi-component, use uniform crossover instead
            uniform_crossover(parent1, parent2, child);
            return;
        }
        
        std::uniform_int_distribution<int> dist(1, config->component_lengths[0] - 1);
        int crossover_point = dist(rng);
        
        const auto& chrom1 = parent1.get_chromosome();
        const auto& chrom2 = parent2.get_chromosome();
        auto& child_chrom = child.get_chromosome();
        
        for (int i = 0; i < crossover_point; i++) {
            child_chrom[i] = chrom1[i];
        }
        for (int i = crossover_point; i < config->component_lengths[0]; i++) {
            child_chrom[i] = chrom2[i];
        }
        child.reset_evaluation();
    }
    
    // Two-point crossover (for single component problems)
    void two_point_crossover(const Individual<T>& parent1, const Individual<T>& parent2,
                            Individual<T>& child) {
        if (config->num_components > 1) {
            // For multi-component, use uniform crossover instead
            uniform_crossover(parent1, parent2, child);
            return;
        }
        
        std::uniform_int_distribution<int> dist(0, config->component_lengths[0] - 1);
        int point1 = dist(rng);
        int point2 = dist(rng);
        
        if (point1 > point2) {
            std::swap(point1, point2);
        }
        
        const auto& chrom1 = parent1.get_chromosome();
        const auto& chrom2 = parent2.get_chromosome();
        auto& child_chrom = child.get_chromosome();
        
        for (int i = 0; i < point1; i++) {
            child_chrom[i] = chrom1[i];
        }
        for (int i = point1; i <= point2; i++) {
            child_chrom[i] = chrom2[i];
        }
        for (int i = point2 + 1; i < config->component_lengths[0]; i++) {
            child_chrom[i] = chrom1[i];
        }
        child.reset_evaluation();
    }
    
    // Gaussian mutation (for real-valued genes)
    void gaussian_mutation(Individual<T>& individual, double mutation_rate = 0.1, double sigma = 0.1) {
        std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
        std::normal_distribution<T> normal_dist(T(0), sigma);
        
        for (int comp = 0; comp < config->num_components; comp++) {
            for (int i = 0; i < config->component_lengths[comp]; i++) {
                if (uniform_dist(rng) < mutation_rate) {
                    individual.get_component(comp)[i] += normal_dist(rng);
                    // Clamp to [0, 1] range for BRKGA
                    individual.get_component(comp)[i] = std::max(T(0), std::min(T(1), individual.get_component(comp)[i]));
                }
            }
        }
        individual.reset_evaluation();
    }
    
    // Utility method to check if device operations are available
    bool is_device_available() const {
        return device_memory_allocated;
    }
    
    // Method to choose between host and device operations based on problem size
    void adaptive_crossover(const std::vector<Individual<T>>& elite,
                           const std::vector<Individual<T>>& non_elite,
                           std::vector<Individual<T>>& offspring) {
        // Use device for larger populations, host for smaller ones
        const int threshold = 500; // Configurable threshold
        
        if (config->population_size >= threshold && is_device_available()) {
            crossover_device(elite, non_elite, offspring);
        } else {
            crossover_host(elite, non_elite, offspring);
        }
    }
    
    void adaptive_mutation(std::vector<Individual<T>>& mutants) {
        const int threshold = 500; // Configurable threshold
        
        if (config->population_size >= threshold && is_device_available()) {
            mutate_device(mutants);
        } else {
            mutate_host(mutants);
        }
    }
};

#endif // GENETIC_OPERATORS_HPP