// core/population.hpp - CORRECTED with diversity preservation
#ifndef POPULATION_HPP
#define POPULATION_HPP

#include "individual.hpp"
#include "config.hpp"
#include <vector>
#include <algorithm>
#include <random>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <limits>
#include <future>
#include <thread>
#include <mutex>
#include <set>
#include <cmath>
#include <cuda_runtime.h>

template<typename T>
class Population {
private:
    std::vector<Individual<T>> individuals;
    std::vector<Individual<T>> next_generation;
    Individual<T> best_individual;
    std::vector<T> fitness_history;
    BRKGAConfig<T>* config;
    std::mt19937 rng;
    
    T* d_population;
    T* d_fitness;
    bool gpu_memory_allocated;
    int total_chrom_len;
    bool use_gpu_evaluation;
    
    std::vector<std::vector<int>> fronts;
    bool is_multi_objective;
    bool gpu_resident_active;  // True when population data is on GPU (CPU data is stale)
    
    struct GPUWorkspace {
        int device_id;
        T* d_objectives;
        int* d_ranks;
        T* d_crowding_dist;
        bool allocated;
        
        GPUWorkspace(int id) : device_id(id), allocated(false) {
            d_objectives = nullptr;
            d_ranks = nullptr;
            d_crowding_dist = nullptr;
        }
        
        ~GPUWorkspace() { cleanup(); }
        
        void cleanup() {
            if (allocated) {
                cudaSetDevice(device_id);
                if (d_objectives) cudaFree(d_objectives);
                if (d_ranks) cudaFree(d_ranks);
                if (d_crowding_dist) cudaFree(d_crowding_dist);
                allocated = false;
            }
        }
    };
    
    std::vector<std::unique_ptr<GPUWorkspace>> gpu_workspaces;
    bool multi_gpu_enabled;
    int num_active_gpus;
    
public:
    Population(BRKGAConfig<T>* cfg)
        : config(cfg), best_individual(cfg->component_lengths),
          rng(std::chrono::steady_clock::now().time_since_epoch().count()),
          d_population(nullptr), d_fitness(nullptr), gpu_memory_allocated(false),
          total_chrom_len(cfg->get_total_chromosome_length()),
          use_gpu_evaluation(false),
          is_multi_objective(cfg->num_objectives > 1),
          gpu_resident_active(false),
          multi_gpu_enabled(false), num_active_gpus(0) {
        
        if (is_multi_objective) {
            individuals.resize(config->population_size, 
                             Individual<T>(cfg->component_lengths, cfg->num_objectives));
            next_generation.resize(config->population_size, 
                                  Individual<T>(cfg->component_lengths, cfg->num_objectives));
            best_individual = Individual<T>(cfg->component_lengths, cfg->num_objectives);
        } else {
            individuals.resize(config->population_size, Individual<T>(cfg->component_lengths));
            next_generation.resize(config->population_size, Individual<T>(cfg->component_lengths));
        }
        
        if (config->has_gpu_evaluation()) {
            setup_gpu_evaluation();
        }
        
        if (is_multi_objective && config->population_size >= 500) {
            setup_multi_gpu_nsga2();
        }
    }
    
    ~Population() {
        cleanup_gpu_memory();
        cleanup_multi_gpu();
    }
    
    void initialize() {
        for (auto& individual : individuals) {
            individual.randomize(rng);
        }
        
        evaluate_all();
        
        if (is_multi_objective) {
            fast_non_dominated_sort();
            calculate_crowding_distance();
            update_best_from_front();
        } else {
            update_best();
        }
        
        fitness_history.clear();
        fitness_history.push_back(best_individual.fitness);
    }
    
    // ============================================================
    // CRITICAL FIX: DIVERSITY PRESERVATION
    // ============================================================
    
    void ensure_minimum_diversity() {
        if (!is_multi_objective) return;
        
        const double MIN_DISTANCE = 0.001;  // Minimum Euclidean distance in objective space
        std::vector<Individual<T>> unique_solutions;
        
        for (const auto& candidate : individuals) {
            bool is_unique = true;
            
            for (const auto& existing : unique_solutions) {
                double distance = 0.0;
                for (int obj = 0; obj < config->num_objectives; obj++) {
                    double diff = candidate.objectives[obj] - existing.objectives[obj];
                    distance += diff * diff;
                }
                distance = std::sqrt(distance);
                
                if (distance < MIN_DISTANCE) {
                    is_unique = false;
                    break;
                }
            }
            
            if (is_unique) {
                unique_solutions.push_back(candidate);
            }
        }
        
        // Check diversity
        double diversity_ratio = (double)unique_solutions.size() / individuals.size();
        
        if (diversity_ratio < 0.70) {  // If less than 70% unique
            std::cout << "⚠ Diversity loss detected: " << unique_solutions.size() 
                      << "/" << individuals.size() << " unique (" 
                      << std::fixed << std::setprecision(1) << diversity_ratio * 100 << "%)" << std::endl;
            std::cout << "  Injecting diversity..." << std::endl;
            
            individuals = unique_solutions;
            
            // Add random individuals to restore population size
            while (individuals.size() < static_cast<size_t>(config->population_size)) {
                Individual<T> new_ind(config->component_lengths, config->num_objectives);
                new_ind.randomize(rng);
                individuals.push_back(new_ind);
            }
            
            evaluate_all();
            fast_non_dominated_sort();
            calculate_crowding_distance();
            
            std::cout << "  ✓ Diversity restored" << std::endl;
        }
    }
    
    // ============================================================
    // MULTI-GPU NSGA-II METHODS
    // ============================================================
    
    void setup_multi_gpu_nsga2() {
        int device_count;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        
        if (error != cudaSuccess || device_count < 2) {
            return;
        }
        
        std::cout << "Setting up multi-GPU NSGA-II with " << device_count << " GPUs" << std::endl;
        
        for (int i = 0; i < device_count; i++) {
            auto workspace = std::make_unique<GPUWorkspace>(i);
            allocate_multi_gpu_workspace(workspace.get());
            if (workspace->allocated) {
                gpu_workspaces.push_back(std::move(workspace));
                num_active_gpus++;
            }
        }
        
        multi_gpu_enabled = (num_active_gpus > 1);
        
        if (multi_gpu_enabled) {
            enable_peer_access();
            std::cout << "Multi-GPU NSGA-II enabled with " << num_active_gpus << " GPUs" << std::endl;
        }
    }
    
    void allocate_multi_gpu_workspace(GPUWorkspace* workspace) {
        cudaSetDevice(workspace->device_id);
        
        int pop_size = config->population_size;
        int num_obj = config->num_objectives;
        
        cudaError_t err1 = cudaMalloc(&workspace->d_objectives, pop_size * num_obj * sizeof(T));
        cudaError_t err2 = cudaMalloc(&workspace->d_ranks, pop_size * sizeof(int));
        cudaError_t err3 = cudaMalloc(&workspace->d_crowding_dist, pop_size * sizeof(T));
        
        if (err1 == cudaSuccess && err2 == cudaSuccess && err3 == cudaSuccess) {
            workspace->allocated = true;
        } else {
            workspace->cleanup();
        }
    }
    
    void enable_peer_access() {
        for (size_t i = 0; i < gpu_workspaces.size(); i++) {
            cudaSetDevice(gpu_workspaces[i]->device_id);
            for (size_t j = 0; j < gpu_workspaces.size(); j++) {
                if (i != j) {
                    int can_access;
                    cudaDeviceCanAccessPeer(&can_access, 
                                          gpu_workspaces[i]->device_id, 
                                          gpu_workspaces[j]->device_id);
                    if (can_access) {
                        cudaDeviceEnablePeerAccess(gpu_workspaces[j]->device_id, 0);
                    }
                }
            }
        }
    }
    
    void cleanup_multi_gpu() {
        for (auto& workspace : gpu_workspaces) {
            workspace->cleanup();
        }
        gpu_workspaces.clear();
        multi_gpu_enabled = false;
        num_active_gpus = 0;
    }
    
    // CRITICAL FIX: Thread-safe multi-GPU dominance checking
    void fast_non_dominated_sort_multi_gpu() {
        if (!multi_gpu_enabled || gpu_workspaces.size() < 2) {
            fast_non_dominated_sort();
            return;
        }
        
        fronts.clear();
        fronts.push_back(std::vector<int>());
        
        for (auto& ind : individuals) {
            ind.reset_nsga2_data();
        }
        
        std::vector<std::future<void>> futures;
        size_t pop_size = individuals.size();
        size_t chunk_size = (pop_size + num_active_gpus - 1) / num_active_gpus;
        
        for (int gpu_idx = 0; gpu_idx < num_active_gpus; gpu_idx++) {
            size_t start_idx = gpu_idx * chunk_size;
            size_t end_idx = std::min(start_idx + chunk_size, pop_size);
            
            if (start_idx >= pop_size) break;
            
            futures.push_back(std::async(std::launch::async, 
                [this, start_idx, end_idx, gpu_idx]() {
                    calculate_domination_chunk_safe(start_idx, end_idx, gpu_idx);
                }));
        }
        
        for (auto& future : futures) {
            future.wait();
        }
        
        build_fronts_from_domination();
    }
    
    // CRITICAL FIX: Thread-safe domination calculation
    void calculate_domination_chunk_safe(size_t start_idx, size_t end_idx, int gpu_idx) {
        if (gpu_idx < static_cast<int>(gpu_workspaces.size())) {
            cudaSetDevice(gpu_workspaces[gpu_idx]->device_id);
        }
        
        // Thread-local storage
        std::vector<std::vector<int>> local_dominated_solutions(end_idx - start_idx);
        std::vector<int> local_domination_counts(end_idx - start_idx, 0);
        
        for (size_t p = start_idx; p < end_idx; p++) {
            size_t local_idx = p - start_idx;
            
            for (size_t q = 0; q < individuals.size(); q++) {
                if (p == q) continue;
                
                if (individuals[p].dominates(individuals[q])) {
                    local_dominated_solutions[local_idx].push_back(q);
                } else if (individuals[q].dominates(individuals[p])) {
                    local_domination_counts[local_idx]++;
                }
            }
        }
        
        // Thread-safe update with mutex
        static std::mutex update_mutex;
        std::lock_guard<std::mutex> lock(update_mutex);
        
        for (size_t p = start_idx; p < end_idx; p++) {
            size_t local_idx = p - start_idx;
            individuals[p].dominated_solutions = local_dominated_solutions[local_idx];
            individuals[p].domination_count = local_domination_counts[local_idx];
            
            if (individuals[p].domination_count == 0) {
                individuals[p].rank = 0;
            }
        }
    }
    
    void build_fronts_from_domination() {
        for (size_t i = 0; i < individuals.size(); i++) {
            if (individuals[i].domination_count == 0) {
                individuals[i].rank = 0;
                fronts[0].push_back(i);
            }
        }
        
        int current_front = 0;
        while (current_front < static_cast<int>(fronts.size()) && !fronts[current_front].empty()) {
            std::vector<int> next_front;
            
            for (int p_idx : fronts[current_front]) {
                if (p_idx < 0 || p_idx >= static_cast<int>(individuals.size())) {
                    continue;
                }
                
                for (int q_idx : individuals[p_idx].dominated_solutions) {
                    if (q_idx < 0 || q_idx >= static_cast<int>(individuals.size())) {
                        continue;
                    }
                    
                    individuals[q_idx].domination_count--;
                    if (individuals[q_idx].domination_count == 0) {
                        individuals[q_idx].rank = current_front + 1;
                        next_front.push_back(q_idx);
                    }
                }
            }
            
            current_front++;
            if (!next_front.empty()) {
                fronts.push_back(next_front);
            }
        }
    }
    
    void calculate_crowding_distance_multi_gpu() {
        if (!multi_gpu_enabled || gpu_workspaces.size() < 2 || fronts.empty()) {
            calculate_crowding_distance();
            return;
        }
        
        std::vector<std::future<void>> futures;
        
        for (size_t i = 0; i < std::min(fronts.size(), gpu_workspaces.size()); i++) {
            if (fronts[i].empty()) continue;
            
            futures.push_back(std::async(std::launch::async,
                [this, i]() {
                    int gpu_idx = i % gpu_workspaces.size();
                    if (gpu_idx < static_cast<int>(gpu_workspaces.size())) {
                        cudaSetDevice(gpu_workspaces[gpu_idx]->device_id);
                    }
                    calculate_crowding_distance_for_front(
                        const_cast<std::vector<int>&>(fronts[i]));
                }));
        }
        
        for (size_t i = gpu_workspaces.size(); i < fronts.size(); i++) {
            if (!fronts[i].empty()) {
                calculate_crowding_distance_for_front(
                    const_cast<std::vector<int>&>(fronts[i]));
            }
        }
        
        for (auto& future : futures) {
            future.wait();
        }
    }
    
    // ============================================================
    // STANDARD NSGA-II METHODS
    // ============================================================
    
    void fast_non_dominated_sort() {
        if (multi_gpu_enabled && gpu_workspaces.size() >= 2) {
            fast_non_dominated_sort_multi_gpu();
            return;
        }
        
        fronts.clear();
        fronts.push_back(std::vector<int>());
        
        for (auto& ind : individuals) {
            ind.reset_nsga2_data();
        }
        
        for (size_t p = 0; p < individuals.size(); p++) {
            for (size_t q = 0; q < individuals.size(); q++) {
                if (p == q) continue;
                
                if (individuals[p].dominates(individuals[q])) {
                    individuals[p].dominated_solutions.push_back(q);
                } else if (individuals[q].dominates(individuals[p])) {
                    individuals[p].domination_count++;
                }
            }
            
            if (individuals[p].domination_count == 0) {
                individuals[p].rank = 0;
                fronts[0].push_back(p);
            }
        }
        
        build_fronts_from_domination();
    }
    
    void calculate_crowding_distance() {
        if (multi_gpu_enabled && gpu_workspaces.size() >= 2) {
            calculate_crowding_distance_multi_gpu();
            return;
        }
        
        for (auto& front : fronts) {
            calculate_crowding_distance_for_front(front);
        }
    }
    
    void calculate_crowding_distance_for_front(std::vector<int>& front_indices) {
        if (front_indices.size() <= 2) {
            for (int idx : front_indices) {
                individuals[idx].crowding_distance = std::numeric_limits<T>::infinity();
            }
            return;
        }
        
        for (int idx : front_indices) {
            individuals[idx].crowding_distance = 0;
        }
        
        int num_objectives = config->num_objectives;
        
        for (int obj = 0; obj < num_objectives; obj++) {
            std::sort(front_indices.begin(), front_indices.end(),
                [this, obj](int a, int b) {
                    return individuals[a].objectives[obj] < individuals[b].objectives[obj];
                });
            
            individuals[front_indices.front()].crowding_distance = 
                std::numeric_limits<T>::infinity();
            individuals[front_indices.back()].crowding_distance = 
                std::numeric_limits<T>::infinity();
            
            T obj_range = individuals[front_indices.back()].objectives[obj] - 
                         individuals[front_indices.front()].objectives[obj];
            
            if (obj_range > 0) {
                for (size_t i = 1; i < front_indices.size() - 1; i++) {
                    T distance = (individuals[front_indices[i+1]].objectives[obj] - 
                                 individuals[front_indices[i-1]].objectives[obj]) / obj_range;
                    individuals[front_indices[i]].crowding_distance += distance;
                }
            }
        }
    }
    
    void select_next_generation_nsga2() {
        std::vector<Individual<T>> selected;
        selected.reserve(config->population_size);

        for (auto& front : fronts) {
            if (selected.size() + front.size() <= static_cast<size_t>(config->population_size)) {
                for (int idx : front) {
                    selected.push_back(individuals[idx]);
                }
            } else {
                std::sort(front.begin(), front.end(), 
                    [this](int a, int b) {
                        return individuals[a].crowding_distance > 
                               individuals[b].crowding_distance;
                    });
                
                size_t remaining = config->population_size - selected.size();
                for (size_t i = 0; i < remaining; i++) {
                    selected.push_back(individuals[front[i]]);
                }
                break;
            }
        }
        
        individuals = selected;
        fronts.clear();
    }
    
    // ============================================================
    // EVALUATION METHODS
    // ============================================================
    
    void evaluate_all() {
        if (use_gpu_evaluation) {
            evaluate_all_gpu();
        } else {
            evaluate_all_cpu();
        }
    }
    
private:
    int population_gpu_device_id = 0;  // Device where d_population/d_fitness are allocated

    void setup_gpu_evaluation() {
        // Record which device we're allocating on (always use device 0 for initial population)
        cudaSetDevice(0);
        population_gpu_device_id = 0;

        size_t pop_bytes = config->population_size * total_chrom_len * sizeof(T);
        size_t fitness_bytes = config->population_size * sizeof(T);

        cudaError_t error1 = cudaMalloc(&d_population, pop_bytes);
        cudaError_t error2 = cudaMalloc(&d_fitness, fitness_bytes);

        if (error1 == cudaSuccess && error2 == cudaSuccess) {
            gpu_memory_allocated = true;
            use_gpu_evaluation = true;
        } else {
            if (d_population) cudaFree(d_population);
            if (d_fitness) cudaFree(d_fitness);
            d_population = nullptr;
            d_fitness = nullptr;
            gpu_memory_allocated = false;
            use_gpu_evaluation = false;
        }
    }
    
    void cleanup_gpu_memory() {
        if (gpu_memory_allocated) {
            cudaFree(d_population);
            cudaFree(d_fitness);
            gpu_memory_allocated = false;
            use_gpu_evaluation = false;
        }
    }
    
    void evaluate_all_gpu() {
        // Ensure we're on the device where d_population was allocated
        cudaSetDevice(population_gpu_device_id);

        copy_population_to_gpu();
        config->evaluate_population_gpu(d_population, d_fitness,
                                       config->population_size, total_chrom_len);

        if (is_multi_objective) {
            copy_objectives_from_gpu();
        } else {
            copy_fitness_from_gpu();
        }
    }
    
    void copy_population_to_gpu() {
        if (!gpu_memory_allocated) return;
        
        std::vector<T> flat_pop;
        flat_pop.reserve(config->population_size * total_chrom_len);
        
        for (const auto& individual : individuals) {
            auto flat_individual = individual.flatten();
            flat_pop.insert(flat_pop.end(), flat_individual.begin(), flat_individual.end());
        }
        
        cudaMemcpy(d_population, flat_pop.data(), 
                   flat_pop.size() * sizeof(T), cudaMemcpyHostToDevice);
    }
    
    void copy_fitness_from_gpu() {
        if (!gpu_memory_allocated) return;
        
        std::vector<T> host_fitness(config->population_size);
        cudaMemcpy(host_fitness.data(), d_fitness, 
                   config->population_size * sizeof(T), cudaMemcpyDeviceToHost);
        
        for (int i = 0; i < config->population_size; i++) {
            individuals[i].set_fitness(host_fitness[i]);
        }
    }
    
    void copy_objectives_from_gpu() {
        if (!gpu_memory_allocated) return;
        
        int num_obj = config->num_objectives;
        std::vector<T> host_objectives(config->population_size * num_obj);
        cudaMemcpy(host_objectives.data(), d_fitness, 
                   host_objectives.size() * sizeof(T), cudaMemcpyDeviceToHost);
        
        for (int i = 0; i < config->population_size; i++) {
            std::vector<T> objs(num_obj);
            for (int j = 0; j < num_obj; j++) {
                objs[j] = host_objectives[i * num_obj + j];
            }
            individuals[i].set_objectives(objs);
        }
    }
    
    void evaluate_all_cpu() {
        if (is_multi_objective) {
            if (config->objective_functions.empty()) {
                throw std::runtime_error("Multi-objective mode but no objective functions defined");
            }
            
            for (auto& individual : individuals) {
                if (!individual.evaluated) {
                    std::vector<T> objs(config->num_objectives);
                    for (int i = 0; i < config->num_objectives; i++) {
                        objs[i] = config->objective_functions[i](individual);
                    }
                    individual.set_objectives(objs);
                }
            }
        } else {
            if (!config->fitness_function) {
                throw std::runtime_error("Fitness function not set");
            }
            
            for (auto& individual : individuals) {
                if (!individual.evaluated) {
                    T fitness = config->fitness_function(individual);
                    individual.set_fitness(fitness);
                }
            }
        }
    }
    
public:
    // ============================================================
    // UTILITY METHODS
    // ============================================================
    
    void sort() {
        if (is_multi_objective) {
            std::sort(individuals.begin(), individuals.end(),
                [](const Individual<T>& a, const Individual<T>& b) {
                    if (a.rank != b.rank) return a.rank < b.rank;
                    return a.crowding_distance > b.crowding_distance;
                });
        } else {
            std::sort(individuals.begin(), individuals.end(),
                [this](const Individual<T>& a, const Individual<T>& b) {
                    return config->comparator ? 
                           config->comparator(a.fitness, b.fitness) :
                           a.fitness < b.fitness;
                });
        }
    }
    
    void update_best() {
        sort();
        if (!individuals.empty()) {
            best_individual = individuals[0];
        }
    }
    
    void update_best_from_front() {
        if (!fronts.empty() && !fronts[0].empty()) {
            int best_idx = fronts[0][0];
            T max_crowding = individuals[best_idx].crowding_distance;
            
            for (int idx : fronts[0]) {
                if (individuals[idx].crowding_distance > max_crowding) {
                    max_crowding = individuals[idx].crowding_distance;
                    best_idx = idx;
                }
            }
            
            best_individual = individuals[best_idx];
        }
    }
    
    void next_generation_step() {
        for (auto& individual : next_generation) {
            individual.reset_evaluation();
        }
        
        if (!is_multi_objective) {
            for (int i = 0; i < config->elite_size; i++) {
                next_generation[i] = individuals[i];
            }
        }
        
        for (int i = config->elite_size; i < config->population_size; i++) {
            next_generation[i].reset_evaluation();
        }
    }
    
    void finalize_generation() {
        individuals.swap(next_generation);
        evaluate_all();
        
        if (is_multi_objective) {
            fast_non_dominated_sort();
            calculate_crowding_distance();
            update_best_from_front();
        } else {
            update_best();
        }
        
        fitness_history.push_back(best_individual.fitness);
    }

    // Helper methods for solver-controlled evaluation
    void swap_generations() {
        individuals.swap(next_generation);
    }

    void evaluate_all_single_gpu() {
        evaluate_all();
    }

    void record_fitness() {
        fitness_history.push_back(best_individual.fitness);
    }

    // For GPU-resident BRKGA: set best fitness directly from GPU
    void set_best_fitness(T fitness) {
        best_individual.fitness = fitness;
        best_individual.evaluated = true;
        gpu_resident_active = true;  // Mark that population data is on GPU
    }

    // Check if GPU-resident mode is active (CPU population data is stale)
    bool is_gpu_resident() const { return gpu_resident_active; }

    // ============================================================
    // GETTERS
    // ============================================================
    
    const Individual<T>& get_best() const { return best_individual; }
    Individual<T>& get_best_mutable() { return best_individual; }
    const std::vector<Individual<T>>& get_individuals() const { return individuals; }
    std::vector<Individual<T>>& get_individuals() { return individuals; }
    std::vector<Individual<T>>& get_next_generation() { return next_generation; }
    const std::vector<T>& get_fitness_history() const { return fitness_history; }
    const std::vector<std::vector<int>>& get_fronts() const { return fronts; }
    std::vector<std::vector<int>>& get_fronts() { return fronts; }
    
    std::vector<Individual<T>> get_pareto_front() const {
        std::vector<Individual<T>> front;
        
        if (fronts.empty()) {
            return front;
        }
        
        if (fronts[0].empty()) {
            return front;
        }
        
        for (int idx : fronts[0]) {
            if (idx >= 0 && idx < static_cast<int>(individuals.size())) {
                front.push_back(individuals[idx]);
            }
        }
        
        return front;
    }
    
    std::vector<Individual<T>> get_elite() const {
        std::vector<Individual<T>> elite;
        elite.reserve(config->elite_size);
        for (int i = 0; i < config->elite_size && i < static_cast<int>(individuals.size()); i++) {
            elite.push_back(individuals[i]);
        }
        return elite;
    }
    
    std::vector<Individual<T>> get_non_elite() const {
        std::vector<Individual<T>> non_elite;
        non_elite.reserve(individuals.size() - config->elite_size);
        for (int i = config->elite_size; i < static_cast<int>(individuals.size()); i++) {
            non_elite.push_back(individuals[i]);
        }
        return non_elite;
    }
    
    T get_average_fitness() const {
        if (individuals.empty()) return T(0);
        T sum = T(0);
        for (const auto& individual : individuals) {
            sum += individual.fitness;
        }
        return sum / static_cast<T>(individuals.size());
    }
    
    T get_worst_fitness() const {
        if (individuals.empty()) return T(0);
        return individuals.back().fitness;
    }
    
    double get_diversity() const {
        if (individuals.size() < 2) return 0.0;
        
        double total_distance = 0.0;
        int comparisons = 0;
        
        for (size_t i = 0; i < individuals.size(); i++) {
            for (size_t j = i + 1; j < individuals.size(); j++) {
                double distance = 0.0;
                
                for (int comp = 0; comp < config->num_components; comp++) {
                    for (int k = 0; k < config->component_lengths[comp]; k++) {
                        double diff = individuals[i].get_component(comp)[k] - 
                                     individuals[j].get_component(comp)[k];
                        distance += diff * diff;
                    }
                }
                
                total_distance += std::sqrt(distance);
                comparisons++;
            }
        }
        
        return total_distance / comparisons;
    }
    
    void print_statistics(int generation) const {
        if (is_multi_objective) {
            if (fronts.empty()) {
                const_cast<Population<T>*>(this)->fast_non_dominated_sort();
                const_cast<Population<T>*>(this)->calculate_crowding_distance();
            }
            
            int pareto_size = 0;
            if (!fronts.empty() && !fronts[0].empty()) {
                for (int idx : fronts[0]) {
                    if (idx >= 0 && idx < static_cast<int>(individuals.size())) {
                        pareto_size++;
                    }
                }
            }
            
            // Count unique solutions in Pareto front
            std::set<std::pair<T, T>> unique_pareto;
            if (!fronts.empty() && fronts[0].size() >= 2) {
                for (int idx : fronts[0]) {
                    if (idx >= 0 && idx < static_cast<int>(individuals.size()) && 
                        individuals[idx].objectives.size() >= 2) {
                        unique_pareto.insert({individuals[idx].objectives[0], 
                                            individuals[idx].objectives[1]});
                    }
                }
            }
            
            std::cout << "Gen " << std::setw(4) << generation 
                      << ": Fronts=" << fronts.size()
                      << ", Pareto=" << pareto_size
                      << ", Unique=" << unique_pareto.size()
                      << ", Diversity=" << std::fixed << std::setprecision(4) << get_diversity();
            
            if (multi_gpu_enabled) {
                std::cout << " [Multi-GPU:" << num_active_gpus << "]";
            } else if (use_gpu_evaluation) {
                std::cout << " [GPU]";
            } else {
                std::cout << " [CPU]";
            }
        } else {
            std::cout << "Generation " << std::setw(4) << generation
                      << ": Best=" << std::fixed << std::setprecision(2) << get_best().fitness;

            // Skip expensive stats when population is on GPU (data is stale on CPU)
            if (!gpu_resident_active) {
                std::cout << ", Avg=" << std::setprecision(2) << get_average_fitness()
                          << ", Worst=" << std::setprecision(2) << get_worst_fitness()
                          << ", Diversity=" << std::setprecision(4) << get_diversity();
            }

            if (gpu_resident_active) {
                std::cout << " [GPU-Resident]";
            } else if (use_gpu_evaluation) {
                std::cout << " [GPU]";
            } else {
                std::cout << " [CPU]";
            }
        }
        std::cout << std::endl;
    }
    
    bool validate_all() const {
        for (const auto& individual : individuals) {
            if (!config->validate_solution(individual)) {
                return false;
            }
        }
        return true;
    }
    
    bool is_using_gpu() const { return use_gpu_evaluation; }
    bool is_using_multi_objective() const { return is_multi_objective; }
    bool is_using_multi_gpu() const { return multi_gpu_enabled; }
    int get_num_active_gpus() const { return num_active_gpus; }
    
    size_t size() const { return individuals.size(); }
    bool empty() const { return individuals.empty(); }
    
    T calculate_hypervolume_2d(T ref_x = 1.0, T ref_y = 1.0) const {
        if (!is_multi_objective || config->num_objectives != 2 || fronts.empty()) {
            return T(0);
        }
        
        auto pareto = get_pareto_front();
        if (pareto.empty()) return T(0);
        
        std::vector<std::pair<T, T>> points;
        for (const auto& ind : pareto) {
            if (ind.objectives[0] <= ref_x && ind.objectives[1] <= ref_y) {
                points.emplace_back(ind.objectives[0], ind.objectives[1]);
            }
        }
        
        if (points.empty()) return T(0);
        
        std::sort(points.begin(), points.end());
        
        T volume = 0;
        T prev_x = 0;
        
        for (const auto& point : points) {
            volume += (point.first - prev_x) * (ref_y - point.second);
            prev_x = point.first;
        }
        
        return volume;
    }
    
    double calculate_spread() const {
        if (!is_multi_objective || fronts.empty() || fronts[0].size() < 2) {
            return 0.0;
        }
        
        auto pareto = get_pareto_front();
        if (pareto.size() < 2) return 0.0;
        
        std::vector<double> distances;
        double sum_distance = 0.0;
        
        for (size_t i = 0; i < pareto.size() - 1; i++) {
            double distance = 0.0;
            for (int obj = 0; obj < config->num_objectives; obj++) {
                double diff = pareto[i].objectives[obj] - pareto[i + 1].objectives[obj];
                distance += diff * diff;
            }
            distance = std::sqrt(distance);
            distances.push_back(distance);
            sum_distance += distance;
        }
        
        if (distances.empty()) return 0.0;
        
        double mean_distance = sum_distance / distances.size();
        
        double spread = 0.0;
        for (double d : distances) {
            spread += std::abs(d - mean_distance);
        }
        
        return spread / (distances.size() * mean_distance);
    }
    
    std::vector<Individual<T>> get_front(int front_index) const {
        std::vector<Individual<T>> front_individuals;
        
        if (front_index < 0 || front_index >= static_cast<int>(fronts.size())) {
            return front_individuals;
        }
        
        for (int idx : fronts[front_index]) {
            if (idx >= 0 && idx < static_cast<int>(individuals.size())) {
                front_individuals.push_back(individuals[idx]);
            }
        }
        
        return front_individuals;
    }
    
    void print_detailed_statistics() const {
        std::cout << "\n=========================================" << std::endl;
        std::cout << "  Detailed Population Statistics" << std::endl;
        std::cout << "=========================================" << std::endl;
        
        std::cout << "Population size: " << individuals.size() << std::endl;
        std::cout << "Multi-objective: " << (is_multi_objective ? "Yes" : "No") << std::endl;
        
        if (is_multi_objective) {
            std::cout << "Number of objectives: " << config->num_objectives << std::endl;
            std::cout << "Number of fronts: " << fronts.size() << std::endl;
            
            if (!fronts.empty()) {
                std::cout << "\nFront sizes:" << std::endl;
                for (size_t i = 0; i < std::min(size_t(5), fronts.size()); i++) {
                    std::cout << "  Front " << i << ": " << fronts[i].size() << " individuals" << std::endl;
                }
                if (fronts.size() > 5) {
                    std::cout << "  ... and " << (fronts.size() - 5) << " more fronts" << std::endl;
                }
            }
            
            if (!individuals.empty() && !individuals[0].objectives.empty()) {
                std::cout << "\nObjective ranges:" << std::endl;
                for (int obj = 0; obj < config->num_objectives; obj++) {
                    T min_val = individuals[0].objectives[obj];
                    T max_val = individuals[0].objectives[obj];
                    
                    for (const auto& ind : individuals) {
                        min_val = std::min(min_val, ind.objectives[obj]);
                        max_val = std::max(max_val, ind.objectives[obj]);
                    }
                    
                    std::cout << "  Objective " << obj << ": [" 
                              << std::fixed << std::setprecision(4) << min_val 
                              << ", " << max_val << "]" << std::endl;
                }
            }
            
            T min_crowding = std::numeric_limits<T>::infinity();
            T max_crowding = 0;
            T sum_crowding = 0;
            int finite_count = 0;
            
            for (const auto& ind : individuals) {
                if (!std::isinf(ind.crowding_distance)) {
                    min_crowding = std::min(min_crowding, ind.crowding_distance);
                    max_crowding = std::max(max_crowding, ind.crowding_distance);
                    sum_crowding += ind.crowding_distance;
                    finite_count++;
                }
            }
            
            if (finite_count > 0) {
                std::cout << "\nCrowding distance:" << std::endl;
                std::cout << "  Min: " << std::setprecision(4) << min_crowding << std::endl;
                std::cout << "  Max: " << std::setprecision(4) << max_crowding << std::endl;
                std::cout << "  Avg: " << std::setprecision(4) << (sum_crowding / finite_count) << std::endl;
            }
            
            std::cout << "\nQuality metrics:" << std::endl;
            std::cout << "  Diversity: " << std::setprecision(4) << get_diversity() << std::endl;
            std::cout << "  Spread: " << std::setprecision(4) << calculate_spread() << std::endl;
            
            if (config->num_objectives == 2) {
                T hv = calculate_hypervolume_2d();
                if (hv > 0) {
                    std::cout << "  Hypervolume (2D): " << std::setprecision(6) << hv << std::endl;
                }
            }
        } else {
            std::cout << "Best fitness: " << std::fixed << std::setprecision(6) 
                      << get_best().fitness << std::endl;
            std::cout << "Average fitness: " << std::setprecision(6) 
                      << get_average_fitness() << std::endl;
            std::cout << "Worst fitness: " << std::setprecision(6) 
                      << get_worst_fitness() << std::endl;
            std::cout << "Diversity: " << std::setprecision(4) 
                      << get_diversity() << std::endl;
        }
        
        std::cout << "\nExecution mode:" << std::endl;
        if (multi_gpu_enabled) {
            std::cout << "  Multi-GPU with " << num_active_gpus << " GPUs" << std::endl;
        } else if (use_gpu_evaluation) {
            std::cout << "  Single GPU" << std::endl;
        } else {
            std::cout << "  CPU only" << std::endl;
        }
        
        std::cout << "=========================================" << std::endl;
    }
};

#endif // POPULATION_HPP