// core/solver.hpp - CORRECTED with diversity preservation
#ifndef SOLVER_HPP
#define SOLVER_HPP

#include "config.hpp"
#include "population.hpp"
#include "individual.hpp"
#include "cuda_kernels.cuh"
#include "cuda_streams.hpp"
#include "../utils/timer.hpp"
#include <memory>
#include <iostream>
#include <iomanip>
#include <future>
#include <thread>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <random>
#include <limits>
#include <omp.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/copy.h>

// GPU Information Structure
struct GPUInfo {
    int device_id;
    std::string name;
    size_t total_memory;
    size_t free_memory;
    int compute_capability_major;
    int compute_capability_minor;
    int multiprocessor_count;
    int max_threads_per_block;
    bool is_available;
    double performance_weight;
};

template<typename T>
class Solver {
private:
    std::unique_ptr<BRKGAConfig<T>> config;
    std::unique_ptr<Population<T>> population;
    std::unique_ptr<Timer> timer;
    
    int current_generation;
    bool verbose;
    int print_frequency;
    
    // GPU Management
    std::vector<GPUInfo> gpu_info;
    std::vector<int> active_devices;
    int gpu_count;
    bool use_gpu;
    bool use_multi_gpu;
    std::string execution_mode;
    
    // GPU-resident BRKGA flag
    bool gpu_resident_brkga;
    bool gpu_resident_initialized;

    // Multi-GPU Island Model
    int migration_interval;           // Generations between migrations
    int num_migrants;                 // Number of individuals to migrate
    std::vector<T> island_best_fitness;  // Best fitness per island
    T global_best_fitness;

    // Local Search Parameters
    int local_search_interval;        // Generations between local search
    int local_search_individuals;     // Number of elite individuals to apply LS to
    int local_search_moves;           // Number of random 2-opt moves per individual
    bool enable_local_search;

    // GPU Memory Management
    struct GPUWorkspace {
        int device_id;
        T* d_population;
        T* d_elite_pop;
        T* d_non_elite_pop;
        T* d_offspring;
        T* d_mutants;
        T* d_fitness;          // For GPU-resident BRKGA
        T* d_fitness_backup;   // For local search fitness comparison
        T* d_backup;           // For local search chromosome backup
        int* d_indices;        // For thrust::sort indices
        curandState* d_states;

        T* d_objectives;
        int* d_ranks;
        T* d_crowding_dist;

        // BrkgaCuda 2.0 optimization: CUDA streams for async operations
        std::unique_ptr<StreamManager> stream_manager;
        std::unique_ptr<PinnedMemory<T>> pinned_migrants;  // For async migration transfers

        bool allocated;
        bool gpu_resident_initialized;

        GPUWorkspace(int id) : device_id(id), allocated(false), gpu_resident_initialized(false) {
            d_population = nullptr;
            d_elite_pop = nullptr;
            d_non_elite_pop = nullptr;
            d_offspring = nullptr;
            d_mutants = nullptr;
            d_fitness = nullptr;
            d_fitness_backup = nullptr;
            d_backup = nullptr;
            d_indices = nullptr;
            d_states = nullptr;
            d_objectives = nullptr;
            d_ranks = nullptr;
            d_crowding_dist = nullptr;

            // Initialize stream manager (3 streams: eval, BRKGA ops, memory transfers)
            try {
                cudaSetDevice(device_id);
                stream_manager = std::make_unique<StreamManager>(3);

                // Phase 3: Initialize pinned memory for async migrations
                // Size will be set when we know population parameters
                // Allocate conservatively for max expected migration size
                const int max_migrants = 1000;  // Conservative upper bound
                const int max_chrom_len = 10000;  // Conservative upper bound
                pinned_migrants = std::make_unique<PinnedMemory<T>>(max_migrants * max_chrom_len);
            } catch (const std::exception& e) {
                std::cerr << "Warning: Failed to initialize StreamManager/PinnedMemory for device "
                          << device_id << ": " << e.what() << std::endl;
                // Continue without streams (will fall back to synchronous operations)
            }
        }

        ~GPUWorkspace() { cleanup(); }

        void cleanup() {
            if (allocated) {
                cudaSetDevice(device_id);
                cudaFree(d_population);
                cudaFree(d_elite_pop);
                cudaFree(d_non_elite_pop);
                cudaFree(d_offspring);
                cudaFree(d_mutants);
                cudaFree(d_fitness);
                cudaFree(d_fitness_backup);
                cudaFree(d_backup);
                cudaFree(d_indices);
                cudaFree(d_states);
                cudaFree(d_objectives);
                cudaFree(d_ranks);
                cudaFree(d_crowding_dist);
                allocated = false;
            }
        }
    };
    
    std::vector<std::unique_ptr<GPUWorkspace>> gpu_workspaces;
    std::mt19937 rng;
    std::vector<size_t> work_distribution;
    
    struct PerformanceStats {
        double total_crossover_time = 0.0;
        double total_mutation_time = 0.0;
        double total_evaluation_time = 0.0;
        double total_sorting_time = 0.0;
        int operations_count = 0;
        
        void reset() {
            total_crossover_time = 0.0;
            total_mutation_time = 0.0;
            total_evaluation_time = 0.0;
            total_sorting_time = 0.0;
            operations_count = 0;
        }
    } perf_stats;
    
public:
    Solver(std::unique_ptr<BRKGAConfig<T>> cfg, bool verb = true, int print_freq = 50)
        : config(std::move(cfg)), verbose(verb), print_frequency(print_freq),
          current_generation(0), gpu_count(0), use_gpu(false), use_multi_gpu(false),
          gpu_resident_brkga(true), gpu_resident_initialized(false),
          migration_interval(100), num_migrants(5),
          global_best_fitness(std::numeric_limits<T>::max()),
          local_search_interval(2), local_search_individuals(100), local_search_moves(5), enable_local_search(true),
          rng(std::chrono::steady_clock::now().time_since_epoch().count()) {
        
        if (!this->config->is_valid()) {
            throw std::runtime_error("Invalid configuration");
        }
        
        population = std::make_unique<Population<T>>(this->config.get());
        timer = std::make_unique<Timer>();
        
        initialize_gpu_system();
        
        if (verbose) {
            print_system_configuration();
        }
    }
    
    ~Solver() {
        cleanup_gpu_resources();
        if (config) {
            config->cleanup_device_functions();
        }
    }
    
    void initialize() {
        if (verbose) {
            std::cout << "\nInitializing population..." << std::endl;
        }
        
        timer->start();
        population->initialize();
        
        current_generation = 0;
        
        if (verbose) {
            std::cout << "Population initialized in " << timer->elapsed_ms() << " ms" << std::endl;
            
            if (config->is_multi_objective()) {
                auto pareto = population->get_pareto_front();
                std::cout << "Initial Pareto front size: " << pareto.size() << std::endl;
            } else {
                std::cout << "Initial best fitness: " << std::fixed << std::setprecision(2) 
                          << population->get_best().fitness << std::endl;
            }
        }
    }
    
    void evolve_generation() {
        if (config->is_multi_objective()) {
            evolve_generation_nsga2();
        } else {
            // Use GPU-resident BRKGA if enabled and GPU available
            if (gpu_resident_brkga && config->has_gpu_evaluation() &&
                !gpu_workspaces.empty() && gpu_workspaces[0]->allocated) {
                // Use multi-GPU island model if multiple GPUs available
                if (gpu_workspaces.size() > 1) {
                    evolve_generation_brkga_multi_gpu_island();
                } else {
                    evolve_generation_brkga_gpu_resident();
                }
            } else {
                evolve_generation_brkga();
                perf_stats.operations_count++;
            }
        }
    }
    
private:
    void evolve_generation_brkga() {
        population->next_generation_step();
        auto& next_gen = population->get_next_generation();
        
        auto elite = population->get_elite();
        auto non_elite = population->get_non_elite();
        
        auto crossover_start = std::chrono::high_resolution_clock::now();
        perform_crossover(elite, non_elite, next_gen);
        auto crossover_end = std::chrono::high_resolution_clock::now();
        perf_stats.total_crossover_time += std::chrono::duration<double>(crossover_end - crossover_start).count();
        
        auto mutation_start = std::chrono::high_resolution_clock::now();
        perform_mutation(next_gen);
        auto mutation_end = std::chrono::high_resolution_clock::now();
        perf_stats.total_mutation_time += std::chrono::duration<double>(mutation_end - mutation_start).count();

        // Swap generations
        population->swap_generations();

        // Multi-GPU fitness evaluation
        auto eval_start = std::chrono::high_resolution_clock::now();
        if (use_multi_gpu && gpu_count > 1) {
            evaluate_population_multi_gpu_brkga(population->get_individuals());
        } else {
            population->evaluate_all_single_gpu();
        }
        auto eval_end = std::chrono::high_resolution_clock::now();
        perf_stats.total_evaluation_time += std::chrono::duration<double>(eval_end - eval_start).count();

        // Sort and update best (timed as sorting)
        auto sort_start = std::chrono::high_resolution_clock::now();
        population->update_best();
        auto sort_end = std::chrono::high_resolution_clock::now();
        perf_stats.total_sorting_time += std::chrono::duration<double>(sort_end - sort_start).count();

        population->record_fitness();

        current_generation++;
    }

    // GPU-Resident BRKGA: All evolution happens on GPU, only best fitness copied to host
    void evolve_generation_brkga_gpu_resident() {
        if (gpu_workspaces.empty() || !gpu_workspaces[0]->allocated) {
            // Fallback to regular BRKGA if GPU not available
            evolve_generation_brkga();
            return;
        }

        auto& workspace = gpu_workspaces[0];
        cudaSetDevice(workspace->device_id);

        int pop_size = config->population_size;
        int chrom_len = config->get_total_chromosome_length();
        int elite_size = config->elite_size;
        int mutant_size = config->mutant_size;
        float elite_prob = static_cast<float>(config->elite_prob);

        int threads = 256;
        int blocks = (pop_size + threads - 1) / threads;

        // First generation: initialize population on GPU
        if (!gpu_resident_initialized) {
            auto& individuals = population->get_individuals();

            // Flatten population to contiguous array
            std::vector<T> flat_pop(pop_size * chrom_len);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < pop_size; i++) {
                const auto& ind = individuals[i];
                size_t base = i * chrom_len;
                size_t pos = 0;
                for (int comp = 0; comp < ind.num_components(); comp++) {
                    const auto& chromosome = ind.get_component(comp);
                    for (size_t j = 0; j < chromosome.size(); j++) {
                        flat_pop[base + pos++] = chromosome[j];
                    }
                }
            }

            // Copy population to GPU
            cudaMemcpy(workspace->d_population, flat_pop.data(),
                       pop_size * chrom_len * sizeof(T), cudaMemcpyHostToDevice);

            gpu_resident_initialized = true;
        }

        // Step 1: Evaluate fitness on GPU (Stream 0)
        auto eval_start = std::chrono::high_resolution_clock::now();

        // Use stream 0 for evaluation if available, otherwise default stream
        cudaStream_t eval_stream = workspace->stream_manager ?
                                   workspace->stream_manager->get_stream(0) : 0;

        config->evaluate_population_gpu(workspace->d_population, workspace->d_fitness,
                                        pop_size, chrom_len);

        // Record event after evaluation for dependency tracking
        if (workspace->stream_manager) {
            workspace->stream_manager->record_event(0);
        }

        // Synchronize evaluation stream before sorting (thrust needs completed fitness)
        if (workspace->stream_manager) {
            workspace->stream_manager->synchronize_stream(0);
        } else {
            cudaDeviceSynchronize();
        }

        auto eval_end = std::chrono::high_resolution_clock::now();
        perf_stats.total_evaluation_time += std::chrono::duration<double>(eval_end - eval_start).count();

        // Step 2: Sort by fitness using thrust (Stream 1)
        auto sort_start = std::chrono::high_resolution_clock::now();

        // Initialize indices: 0, 1, 2, ..., pop_size-1
        thrust::device_ptr<int> d_indices_ptr(workspace->d_indices);
        thrust::sequence(d_indices_ptr, d_indices_ptr + pop_size);

        // Sort indices by fitness (ascending for minimization)
        // Note: Thrust uses default stream, future optimization could use custom stream
        thrust::device_ptr<T> d_fitness_ptr(workspace->d_fitness);
        thrust::sort_by_key(d_fitness_ptr, d_fitness_ptr + pop_size, d_indices_ptr);

        auto sort_end = std::chrono::high_resolution_clock::now();
        perf_stats.total_sorting_time += std::chrono::duration<double>(sort_end - sort_start).count();

        // Step 3: Run BRKGA generation kernel (elite copy + crossover + mutation) on Stream 1
        auto crossover_start = std::chrono::high_resolution_clock::now();

        // Use stream 1 for BRKGA operations
        cudaStream_t brkga_stream = workspace->stream_manager ?
                                    workspace->stream_manager->get_stream(1) : 0;

        brkga_generation_kernel<<<blocks, threads, 0, brkga_stream>>>(
            workspace->d_population,
            workspace->d_offspring,
            workspace->d_indices,
            workspace->d_states,
            pop_size,
            elite_size,
            mutant_size,
            chrom_len,
            elite_prob
        );

        // Phase 2 Optimization: Delay synchronization for better pipelining
        // Record event for BRKGA completion (for dependency tracking)
        if (workspace->stream_manager) {
            workspace->stream_manager->record_event(1);
        }

        // Only synchronize if we don't have streams (fallback to safe blocking behavior)
        if (!workspace->stream_manager) {
            cudaDeviceSynchronize();
        }
        // Otherwise, let BRKGA kernel run asynchronously - will sync before next generation

        auto crossover_end = std::chrono::high_resolution_clock::now();
        perf_stats.total_crossover_time += std::chrono::duration<double>(crossover_end - crossover_start).count();

        // Step 4: Synchronize BRKGA stream NOW before swapping (kernel must complete)
        // This is the critical sync point - can't swap pointers until kernel finishes
        if (workspace->stream_manager) {
            workspace->stream_manager->synchronize_stream(1);
        }

        // Swap population buffers
        std::swap(workspace->d_population, workspace->d_offspring);

        // Step 5: Copy best fitness to host for monitoring (Async on Stream 2)
        // This can overlap with next generation's setup work
        T best_fitness;

        cudaStream_t copy_stream = workspace->stream_manager ?
                                   workspace->stream_manager->get_stream(2) : 0;

        if (workspace->stream_manager) {
            // Async copy - starts immediately
            cudaMemcpyAsync(&best_fitness, workspace->d_fitness, sizeof(T),
                           cudaMemcpyDeviceToHost, copy_stream);

            // Phase 2 Optimization: Defer synchronization until we actually need the value
            // This allows the copy to overlap with loop overhead and next iteration setup
            // We'll sync just before using best_fitness below
            workspace->stream_manager->synchronize_stream(2);
        } else {
            cudaMemcpy(&best_fitness, workspace->d_fitness, sizeof(T), cudaMemcpyDeviceToHost);
        }

        // Update population's best fitness record (for display purposes)
        // Note: best_fitness is now synchronized and safe to use
        population->set_best_fitness(best_fitness);
        population->record_fitness();

        perf_stats.operations_count++;
        current_generation++;
    }

    // Multi-GPU Island Model: Each GPU runs independent evolution with periodic migration
    void evolve_generation_brkga_multi_gpu_island() {
        int num_islands = gpu_workspaces.size();
        int chrom_len = config->get_total_chromosome_length();
        int island_pop_size = config->population_size / num_islands;
        int island_elite_size = config->elite_size / num_islands;
        int island_mutant_size = config->mutant_size / num_islands;
        float elite_prob = static_cast<float>(config->elite_prob);

        // Initialize islands on first call
        if (!gpu_resident_initialized) {
            initialize_multi_gpu_islands(island_pop_size, chrom_len);
            island_best_fitness.resize(num_islands, std::numeric_limits<T>::max());
            gpu_resident_initialized = true;
        }

        // Timing accumulators
        double eval_time = 0, sort_time = 0, crossover_time = 0;

        // Phase 3 Optimization: Run all GPU islands asynchronously (non-blocking)
        // Launch all islands first, then synchronize - better GPU utilization
        for (int island_id = 0; island_id < num_islands; island_id++) {
            auto& workspace = gpu_workspaces[island_id];
            cudaSetDevice(workspace->device_id);

            cudaStream_t eval_stream = workspace->stream_manager ?
                                      workspace->stream_manager->get_stream(0) : 0;

            // Step 1: Evaluate fitness on dedicated stream (non-blocking)
            config->evaluate_population_gpu(workspace->d_population, workspace->d_fitness,
                                           island_pop_size, chrom_len);

            // Record event for dependency tracking
            if (workspace->stream_manager) {
                workspace->stream_manager->record_event(0);
            }
        }

        // Now process each island's evolution (overlap across GPUs where possible)
        #pragma omp parallel num_threads(num_islands) reduction(+:eval_time,sort_time,crossover_time)
        {
            int island_id = omp_get_thread_num();
            auto& workspace = gpu_workspaces[island_id];

            // Set device and verify it was set correctly
            cudaError_t set_err = cudaSetDevice(workspace->device_id);
            if (set_err != cudaSuccess) {
                // Skip this island if device is invalid
                continue;
            }

            int threads = 256;
            int blocks = (island_pop_size + threads - 1) / threads;

            auto eval_start = std::chrono::high_resolution_clock::now();

            // Synchronize evaluation stream (evaluation should be done or finishing)
            if (workspace->stream_manager) {
                workspace->stream_manager->synchronize_stream(0);
            } else {
                cudaDeviceSynchronize();
            }

            auto eval_end = std::chrono::high_resolution_clock::now();
            eval_time += std::chrono::duration<double>(eval_end - eval_start).count();

            // Step 2: Sort by fitness using thrust (with exception handling)
            auto sort_start = std::chrono::high_resolution_clock::now();
            try {
                // Re-set device before thrust operations to ensure context is correct
                cudaSetDevice(workspace->device_id);
                thrust::device_ptr<int> d_indices_ptr(workspace->d_indices);
                thrust::sequence(d_indices_ptr, d_indices_ptr + island_pop_size);
                thrust::device_ptr<T> d_fitness_ptr(workspace->d_fitness);
                thrust::sort_by_key(d_fitness_ptr, d_fitness_ptr + island_pop_size, d_indices_ptr);
            } catch (const thrust::system_error& e) {
                // If thrust fails, skip this island's evolution
                std::cerr << "Warning: Thrust error on island " << island_id
                          << " (device " << workspace->device_id << "): " << e.what() << std::endl;
                continue;
            }
            auto sort_end = std::chrono::high_resolution_clock::now();
            sort_time += std::chrono::duration<double>(sort_end - sort_start).count();

            // Step 3: Run BRKGA generation kernel on Stream 1
            auto crossover_start = std::chrono::high_resolution_clock::now();

            cudaStream_t brkga_stream = workspace->stream_manager ?
                                        workspace->stream_manager->get_stream(1) : 0;

            brkga_generation_kernel<<<blocks, threads, 0, brkga_stream>>>(
                workspace->d_population,
                workspace->d_offspring,
                workspace->d_indices,
                workspace->d_states,
                island_pop_size,
                island_elite_size,
                island_mutant_size,
                chrom_len,
                elite_prob
            );

            // Phase 3: Record event and defer sync
            if (workspace->stream_manager) {
                workspace->stream_manager->record_event(1);
            }

            // Must sync before buffer swap (kernel needs to complete)
            if (workspace->stream_manager) {
                workspace->stream_manager->synchronize_stream(1);
            } else {
                cudaDeviceSynchronize();
            }

            auto crossover_end = std::chrono::high_resolution_clock::now();
            crossover_time += std::chrono::duration<double>(crossover_end - crossover_start).count();

            // Swap buffers
            std::swap(workspace->d_population, workspace->d_offspring);

            // Phase 3: Async fitness copy on Stream 2
            T island_best;
            cudaStream_t copy_stream = workspace->stream_manager ?
                                       workspace->stream_manager->get_stream(2) : 0;

            if (workspace->stream_manager) {
                cudaMemcpyAsync(&island_best, workspace->d_fitness, sizeof(T),
                               cudaMemcpyDeviceToHost, copy_stream);
                workspace->stream_manager->synchronize_stream(2);
            } else {
                cudaMemcpy(&island_best, workspace->d_fitness, sizeof(T), cudaMemcpyDeviceToHost);
            }

            #pragma omp critical
            {
                island_best_fitness[island_id] = island_best;
            }
        }

        // Update timing stats (average across islands)
        perf_stats.total_evaluation_time += eval_time / num_islands;
        perf_stats.total_sorting_time += sort_time / num_islands;
        perf_stats.total_crossover_time += crossover_time / num_islands;

        // Apply local search periodically (problem-specific, defined in config)
        if (enable_local_search && config->has_local_search() &&
            current_generation > 0 && current_generation % local_search_interval == 0) {
            apply_local_search(island_pop_size, chrom_len, island_elite_size);
        }

        // Perform migration every migration_interval generations
        if (current_generation > 0 && current_generation % migration_interval == 0) {
            perform_island_migration(island_pop_size, chrom_len);
        }

        // Find global best
        global_best_fitness = *std::min_element(island_best_fitness.begin(), island_best_fitness.end());
        population->set_best_fitness(global_best_fitness);
        population->record_fitness();

        perf_stats.operations_count++;
        current_generation++;
    }

    void initialize_multi_gpu_islands(int island_pop_size, int chrom_len) {
        auto& individuals = population->get_individuals();
        int num_islands = gpu_workspaces.size();
        int total_pop = individuals.size();

        // Distribute population evenly across islands
        #pragma omp parallel for num_threads(num_islands)
        for (int island_id = 0; island_id < num_islands; island_id++) {
            auto& workspace = gpu_workspaces[island_id];
            cudaSetDevice(workspace->device_id);

            // Flatten this island's portion of the population
            std::vector<T> flat_pop(island_pop_size * chrom_len);
            int start_idx = island_id * island_pop_size;

            for (int i = 0; i < island_pop_size; i++) {
                int global_idx = (start_idx + i) % total_pop;
                const auto& ind = individuals[global_idx];
                size_t base = i * chrom_len;
                size_t pos = 0;
                for (int comp = 0; comp < ind.num_components(); comp++) {
                    const auto& chromosome = ind.get_component(comp);
                    for (size_t j = 0; j < chromosome.size(); j++) {
                        flat_pop[base + pos++] = chromosome[j];
                    }
                }
            }

            // Copy to GPU
            cudaMemcpy(workspace->d_population, flat_pop.data(),
                       island_pop_size * chrom_len * sizeof(T), cudaMemcpyHostToDevice);
        }

        if (verbose) {
            std::cout << "Initialized " << num_islands << " GPU islands, "
                      << island_pop_size << " individuals per island" << std::endl;
        }
    }

    void perform_island_migration(int island_pop_size, int chrom_len) {
        int num_islands = gpu_workspaces.size();
        if (num_islands < 2) return;

        // Ring topology migration: island i sends to island (i+1) % num_islands
        std::vector<std::vector<T>> migrants(num_islands);

        // Extract best individuals from each island
        #pragma omp parallel for num_threads(num_islands)
        for (int i = 0; i < num_islands; i++) {
            auto& workspace = gpu_workspaces[i];
            cudaSetDevice(workspace->device_id);

            // Copy top num_migrants individuals (they're sorted by fitness)
            migrants[i].resize(num_migrants * chrom_len);
            cudaMemcpy(migrants[i].data(), workspace->d_population,
                       num_migrants * chrom_len * sizeof(T), cudaMemcpyDeviceToHost);
        }

        // Inject migrants into destination islands (replace worst individuals)
        #pragma omp parallel for num_threads(num_islands)
        for (int i = 0; i < num_islands; i++) {
            int src_island = (i + num_islands - 1) % num_islands;  // Receive from previous island
            auto& workspace = gpu_workspaces[i];
            cudaSetDevice(workspace->device_id);

            // Replace worst individuals (end of population after sorting)
            int dest_offset = (island_pop_size - num_migrants) * chrom_len;
            cudaMemcpy(workspace->d_population + dest_offset, migrants[src_island].data(),
                       num_migrants * chrom_len * sizeof(T), cudaMemcpyHostToDevice);
        }

        if (verbose && current_generation % (migration_interval * 4) == 0) {
            std::cout << "Migration at gen " << current_generation
                      << ": best fitness per island = [";
            for (int i = 0; i < num_islands; i++) {
                std::cout << std::fixed << std::setprecision(0) << island_best_fitness[i];
                if (i < num_islands - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
    }

    // Apply problem-specific local search to elite individuals on each GPU island
    // The actual local search logic is defined in the config (problem-specific)
    void apply_local_search(int island_pop_size, int chrom_len, int island_elite_size) {
        int num_islands = gpu_workspaces.size();

        // Number of elite individuals to apply local search to
        int num_to_improve = std::min(local_search_individuals, island_elite_size);
        if (num_to_improve <= 0) return;

        // Track improvements across all islands
        std::vector<int> improvements_per_island(num_islands, 0);
        std::vector<T> total_improvement_per_island(num_islands, 0);
        std::vector<T> best_before_per_island(num_islands);
        std::vector<T> best_after_per_island(num_islands);

        // Apply local search in parallel on all islands
        #pragma omp parallel for num_threads(num_islands)
        for (int island_id = 0; island_id < num_islands; island_id++) {
            auto& workspace = gpu_workspaces[island_id];
            cudaSetDevice(workspace->device_id);

            // Copy fitness to host for tracking (before)
            std::vector<T> fitness_before(num_to_improve);
            cudaMemcpy(fitness_before.data(), workspace->d_fitness,
                       num_to_improve * sizeof(T), cudaMemcpyDeviceToHost);
            best_before_per_island[island_id] = fitness_before[0];

            // Call problem-specific local search (defined in config)
            config->apply_local_search_gpu(
                workspace->d_population,
                workspace->d_backup,
                workspace->d_fitness,
                workspace->d_fitness_backup,
                workspace->d_states,
                island_pop_size,
                chrom_len,
                num_to_improve,
                local_search_moves
            );

            // Copy fitness to host for tracking (after local search)
            std::vector<T> fitness_after(num_to_improve);
            cudaMemcpy(fitness_after.data(), workspace->d_fitness,
                       num_to_improve * sizeof(T), cudaMemcpyDeviceToHost);

            // Count improvements
            int island_improvements = 0;
            T island_total_improvement = 0;
            for (int i = 0; i < num_to_improve; i++) {
                if (fitness_after[i] < fitness_before[i]) {
                    island_improvements++;
                    island_total_improvement += fitness_before[i] - fitness_after[i];
                }
            }
            improvements_per_island[island_id] = island_improvements;
            total_improvement_per_island[island_id] = island_total_improvement;
            best_after_per_island[island_id] = std::min(fitness_before[0], fitness_after[0]);
        }

        // Print local search statistics (only when there are improvements)
        int total_improvements = 0;
        T total_improvement = 0;
        for (int i = 0; i < num_islands; i++) {
            total_improvements += improvements_per_island[i];
            total_improvement += total_improvement_per_island[i];
        }

        if (verbose && total_improvements > 0) {
            int total_tried = num_to_improve * num_islands;

            std::cout << "  [LS Gen " << current_generation << "] "
                      << total_improvements << "/" << total_tried << " improved ("
                      << std::fixed << std::setprecision(1)
                      << (100.0 * total_improvements / total_tried) << "%), "
                      << "Gain: " << std::setprecision(1) << total_improvement;

            // Show best improvement
            T best_gain = 0;
            for (int i = 0; i < num_islands; i++) {
                T gain = best_before_per_island[i] - best_after_per_island[i];
                if (gain > best_gain) best_gain = gain;
            }
            if (best_gain > 0) {
                std::cout << ", Best elite: " << std::setprecision(1) << best_gain;
            }
            std::cout << std::endl;
        }
    }

    // Sync best individual's chromosome from GPU to CPU at end of evolution
    void sync_best_individual_from_gpu() {
        if (!use_multi_gpu && !gpu_resident_brkga) return;
        if (gpu_workspaces.empty()) return;

        int chrom_len = config->get_total_chromosome_length();
        int num_islands = gpu_workspaces.size();

        if (use_multi_gpu) {
            // Safety check: ensure island_best_fitness is populated
            if (island_best_fitness.empty() || island_best_fitness.size() != static_cast<size_t>(num_islands)) {
                // Fall back to population's best (already tracked on CPU)
                return;
            }

            // Verify at least one workspace has valid GPU memory
            bool any_valid = false;
            for (const auto& ws : gpu_workspaces) {
                if (ws && ws->allocated && ws->d_population && ws->d_fitness) {
                    any_valid = true;
                    break;
                }
            }
            if (!any_valid) {
                return;  // No valid GPU data, best is already tracked on CPU
            }

            // Find which island has the global best
            int best_island = 0;
            T best_fitness = island_best_fitness[0];
            for (int i = 1; i < num_islands; i++) {
                if (island_best_fitness[i] < best_fitness) {
                    best_fitness = island_best_fitness[i];
                    best_island = i;
                }
            }

            auto& workspace = gpu_workspaces[best_island];
            cudaSetDevice(workspace->device_id);

            // Re-evaluate fitness to get current state
            int island_pop_size = config->population_size / num_islands;
            config->evaluate_population_gpu(workspace->d_population, workspace->d_fitness,
                                            island_pop_size, chrom_len);
            cudaDeviceSynchronize();

            // Sort to find best
            thrust::device_ptr<int> d_indices_ptr(workspace->d_indices);
            thrust::sequence(d_indices_ptr, d_indices_ptr + island_pop_size);
            thrust::device_ptr<T> d_fitness_ptr(workspace->d_fitness);
            thrust::sort_by_key(d_fitness_ptr, d_fitness_ptr + island_pop_size, d_indices_ptr);

            // Get best fitness and index
            T final_best_fitness;
            int best_idx;
            cudaMemcpy(&final_best_fitness, workspace->d_fitness, sizeof(T), cudaMemcpyDeviceToHost);
            cudaMemcpy(&best_idx, workspace->d_indices, sizeof(int), cudaMemcpyDeviceToHost);

            // Copy best chromosome to host
            std::vector<T> best_chromosome(chrom_len);
            cudaMemcpy(best_chromosome.data(),
                       workspace->d_population + best_idx * chrom_len,
                       chrom_len * sizeof(T), cudaMemcpyDeviceToHost);

            // Update the population's best individual
            auto& best = population->get_best_mutable();
            best.fitness = final_best_fitness;
            best.evaluated = true;

            // Copy chromosome data
            int pos = 0;
            for (int comp = 0; comp < best.num_components(); comp++) {
                auto& chromosome = best.get_component(comp);
                for (size_t i = 0; i < chromosome.size(); i++) {
                    chromosome[i] = best_chromosome[pos++];
                }
            }

        } else if (gpu_resident_brkga && !gpu_workspaces.empty()) {
            // Single-GPU resident mode
            auto& workspace = gpu_workspaces[0];
            cudaSetDevice(workspace->device_id);

            int pop_size = config->population_size;

            // Re-evaluate and sort
            config->evaluate_population_gpu(workspace->d_population, workspace->d_fitness, pop_size, chrom_len);
            cudaDeviceSynchronize();

            thrust::device_ptr<int> d_indices_ptr(workspace->d_indices);
            thrust::sequence(d_indices_ptr, d_indices_ptr + pop_size);
            thrust::device_ptr<T> d_fitness_ptr(workspace->d_fitness);
            thrust::sort_by_key(d_fitness_ptr, d_fitness_ptr + pop_size, d_indices_ptr);

            // Get best
            T final_best_fitness;
            int best_idx;
            cudaMemcpy(&final_best_fitness, workspace->d_fitness, sizeof(T), cudaMemcpyDeviceToHost);
            cudaMemcpy(&best_idx, workspace->d_indices, sizeof(int), cudaMemcpyDeviceToHost);

            // Copy chromosome
            std::vector<T> best_chromosome(chrom_len);
            cudaMemcpy(best_chromosome.data(),
                       workspace->d_population + best_idx * chrom_len,
                       chrom_len * sizeof(T), cudaMemcpyDeviceToHost);

            auto& best = population->get_best_mutable();
            best.fitness = final_best_fitness;
            best.evaluated = true;

            int pos = 0;
            for (int comp = 0; comp < best.num_components(); comp++) {
                auto& chromosome = best.get_component(comp);
                for (size_t i = 0; i < chromosome.size(); i++) {
                    chromosome[i] = best_chromosome[pos++];
                }
            }
        }
    }

    void evolve_generation_nsga2() {
        auto& current_pop = population->get_individuals();
        
        // Evaluate parent population if needed
        auto eval_start = std::chrono::high_resolution_clock::now();
        if (use_multi_gpu && gpu_count > 1) {
            evaluate_population_multi_gpu(current_pop);
        } else {
            for (auto& ind : current_pop) {
                if (!ind.is_evaluated() || ind.objectives.empty()) {
                    evaluate_individual(ind);
                }
            }
        }
        auto eval_end = std::chrono::high_resolution_clock::now();
        perf_stats.total_evaluation_time += std::chrono::duration<double>(eval_end - eval_start).count();
        
        // Generate offspring
        auto crossover_start = std::chrono::high_resolution_clock::now();
        std::vector<Individual<T>> offspring;
        if (use_multi_gpu && gpu_count > 1) {
            offspring = generate_offspring_multi_gpu(current_pop);
        } else {
            offspring = generate_offspring_cpu(current_pop);
        }
        auto crossover_end = std::chrono::high_resolution_clock::now();
        perf_stats.total_crossover_time += std::chrono::duration<double>(crossover_end - crossover_start).count();
        
        // Evaluate offspring
        eval_start = std::chrono::high_resolution_clock::now();
        if (use_multi_gpu && gpu_count > 1) {
            evaluate_population_multi_gpu(offspring);
        } else {
            for (auto& ind : offspring) {
                evaluate_individual(ind);
            }
        }
        eval_end = std::chrono::high_resolution_clock::now();
        perf_stats.total_evaluation_time += std::chrono::duration<double>(eval_end - eval_start).count();
        
        // Combine populations
        std::vector<Individual<T>> combined;
        combined.reserve(current_pop.size() + offspring.size());
        combined.insert(combined.end(), current_pop.begin(), current_pop.end());
        combined.insert(combined.end(), offspring.begin(), offspring.end());
        
        population->get_individuals() = combined;
        
        // NSGA-II selection
        auto sort_start = std::chrono::high_resolution_clock::now();
        if (use_multi_gpu && gpu_count > 1) {
            nsga2_selection_multi_gpu();
        } else {
            population->fast_non_dominated_sort();
            population->calculate_crowding_distance();
            population->select_next_generation_nsga2();
        }
        auto sort_end = std::chrono::high_resolution_clock::now();
        perf_stats.total_sorting_time += std::chrono::duration<double>(sort_end - sort_start).count();
        
        // CRITICAL FIX: Add diversity preservation
        if (current_generation % 25 == 0) {
            population->ensure_minimum_diversity();
        }
        
        current_generation++;
    }
    
    // Multi-GPU offspring generation
    std::vector<Individual<T>> generate_offspring_multi_gpu(const std::vector<Individual<T>>& parents) {
        std::vector<Individual<T>> offspring;
        offspring.reserve(config->population_size);
        
        auto work_dist = distribute_work(config->population_size);
        std::vector<std::future<std::vector<Individual<T>>>> futures;
        
        size_t offset = 0;
        for (size_t gpu_idx = 0; gpu_idx < gpu_workspaces.size(); gpu_idx++) {
            size_t local_count = work_dist[gpu_idx];
            if (local_count == 0) continue;
            
            futures.push_back(std::async(std::launch::async, 
                [this, &parents, local_count, offset, gpu_idx]() {
                    return generate_offspring_on_gpu(parents, local_count, offset, gpu_idx);
                }));
            
            offset += local_count;
        }
        
        for (auto& future : futures) {
            auto partial = future.get();
            offspring.insert(offspring.end(), partial.begin(), partial.end());
        }
        
        return offspring;
    }
    
    std::vector<Individual<T>> generate_offspring_on_gpu(
        const std::vector<Individual<T>>& parents,
        size_t count, size_t offset, size_t gpu_idx) {
        
        cudaSetDevice(gpu_workspaces[gpu_idx]->device_id);
        
        std::vector<Individual<T>> local_offspring;
        local_offspring.reserve(count);
        
        std::mt19937 local_rng(rng() + offset);
        
        for (size_t i = 0; i < count; i++) {
            auto parent1 = tournament_selection(parents);
            auto parent2 = tournament_selection(parents);
            Individual<T> child = crossover_two_parents(parent1, parent2);
            mutate_individual(child, local_rng);  // FIXED: Pass RNG
            local_offspring.push_back(child);
        }
        
        return local_offspring;
    }
    
    void evaluate_population_multi_gpu(std::vector<Individual<T>>& pop) {
        auto work_dist = distribute_work(pop.size());
        std::vector<std::future<void>> futures;

        size_t offset = 0;
        for (size_t gpu_idx = 0; gpu_idx < gpu_workspaces.size(); gpu_idx++) {
            size_t local_count = work_dist[gpu_idx];
            if (local_count == 0) continue;

            futures.push_back(std::async(std::launch::async,
                [this, &pop, local_count, offset, gpu_idx]() {
                    evaluate_population_on_gpu(pop, offset, local_count, gpu_idx);
                }));

            offset += local_count;
        }

        for (auto& future : futures) {
            future.wait();
        }
    }

    void evaluate_population_on_gpu(std::vector<Individual<T>>& pop,
                                    size_t offset, size_t count, size_t gpu_idx) {
        cudaSetDevice(gpu_workspaces[gpu_idx]->device_id);

        for (size_t i = offset; i < offset + count; i++) {
            if (i >= pop.size()) break;
            evaluate_individual(pop[i]);
        }
    }

    // Multi-GPU batch evaluation for BRKGA using GPU kernels
    void evaluate_population_multi_gpu_brkga(std::vector<Individual<T>>& pop) {
        if (!config->has_gpu_evaluation()) {
            // Fallback to CPU if no GPU kernel available
            for (auto& ind : pop) {
                if (!ind.is_evaluated()) {
                    ind.set_fitness(config->fitness_function(ind));
                }
            }
            return;
        }

        auto work_dist = distribute_work(pop.size());
        std::vector<std::future<void>> futures;
        int chrom_len = config->get_total_chromosome_length();

        size_t offset = 0;
        for (size_t gpu_idx = 0; gpu_idx < gpu_workspaces.size(); gpu_idx++) {
            size_t local_count = work_dist[gpu_idx];
            if (local_count == 0) continue;

            futures.push_back(std::async(std::launch::async,
                [this, &pop, local_count, offset, gpu_idx, chrom_len]() {
                    evaluate_batch_on_gpu(pop, offset, local_count, gpu_idx, chrom_len);
                }));

            offset += local_count;
        }

        for (auto& future : futures) {
            future.wait();
        }
    }

    void evaluate_batch_on_gpu(std::vector<Individual<T>>& pop,
                               size_t offset, size_t count, size_t gpu_idx, int chrom_len) {
        auto& workspace = gpu_workspaces[gpu_idx];
        cudaSetDevice(workspace->device_id);

        // Use pre-allocated workspace buffers
        T* d_batch_pop = workspace->d_population;
        T* d_batch_fitness = reinterpret_cast<T*>(workspace->d_offspring);

        // Flatten population directly into contiguous array (parallelized)
        std::vector<T> flat_pop(count * chrom_len);

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < count; i++) {
            const auto& ind = pop[offset + i];
            size_t base = i * chrom_len;
            size_t pos = 0;
            for (int comp = 0; comp < ind.num_components(); comp++) {
                const auto& chromosome = ind.get_component(comp);
                for (size_t j = 0; j < chromosome.size(); j++) {
                    flat_pop[base + pos++] = chromosome[j];
                }
            }
        }

        // Copy to GPU and run kernel
        cudaMemcpy(d_batch_pop, flat_pop.data(), count * chrom_len * sizeof(T), cudaMemcpyHostToDevice);
        config->evaluate_population_gpu(d_batch_pop, d_batch_fitness, count, chrom_len);

        // Copy fitness back and update individuals
        std::vector<T> host_fitness(count);
        cudaMemcpy(host_fitness.data(), d_batch_fitness, count * sizeof(T), cudaMemcpyDeviceToHost);

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < count; i++) {
            pop[offset + i].set_fitness(host_fitness[i]);
        }
    }
    
    void nsga2_selection_multi_gpu() {
        population->fast_non_dominated_sort();
        
        auto& fronts = population->get_fronts();
        if (fronts.size() > 1 && gpu_count > 1) {
            calculate_crowding_distance_multi_gpu(fronts);
        } else {
            population->calculate_crowding_distance();
        }
        
        population->select_next_generation_nsga2();
    }
    
    void calculate_crowding_distance_multi_gpu(const std::vector<std::vector<int>>& fronts) {
        if (fronts.size() <= 1) {
            population->calculate_crowding_distance();
            return;
        }
        
        std::vector<std::future<void>> futures;
        
        for (size_t i = 0; i < std::min(fronts.size(), gpu_workspaces.size()); i++) {
            if (fronts[i].empty()) continue;
            
            futures.push_back(std::async(std::launch::async,
                [this, &fronts, i]() {
                    cudaSetDevice(gpu_workspaces[i % gpu_workspaces.size()]->device_id);
                    population->calculate_crowding_distance_for_front(
                        const_cast<std::vector<int>&>(fronts[i]));
                }));
        }
        
        for (size_t i = gpu_workspaces.size(); i < fronts.size(); i++) {
            if (!fronts[i].empty()) {
                population->calculate_crowding_distance_for_front(
                    const_cast<std::vector<int>&>(fronts[i]));
            }
        }
        
        for (auto& future : futures) {
            future.wait();
        }
    }
    
    std::vector<Individual<T>> generate_offspring_cpu(const std::vector<Individual<T>>& parents) {
        std::vector<Individual<T>> offspring;
        offspring.reserve(config->population_size);
        
        for (int i = 0; i < config->population_size; i++) {
            auto parent1 = tournament_selection(parents);
            auto parent2 = tournament_selection(parents);
            Individual<T> child = crossover_two_parents(parent1, parent2);
            mutate_individual(child, rng);  // FIXED: Pass RNG
            offspring.push_back(child);
        }
        
        return offspring;
    }
    
    void evaluate_individual(Individual<T>& individual) {
        std::vector<T> objs(config->num_objectives);
        for (int j = 0; j < config->num_objectives; j++) {
            objs[j] = config->objective_functions[j](individual);
        }
        individual.set_objectives(objs);
    }
    
    Individual<T> tournament_selection(const std::vector<Individual<T>>& pop) {
        std::uniform_int_distribution<int> dist(0, pop.size() - 1);
        
        int idx1 = dist(rng);
        int idx2 = dist(rng);
        
        const auto& ind1 = pop[idx1];
        const auto& ind2 = pop[idx2];
        
        if (ind1.rank < ind2.rank) return ind1;
        if (ind2.rank < ind1.rank) return ind2;
        
        if (ind1.crowding_distance > ind2.crowding_distance) return ind1;
        return ind2;
    }
    
    Individual<T> crossover_two_parents(const Individual<T>& parent1, const Individual<T>& parent2) {
        Individual<T> child(config->component_lengths, config->num_objectives);
        
        std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
        
        for (int comp = 0; comp < config->num_components; comp++) {
            for (int j = 0; j < config->component_lengths[comp]; j++) {
                if (prob_dist(rng) < config->elite_prob) {
                    child.get_component(comp)[j] = parent1.get_component(comp)[j];
                } else {
                    child.get_component(comp)[j] = parent2.get_component(comp)[j];
                }
            }
        }
        
        return child;
    }
    
    // CRITICAL FIX: Increased mutation rate with RNG parameter
    void mutate_individual(Individual<T>& individual, std::mt19937& local_rng) {
        std::uniform_real_distribution<T> dist(0.0, 1.0);
        std::uniform_real_distribution<double> mut_prob(0.0, 1.0);
        
        // INCREASED: 5x higher mutation rate for better diversity
        double mutation_rate = 5.0 / config->get_total_chromosome_length();
        
        for (int comp = 0; comp < config->num_components; comp++) {
            for (int j = 0; j < config->component_lengths[comp]; j++) {
                if (mut_prob(local_rng) < mutation_rate) {
                    individual.get_component(comp)[j] = dist(local_rng);
                }
            }
        }
    }
    
    void perform_crossover(const std::vector<Individual<T>>& elite,
                          const std::vector<Individual<T>>& non_elite,
                          std::vector<Individual<T>>& next_gen) {
        
        int offspring_count = config->get_offspring_size();
        if (offspring_count <= 0) return;
        
        if (execution_mode == "cpu") {
            perform_crossover_cpu(elite, non_elite, next_gen);
        } else if (execution_mode == "single_gpu") {
            perform_crossover_single_gpu(elite, non_elite, next_gen);
        } else if (execution_mode == "multi_gpu") {
            perform_crossover_multi_gpu(elite, non_elite, next_gen);
        }
    }
    
    void perform_crossover_cpu(const std::vector<Individual<T>>& elite,
                              const std::vector<Individual<T>>& non_elite,
                              std::vector<Individual<T>>& next_gen) {
        
        std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
        std::uniform_int_distribution<int> elite_dist(0, elite.size() - 1);
        std::uniform_int_distribution<int> non_elite_dist(0, non_elite.size() - 1);
        
        int offspring_count = config->get_offspring_size();
        for (int i = 0; i < offspring_count; i++) {
            int elite_parent = elite_dist(rng);
            int non_elite_parent = non_elite_dist(rng);
            
            auto& child = next_gen[config->elite_size + i];
            
            for (int comp = 0; comp < config->num_components; comp++) {
                for (int j = 0; j < config->component_lengths[comp]; j++) {
                    if (prob_dist(rng) < config->elite_prob) {
                        child.get_component(comp)[j] = elite[elite_parent].get_component(comp)[j];
                    } else {
                        child.get_component(comp)[j] = non_elite[non_elite_parent].get_component(comp)[j];
                    }
                }
            }
            child.reset_evaluation();
        }
    }
    
    void perform_crossover_single_gpu(const std::vector<Individual<T>>& elite,
                                     const std::vector<Individual<T>>& non_elite,
                                     std::vector<Individual<T>>& next_gen) {
        if (gpu_workspaces.empty() || !gpu_workspaces[0]->allocated) {
            perform_crossover_cpu(elite, non_elite, next_gen);
            return;
        }

        perform_crossover_on_gpu(elite, non_elite, next_gen, 0);
    }

    void perform_crossover_multi_gpu(const std::vector<Individual<T>>& elite,
                                    const std::vector<Individual<T>>& non_elite,
                                    std::vector<Individual<T>>& next_gen) {
        if (gpu_workspaces.empty()) {
            perform_crossover_cpu(elite, non_elite, next_gen);
            return;
        }

        // Use first GPU for crossover (could distribute across GPUs for larger populations)
        perform_crossover_on_gpu(elite, non_elite, next_gen, 0);
    }

    void perform_crossover_on_gpu(const std::vector<Individual<T>>& elite,
                                  const std::vector<Individual<T>>& non_elite,
                                  std::vector<Individual<T>>& next_gen,
                                  size_t gpu_idx) {
        auto& workspace = gpu_workspaces[gpu_idx];
        cudaSetDevice(workspace->device_id);

        int chrom_len = config->get_total_chromosome_length();
        int elite_size = elite.size();
        int non_elite_size = non_elite.size();
        int offspring_count = config->get_offspring_size();

        if (offspring_count <= 0) return;

        // Flatten elite population to contiguous array
        std::vector<T> flat_elite(elite_size * chrom_len);
        for (int i = 0; i < elite_size; i++) {
            auto flat = elite[i].flatten();
            std::copy(flat.begin(), flat.end(), flat_elite.begin() + i * chrom_len);
        }

        // Flatten non-elite population
        std::vector<T> flat_non_elite(non_elite_size * chrom_len);
        for (int i = 0; i < non_elite_size; i++) {
            auto flat = non_elite[i].flatten();
            std::copy(flat.begin(), flat.end(), flat_non_elite.begin() + i * chrom_len);
        }

        // Allocate temporary GPU memory for elite and non-elite
        T* d_elite = nullptr;
        T* d_non_elite = nullptr;
        cudaMalloc(&d_elite, elite_size * chrom_len * sizeof(T));
        cudaMalloc(&d_non_elite, non_elite_size * chrom_len * sizeof(T));

        // Copy to GPU
        cudaMemcpy(d_elite, flat_elite.data(), elite_size * chrom_len * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(d_non_elite, flat_non_elite.data(), non_elite_size * chrom_len * sizeof(T), cudaMemcpyHostToDevice);

        // Launch crossover kernel
        int threads_per_block = 256;
        int num_blocks = (offspring_count + threads_per_block - 1) / threads_per_block;

        crossover_kernel<<<num_blocks, threads_per_block>>>(
            d_elite, d_non_elite, workspace->d_offspring,
            workspace->d_states, offspring_count, chrom_len,
            config->elite_prob, elite_size, non_elite_size
        );
        cudaDeviceSynchronize();

        // Copy offspring back to host
        std::vector<T> flat_offspring(offspring_count * chrom_len);
        cudaMemcpy(flat_offspring.data(), workspace->d_offspring,
                   offspring_count * chrom_len * sizeof(T), cudaMemcpyDeviceToHost);

        // Unflatten into next_gen individuals
        for (int i = 0; i < offspring_count; i++) {
            auto& child = next_gen[config->elite_size + i];
            int offset = 0;
            for (int comp = 0; comp < config->num_components; comp++) {
                for (int j = 0; j < config->component_lengths[comp]; j++) {
                    child.get_component(comp)[j] = flat_offspring[i * chrom_len + offset + j];
                }
                offset += config->component_lengths[comp];
            }
            child.reset_evaluation();
        }

        // Cleanup
        cudaFree(d_elite);
        cudaFree(d_non_elite);
    }

    void perform_mutation(std::vector<Individual<T>>& next_gen) {
        if (config->mutant_size <= 0) return;

        if (execution_mode == "cpu") {
            perform_mutation_cpu(next_gen);
        } else if (!gpu_workspaces.empty() && gpu_workspaces[0]->allocated) {
            perform_mutation_gpu(next_gen, 0);
        } else {
            perform_mutation_cpu(next_gen);
        }
    }

    void perform_mutation_cpu(std::vector<Individual<T>>& next_gen) {
        std::uniform_real_distribution<T> dist(0.0, 1.0);

        int mutant_start = config->population_size - config->mutant_size;
        for (int i = 0; i < config->mutant_size; i++) {
            auto& mutant = next_gen[mutant_start + i];

            for (int comp = 0; comp < config->num_components; comp++) {
                for (int j = 0; j < config->component_lengths[comp]; j++) {
                    mutant.get_component(comp)[j] = dist(rng);
                }
            }
            mutant.reset_evaluation();
        }
    }

    void perform_mutation_gpu(std::vector<Individual<T>>& next_gen, size_t gpu_idx) {
        auto& workspace = gpu_workspaces[gpu_idx];
        cudaSetDevice(workspace->device_id);

        int chrom_len = config->get_total_chromosome_length();
        int mutant_size = config->mutant_size;
        int mutant_start = config->population_size - mutant_size;

        // Launch mutation kernel
        int threads_per_block = 256;
        int num_blocks = (mutant_size + threads_per_block - 1) / threads_per_block;

        mutation_kernel<<<num_blocks, threads_per_block>>>(
            workspace->d_mutants, workspace->d_states,
            mutant_size, chrom_len
        );
        cudaDeviceSynchronize();

        // Copy mutants back to host
        std::vector<T> flat_mutants(mutant_size * chrom_len);
        cudaMemcpy(flat_mutants.data(), workspace->d_mutants,
                   mutant_size * chrom_len * sizeof(T), cudaMemcpyDeviceToHost);

        // Unflatten into next_gen individuals
        for (int i = 0; i < mutant_size; i++) {
            auto& mutant = next_gen[mutant_start + i];
            int offset = 0;
            for (int comp = 0; comp < config->num_components; comp++) {
                for (int j = 0; j < config->component_lengths[comp]; j++) {
                    mutant.get_component(comp)[j] = flat_mutants[i * chrom_len + offset + j];
                }
                offset += config->component_lengths[comp];
            }
            mutant.reset_evaluation();
        }
    }
    
    void initialize_gpu_system() {
        detect_gpus();
        determine_execution_strategy();
        setup_gpu_resources();
        
        if (config) {
            config->initialize_device_functions();
        }
    }
    
    void detect_gpus() {
        gpu_info.clear();
        active_devices.clear();
        gpu_count = 0;
        
        int device_count;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        
        if (error != cudaSuccess || device_count == 0) {
            return;
        }
        
        for (int i = 0; i < device_count; i++) {
            cudaDeviceProp prop;
            error = cudaGetDeviceProperties(&prop, i);
            
            if (error != cudaSuccess) continue;
            
            GPUInfo info;
            info.device_id = i;
            info.name = prop.name;
            info.total_memory = prop.totalGlobalMem;
            info.compute_capability_major = prop.major;
            info.compute_capability_minor = prop.minor;
            info.multiprocessor_count = prop.multiProcessorCount;
            info.max_threads_per_block = prop.maxThreadsPerBlock;
            
            cudaSetDevice(i);
            cudaMemGetInfo(&info.free_memory, &info.total_memory);
            
            info.is_available = (info.compute_capability_major > 3 || 
                               (info.compute_capability_major == 3 && info.compute_capability_minor >= 5)) &&
                               info.free_memory > 100 * 1024 * 1024;
            
            info.performance_weight = info.multiprocessor_count * 
                                     (info.compute_capability_major * 10 + info.compute_capability_minor);
            
            gpu_info.push_back(info);
            
            if (info.is_available) {
                active_devices.push_back(i);
                gpu_count++;
            }
        }
    }
    
    void determine_execution_strategy() {
        const int min_pop_for_gpu = 100;
        const int min_pop_for_multi_gpu = 500;
        
        bool large_enough_for_gpu = config->population_size >= min_pop_for_gpu;
        bool large_enough_for_multi = config->population_size >= min_pop_for_multi_gpu;
        
        if (gpu_count == 0) {
            execution_mode = "cpu";
            use_gpu = false;
            use_multi_gpu = false;
        } else if (gpu_count > 1 && large_enough_for_multi) {
            execution_mode = "multi_gpu";
            use_gpu = true;
            use_multi_gpu = true;
            if (verbose) {
                std::cout << "Using multi-GPU mode with " << gpu_count << " GPUs for NSGA-II" << std::endl;
            }
        } else if (large_enough_for_gpu) {
            execution_mode = "single_gpu";
            use_gpu = true;
            use_multi_gpu = false;
            if (verbose) {
                std::cout << "Using single GPU mode" << std::endl;
            }
        } else {
            execution_mode = "cpu";
            use_gpu = false;
            use_multi_gpu = false;
            if (verbose) {
                std::cout << "Population too small for GPU, using CPU mode" << std::endl;
            }
        }
    }
    
    void setup_gpu_resources() {
        if (!use_gpu || gpu_count == 0) return;
        
        if (use_multi_gpu) {
            setup_multi_gpu_workspaces();
        }
    }
    
    void setup_multi_gpu_workspaces() {
        for (int device_id : active_devices) {
            auto workspace = std::make_unique<GPUWorkspace>(device_id);
            allocate_gpu_workspace(workspace.get());
            gpu_workspaces.push_back(std::move(workspace));
        }
        
        enable_peer_access();
    }
    
    void allocate_gpu_workspace(GPUWorkspace* workspace) {
        cudaSetDevice(workspace->device_id);

        int total_chrom_len = config->get_total_chromosome_length();
        int pop_size = config->population_size;
        int num_objectives = config->num_objectives;

        cudaMalloc(&workspace->d_population, pop_size * total_chrom_len * sizeof(T));
        cudaMalloc(&workspace->d_offspring, pop_size * total_chrom_len * sizeof(T));
        cudaMalloc(&workspace->d_mutants, config->mutant_size * total_chrom_len * sizeof(T));
        cudaMalloc(&workspace->d_states, pop_size * sizeof(curandState));

        // Allocate fitness and indices arrays for GPU-resident BRKGA
        cudaMalloc(&workspace->d_fitness, pop_size * sizeof(T));
        cudaMalloc(&workspace->d_indices, pop_size * sizeof(int));

        // Allocate backup buffers for local search
        cudaMalloc(&workspace->d_backup, pop_size * total_chrom_len * sizeof(T));
        cudaMalloc(&workspace->d_fitness_backup, pop_size * sizeof(T));

        // Initialize cuRAND states with unique seed per GPU
        int threads = 256;
        int blocks = (pop_size + threads - 1) / threads;
        unsigned long seed = 42 + workspace->device_id * 10000;
        init_curand_states_kernel<<<blocks, threads>>>(workspace->d_states, pop_size, seed);
        cudaDeviceSynchronize();

        if (config->is_multi_objective()) {
            cudaMalloc(&workspace->d_objectives, pop_size * num_objectives * sizeof(T));
            cudaMalloc(&workspace->d_ranks, pop_size * sizeof(int));
            cudaMalloc(&workspace->d_crowding_dist, pop_size * sizeof(T));
        }

        cudaError_t error = cudaGetLastError();
        if (error == cudaSuccess) {
            workspace->allocated = true;
        }
    }
    
    void enable_peer_access() {
        for (size_t i = 0; i < active_devices.size(); i++) {
            cudaSetDevice(active_devices[i]);
            for (size_t j = 0; j < active_devices.size(); j++) {
                if (i != j) {
                    int can_access;
                    cudaDeviceCanAccessPeer(&can_access, active_devices[i], active_devices[j]);
                    if (can_access) {
                        cudaDeviceEnablePeerAccess(active_devices[j], 0);
                    }
                }
            }
        }
    }
    
    void cleanup_gpu_resources() {
        for (auto& workspace : gpu_workspaces) {
            workspace->cleanup();
        }
        gpu_workspaces.clear();
    }
    
    std::vector<size_t> distribute_work(size_t total_work) {
        std::vector<size_t> distribution;
        
        if (!use_multi_gpu || gpu_workspaces.size() <= 1) {
            distribution.push_back(total_work);
            return distribution;
        }
        
        double total_weight = 0.0;
        for (size_t i = 0; i < gpu_workspaces.size(); i++) {
            total_weight += gpu_info[active_devices[i]].performance_weight;
        }
        
        size_t assigned = 0;
        for (size_t i = 0; i < gpu_workspaces.size() - 1; i++) {
            double weight = gpu_info[active_devices[i]].performance_weight;
            size_t work = static_cast<size_t>((weight / total_weight) * total_work);
            distribution.push_back(work);
            assigned += work;
        }
        
        distribution.push_back(total_work - assigned);
        
        return distribution;
    }
    
    void print_system_configuration() const {
        std::cout << "\n=== System Configuration ===" << std::endl;
        std::cout << "Mode: " << (config->is_multi_objective() ? "NSGA-II" : "BRKGA") << std::endl;
        std::cout << "Population size: " << config->population_size << std::endl;
        std::cout << "Chromosome length: " << config->get_total_chromosome_length() << std::endl;
        std::cout << "Max generations: " << config->max_generations << std::endl;
        std::cout << "Execution mode: " << execution_mode << std::endl;
        
        if (config->is_multi_objective()) {
            std::cout << "Objectives: " << config->num_objectives << std::endl;
        }
        
        if (use_multi_gpu) {
            std::cout << "GPUs: " << gpu_count << " active devices" << std::endl;
            for (size_t i = 0; i < gpu_workspaces.size(); i++) {
                std::cout << "  GPU " << gpu_workspaces[i]->device_id << ": " 
                          << gpu_info[active_devices[i]].name << std::endl;
            }
        }
        
        std::cout << "================================" << std::endl;
    }
    
public:
    void run() {
        if (verbose) {
            std::cout << "\n=========================================" << std::endl;
            std::cout << "Starting evolution (" << execution_mode << ")..." << std::endl;
            std::cout << "=========================================" << std::endl;
        }
        
        timer->start();
        initialize();
        
        for (int gen = 0; gen < config->max_generations; gen++) {
            evolve_generation();
            
            if (verbose && (gen % print_frequency == 0 || gen == config->max_generations - 1)) {
                population->print_statistics(gen);
            }
        }
        
        if (config->is_multi_objective()) {
            population->fast_non_dominated_sort();
        }

        // Sync best individual from GPU to CPU (for GPU-resident modes)
        if (use_multi_gpu || gpu_resident_brkga) {
            sync_best_individual_from_gpu();
        }

        double total_time = timer->elapsed_seconds();
        
        if (verbose) {
            std::cout << "\n=========================================" << std::endl;
            std::cout << "Evolution completed!" << std::endl;
            std::cout << "=========================================" << std::endl;
            print_final_results(total_time);
            
            if (use_multi_gpu) {
                print_multi_gpu_performance();
            }
        }
    }
    
    void print_final_results(double execution_time) const {
        std::cout << "Final Results:" << std::endl;
        std::cout << "  Execution time: " << std::fixed << std::setprecision(2) 
                  << execution_time << " seconds" << std::endl;
        std::cout << "  Generations: " << current_generation << "/" << config->max_generations << std::endl;
        
        if (config->is_multi_objective()) {
            try {
                auto pareto = population->get_pareto_front();
                std::cout << "  Pareto front size: " << pareto.size() << std::endl;
                
                // Count unique solutions
                std::set<std::pair<T, T>> unique_sols;
                for (const auto& ind : pareto) {
                    if (ind.objectives.size() >= 2) {
                        unique_sols.insert({ind.objectives[0], ind.objectives[1]});
                    }
                }
                std::cout << "  Unique solutions: " << unique_sols.size() << std::endl;
                std::cout << "  Diversity ratio: " << std::fixed << std::setprecision(2) 
                          << (double)unique_sols.size() / pareto.size() * 100 << "%" << std::endl;
                
                std::cout << "  Diversity: " << std::setprecision(4) 
                          << population->get_diversity() << std::endl;
            } catch (const std::exception& e) {
                std::cout << "  Error getting Pareto front: " << e.what() << std::endl;
            }
        } else {
            const auto& best = population->get_best();
            std::cout << "  Best fitness: " << std::setprecision(6) << best.fitness << std::endl;
        }
    }
    
    void print_multi_gpu_performance() const {
        if (!use_multi_gpu) return;
        
        std::cout << "\n=== Multi-GPU Performance Statistics ===" << std::endl;
        std::cout << "Active GPUs: " << gpu_workspaces.size() << std::endl;
        
        double total_time = perf_stats.total_evaluation_time + 
                           perf_stats.total_crossover_time + 
                           perf_stats.total_sorting_time;
        
        std::cout << "Time breakdown:" << std::endl;
        std::cout << "  Evaluation: " << std::fixed << std::setprecision(2) 
                  << perf_stats.total_evaluation_time << "s ("
                  << (perf_stats.total_evaluation_time / total_time * 100) << "%)" << std::endl;
        std::cout << "  Crossover: " << std::setprecision(2) 
                  << perf_stats.total_crossover_time << "s ("
                  << (perf_stats.total_crossover_time / total_time * 100) << "%)" << std::endl;
        std::cout << "  Sorting: " << std::setprecision(2) 
                  << perf_stats.total_sorting_time << "s ("
                  << (perf_stats.total_sorting_time / total_time * 100) << "%)" << std::endl;
        
        std::cout << "\nAverage time per generation:" << std::endl;
        if (perf_stats.operations_count > 0) {
            std::cout << "  Evaluation: " << std::setprecision(4) 
                      << (perf_stats.total_evaluation_time / perf_stats.operations_count * 1000) << " ms" << std::endl;
            std::cout << "  Crossover: " << std::setprecision(4) 
                      << (perf_stats.total_crossover_time / perf_stats.operations_count * 1000) << " ms" << std::endl;
            std::cout << "  Sorting: " << std::setprecision(4) 
                      << (perf_stats.total_sorting_time / perf_stats.operations_count * 1000) << " ms" << std::endl;
        }
        
        std::cout << "=========================================" << std::endl;
    }
    
    const Individual<T>& get_best_individual() const { 
        return population->get_best(); 
    }
    
    std::vector<Individual<T>> get_pareto_front() const {
        return population->get_pareto_front();
    }
    
    const std::vector<T>& get_fitness_history() const { 
        return population->get_fitness_history(); 
    }
    
    BRKGAConfig<T>* get_config() { 
        return config.get(); 
    }
    
    void export_pareto_front(const std::string& filename) {
        if (!config->is_multi_objective()) {
            std::cout << "Not a multi-objective problem" << std::endl;
            return;
        }
        
        auto pareto = get_pareto_front();
        std::ofstream file(filename);
        
        file << "# Pareto Front" << std::endl;
        file << "# Objectives: " << config->num_objectives << std::endl;
        file << "# Solutions: " << pareto.size() << std::endl;
        file << "# Generated by Multi-GPU NSGA-II" << std::endl;
        file << "# GPUs used: " << (use_multi_gpu ? gpu_count : 1) << std::endl;
        
        for (const auto& ind : pareto) {
            for (size_t i = 0; i < ind.objectives.size(); i++) {
                file << ind.objectives[i];
                if (i < ind.objectives.size() - 1) file << " ";
            }
            file << std::endl;
        }
        
        file.close();
        
        if (verbose) {
            std::cout << "Pareto front exported to: " << filename << std::endl;
        }
    }
    
    void benchmark_modes() {
        if (gpu_count < 2) {
            std::cout << "Multi-GPU benchmarking requires at least 2 GPUs" << std::endl;
            return;
        }
        
        std::cout << "\n=== Multi-GPU Benchmark ===" << std::endl;
        
        bool original_multi_gpu = use_multi_gpu;
        std::string original_mode = execution_mode;
        
        use_multi_gpu = false;
        execution_mode = "single_gpu";
        perf_stats.reset();
        
        std::cout << "Running with single GPU..." << std::endl;
        auto single_start = std::chrono::high_resolution_clock::now();
        
        for (int gen = 0; gen < 10; gen++) {
            evolve_generation();
        }
        
        auto single_end = std::chrono::high_resolution_clock::now();
        double single_time = std::chrono::duration<double>(single_end - single_start).count();
        
        population->initialize();
        use_multi_gpu = true;
        execution_mode = "multi_gpu";
        perf_stats.reset();
        
        std::cout << "Running with multi-GPU..." << std::endl;
        auto multi_start = std::chrono::high_resolution_clock::now();
        
        for (int gen = 0; gen < 10; gen++) {
            evolve_generation();
        }
        
        auto multi_end = std::chrono::high_resolution_clock::now();
        double multi_time = std::chrono::duration<double>(multi_end - multi_start).count();
        
        std::cout << "\nBenchmark Results (10 generations):" << std::endl;
        std::cout << "  Single GPU: " << std::fixed << std::setprecision(3) 
                  << single_time << " seconds" << std::endl;
        std::cout << "  Multi-GPU:  " << std::setprecision(3) 
                  << multi_time << " seconds" << std::endl;
        std::cout << "  Speedup:    " << std::setprecision(2) 
                  << (single_time / multi_time) << "x" << std::endl;
        std::cout << "  Efficiency: " << std::setprecision(1) 
                  << ((single_time / multi_time) / gpu_count * 100) << "%" << std::endl;
        std::cout << "===========================" << std::endl;
        
        use_multi_gpu = original_multi_gpu;
        execution_mode = original_mode;
    }
};

#endif // SOLVER_HPP