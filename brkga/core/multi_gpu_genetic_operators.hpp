#ifndef MULTI_GPU_GENETIC_OPERATORS_HPP
#define MULTI_GPU_GENETIC_OPERATORS_HPP

#include "genetic_operators.hpp"
#include "multi_gpu_manager.hpp"
#include <thread>
#include <future>
#include <vector>
#include <memory>

template<typename T>
class MultiGPUGeneticOperators : public GeneticOperators<T> {
private:
    struct GPUWorkspace {
        int device_id;
        T* d_population;
        T* d_elite_pop;
        T* d_non_elite_pop;
        T* d_offspring;
        T* d_mutants;
        curandState* d_states;
        bool allocated;
        
        GPUWorkspace(int id) : device_id(id), allocated(false) {
            d_population = nullptr;
            d_elite_pop = nullptr;
            d_non_elite_pop = nullptr;
            d_offspring = nullptr;
            d_mutants = nullptr;
            d_states = nullptr;
        }
        
        ~GPUWorkspace() {
            cleanup();
        }
        
        void cleanup() {
            if (allocated) {
                cudaSetDevice(device_id);
                cudaFree(d_population);
                cudaFree(d_elite_pop);
                cudaFree(d_non_elite_pop);
                cudaFree(d_offspring);
                cudaFree(d_mutants);
                cudaFree(d_states);
                allocated = false;
            }
        }
    };
    
    std::vector<std::unique_ptr<GPUWorkspace>> gpu_workspaces;
    MultiGPUManager& gpu_manager;
    bool multi_gpu_initialized;
    
    // Work distribution
    std::vector<size_t> elite_distribution;
    std::vector<size_t> non_elite_distribution;
    std::vector<size_t> offspring_distribution;
    std::vector<size_t> mutant_distribution;
    
public:
    MultiGPUGeneticOperators(BRKGAConfig<T>* cfg) 
        : GeneticOperators<T>(cfg), gpu_manager(g_multi_gpu_manager), multi_gpu_initialized(false) {
        
        if (gpu_manager.is_multi_gpu_enabled()) {
            initialize_multi_gpu();
        }
    }
    
    ~MultiGPUGeneticOperators() {
        cleanup_multi_gpu();
    }
    
private:
    void initialize_multi_gpu() {
        if (multi_gpu_initialized) return;
        
        auto active_devices = gpu_manager.get_active_devices();
        
        for (int device_id : active_devices) {
            auto workspace = std::make_unique<GPUWorkspace>(device_id);
            allocate_device_memory(*workspace);
            gpu_workspaces.push_back(std::move(workspace));
        }
        
        // Enable P2P access for efficient data transfer
        gpu_manager.enable_peer_access();
        
        multi_gpu_initialized = true;
        
        std::cout << "Multi-GPU genetic operators initialized with " 
                  << gpu_workspaces.size() << " GPUs" << std::endl;
    }
    
    void allocate_device_memory(GPUWorkspace& workspace) {
        cudaSetDevice(workspace.device_id);
        
        int pop_size = this->config->population_size;
        int total_chrom_len = this->config->get_total_chromosome_length();
        int elite_size = this->config->elite_size;
        int non_elite_size = pop_size - elite_size;
        int offspring_size = this->config->get_offspring_size();
        int mutant_size = this->config->mutant_size;
        
        // Allocate memory based on work distribution
        auto work_dist = gpu_manager.distribute_work(pop_size);
        size_t gpu_index = &workspace - &gpu_workspaces[0];
        size_t local_pop_size = (gpu_index < work_dist.size()) ? work_dist[gpu_index] : 0;
        
        if (local_pop_size > 0) {
            cudaMalloc(&workspace.d_population, local_pop_size * total_chrom_len * sizeof(T));
            cudaMalloc(&workspace.d_elite_pop, elite_size * total_chrom_len * sizeof(T));
            cudaMalloc(&workspace.d_non_elite_pop, non_elite_size * total_chrom_len * sizeof(T));
            cudaMalloc(&workspace.d_offspring, offspring_size * total_chrom_len * sizeof(T));
            cudaMalloc(&workspace.d_mutants, mutant_size * total_chrom_len * sizeof(T));
            cudaMalloc(&workspace.d_states, local_pop_size * sizeof(curandState));
            
            CudaUtils::check_cuda_error(cudaGetLastError(), 
                "Multi-GPU device memory allocation for GPU " + std::to_string(workspace.device_id));
            
            workspace.allocated = true;
        }
    }
    
    void cleanup_multi_gpu() {
        for (auto& workspace : gpu_workspaces) {
            workspace->cleanup();
        }
        gpu_workspaces.clear();
        multi_gpu_initialized = false;
    }
    
    void distribute_work_sizes(size_t total_elite, size_t total_non_elite, 
                              size_t total_offspring, size_t total_mutants) {
        elite_distribution = gpu_manager.distribute_work(total_elite);
        non_elite_distribution = gpu_manager.distribute_work(total_non_elite);
        offspring_distribution = gpu_manager.distribute_work(total_offspring);
        mutant_distribution = gpu_manager.distribute_work(total_mutants);
    }
    
public:
    // Multi-GPU crossover operation
    void multi_gpu_crossover(const std::vector<Individual<T>>& elite,
                            const std::vector<Individual<T>>& non_elite,
                            std::vector<Individual<T>>& offspring) override {
        
        if (!multi_gpu_initialized || gpu_workspaces.empty()) {
            // Fallback to single GPU
            this->crossover_device(elite, non_elite, offspring);
            return;
        }
        
        int total_chrom_len = this->config->get_total_chromosome_length();
        distribute_work_sizes(elite.size(), non_elite.size(), offspring.size(), 0);
        
        // Flatten populations
        auto elite_flat = this->flatten_population(elite);
        auto non_elite_flat = this->flatten_population(non_elite);
        
        // Launch crossover on multiple GPUs
        std::vector<std::future<void>> futures;
        size_t offspring_offset = 0;
        
        for (size_t gpu_idx = 0; gpu_idx < gpu_workspaces.size(); gpu_idx++) {
            auto& workspace = *gpu_workspaces[gpu_idx];
            size_t local_offspring_count = offspring_distribution[gpu_idx];
            
            if (local_offspring_count == 0) continue;
            
            futures.push_back(std::async(std::launch::async, [&, gpu_idx, offspring_offset, local_offspring_count]() {
                cudaSetDevice(workspace.device_id);
                
                // Copy elite and non-elite populations to this GPU
                cudaMemcpy(workspace.d_elite_pop, elite_flat.data(), 
                          elite.size() * total_chrom_len * sizeof(T), cudaMemcpyHostToDevice);
                cudaMemcpy(workspace.d_non_elite_pop, non_elite_flat.data(), 
                          non_elite.size() * total_chrom_len * sizeof(T), cudaMemcpyHostToDevice);
                
                // Launch crossover kernel for this GPU's portion
                dim3 block(this->config->threads_per_block);
                dim3 grid = CudaUtils::calculate_grid_size(local_offspring_count, this->config->threads_per_block);
                
                crossover_kernel<<<grid, block>>>(
                    workspace.d_elite_pop, workspace.d_non_elite_pop, workspace.d_offspring, workspace.d_states,
                    local_offspring_count, total_chrom_len, this->config->elite_prob, 
                    elite.size(), non_elite.size()
                );
                
                CudaUtils::sync_and_check("Multi-GPU crossover kernel on GPU " + std::to_string(workspace.device_id));
            }));
            
            offspring_offset += local_offspring_count;
        }
        
        // Wait for all GPUs to complete
        for (auto& future : futures) {
            future.wait();
        }
        
        // Gather results from all GPUs
        std::vector<T> offspring_flat(offspring.size() * total_chrom_len);
        offspring_offset = 0;
        
        for (size_t gpu_idx = 0; gpu_idx < gpu_workspaces.size(); gpu_idx++) {
            auto& workspace = *gpu_workspaces[gpu_idx];
            size_t local_offspring_count = offspring_distribution[gpu_idx];
            
            if (local_offspring_count == 0) continue;
            
            cudaSetDevice(workspace.device_id);
            cudaMemcpy(offspring_flat.data() + offspring_offset * total_chrom_len,
                      workspace.d_offspring, local_offspring_count * total_chrom_len * sizeof(T),
                      cudaMemcpyDeviceToHost);
            
            offspring_offset += local_offspring_count;
        }
        
        // Unflatten results
        this->unflatten_to_population(offspring_flat, offspring);
    }
    
    // Multi-GPU mutation operation
    void multi_gpu_mutation(std::vector<Individual<T>>& mutants) override {
        if (!multi_gpu_initialized || gpu_workspaces.empty()) {
            // Fallback to single GPU
            this->mutate_device(mutants);
            return;
        }
        
        if (mutants.empty()) return;
        
        int total_chrom_len = this->config->get_total_chromosome_length();
        distribute_work_sizes(0, 0, 0, mutants.size());
        
        // Launch mutation on multiple GPUs
        std::vector<std::future<void>> futures;
        size_t mutant_offset = 0;
        
        for (size_t gpu_idx = 0; gpu_idx < gpu_workspaces.size(); gpu_idx++) {
            auto& workspace = *gpu_workspaces[gpu_idx];
            size_t local_mutant_count = mutant_distribution[gpu_idx];
            
            if (local_mutant_count == 0) continue;
            
            futures.push_back(std::async(std::launch::async, [&, gpu_idx, local_mutant_count]() {
                cudaSetDevice(workspace.device_id);
                
                // Launch mutation kernel for this GPU's portion
                dim3 block(this->config->threads_per_block);
                dim3 grid = CudaUtils::calculate_grid_size(local_mutant_count, this->config->threads_per_block);
                
                mutation_kernel<<<grid, block>>>(
                    workspace.d_mutants, workspace.d_states, local_mutant_count, total_chrom_len
                );
                
                CudaUtils::sync_and_check("Multi-GPU mutation kernel on GPU " + std::to_string(workspace.device_id));
            }));
            
            mutant_offset += local_mutant_count;
        }
        
        // Wait for all GPUs to complete
        for (auto& future : futures) {
            future.wait();
        }
        
        // Gather results from all GPUs
        std::vector<T> mutants_flat(mutants.size() * total_chrom_len);
        mutant_offset = 0;
        
        for (size_t gpu_idx = 0; gpu_idx < gpu_workspaces.size(); gpu_idx++) {
            auto& workspace = *gpu_workspaces[gpu_idx];
            size_t local_mutant_count = mutant_distribution[gpu_idx];
            
            if (local_mutant_count == 0) continue;
            
            cudaSetDevice(workspace.device_id);
            cudaMemcpy(mutants_flat.data() + mutant_offset * total_chrom_len,
                      workspace.d_mutants, local_mutant_count * total_chrom_len * sizeof(T),
                      cudaMemcpyDeviceToHost);
            
            mutant_offset += local_mutant_count;
        }
        
        // Unflatten results
        this->unflatten_to_population(mutants_flat, mutants);
    }
    
    // Initialize random states on all GPUs
    void initialize_multi_gpu_random_states() {
        if (!multi_gpu_initialized) return;
        
        std::vector<std::future<void>> futures;
        
        for (size_t gpu_idx = 0; gpu_idx < gpu_workspaces.size(); gpu_idx++) {
            auto& workspace = *gpu_workspaces[gpu_idx];
            
            futures.push_back(std::async(std::launch::async, [&]() {
                cudaSetDevice(workspace.device_id);
                
                size_t local_pop_size = gpu_manager.distribute_work(this->config->population_size)[gpu_idx];
                if (local_pop_size == 0) return;
                
                dim3 block(this->config->threads_per_block);
                dim3 grid = CudaUtils::calculate_grid_size(local_pop_size, this->config->threads_per_block);
                
                int total_chrom_len = this->config->get_total_chromosome_length();
                
                initialize_population_kernel<<<grid, block>>>(
                    workspace.d_population, workspace.d_states, local_pop_size, total_chrom_len
                );
                
                CudaUtils::sync_and_check("Multi-GPU random state initialization on GPU " + 
                                        std::to_string(workspace.device_id));
            }));
        }
        
        // Wait for all GPUs to complete initialization
        for (auto& future : futures) {
            future.wait();
        }
        
        std::cout << "Multi-GPU random states initialized" << std::endl;
    }
    
    // Override adaptive methods to use multi-GPU when appropriate
    void adaptive_crossover(const std::vector<Individual<T>>& elite,
                           const std::vector<Individual<T>>& non_elite,
                           std::vector<Individual<T>>& offspring) override {
        
        const int multi_gpu_threshold = 1000; // Use multi-GPU for populations >= 1000
        const int single_gpu_threshold = 200; // Use GPU for populations >= 200
        
        if (multi_gpu_initialized && this->config->population_size >= multi_gpu_threshold) {
            multi_gpu_crossover(elite, non_elite, offspring);
        } else if (this->config->population_size >= single_gpu_threshold && this->is_device_available()) {
            this->crossover_device(elite, non_elite, offspring);
        } else {
            this->crossover_host(elite, non_elite, offspring);
        }
    }
    
    void adaptive_mutation(std::vector<Individual<T>>& mutants) override {
        const int multi_gpu_threshold = 1000;
        const int single_gpu_threshold = 200;
        
        if (multi_gpu_initialized && this->config->population_size >= multi_gpu_threshold) {
            multi_gpu_mutation(mutants);
        } else if (this->config->population_size >= single_gpu_threshold && this->is_device_available()) {
            this->mutate_device(mutants);
        } else {
            this->mutate_host(mutants);
        }
    }
    
    // Population evaluation across multiple GPUs
    void multi_gpu_evaluate_population(std::vector<Individual<T>>& population) {
        if (!multi_gpu_initialized || gpu_workspaces.empty()) {
            return; // Fallback to host-side evaluation
        }
        
        // Distribute population across GPUs for parallel evaluation
        auto work_distribution = gpu_manager.distribute_work(population.size());
        std::vector<std::future<void>> futures;
        
        size_t pop_offset = 0;
        for (size_t gpu_idx = 0; gpu_idx < gpu_workspaces.size(); gpu_idx++) {
            auto& workspace = *gpu_workspaces[gpu_idx];
            size_t local_pop_size = work_distribution[gpu_idx];
            
            if (local_pop_size == 0) continue;
            
            futures.push_back(std::async(std::launch::async, 
                [&, gpu_idx, pop_offset, local_pop_size]() {
                    cudaSetDevice(workspace.device_id);
                    
                    // Evaluate fitness for this GPU's portion of population
                    for (size_t i = pop_offset; i < pop_offset + local_pop_size; i++) {
                        if (!population[i].is_evaluated()) {
                            T fitness = this->config->fitness_function(population[i]);
                            population[i].set_fitness(fitness);
                        }
                    }
                }));
            
            pop_offset += local_pop_size;
        }
        
        // Wait for all evaluations to complete
        for (auto& future : futures) {
            future.wait();
        }
    }
    
    // Performance monitoring and optimization
    void benchmark_multi_gpu_performance() {
        if (!multi_gpu_initialized) {
            std::cout << "Multi-GPU not initialized, skipping benchmark." << std::endl;
            return;
        }
        
        std::cout << "\n=== Multi-GPU Performance Benchmark ===" << std::endl;
        
        // Test crossover performance
        auto test_elite = this->generate_test_population(this->config->elite_size);
        auto test_non_elite = this->generate_test_population(
            this->config->population_size - this->config->elite_size);
        auto test_offspring = this->generate_test_population(this->config->get_offspring_size());
        
        // Single GPU benchmark
        auto start_time = std::chrono::high_resolution_clock::now();
        this->crossover_device(test_elite, test_non_elite, test_offspring);
        auto single_gpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_time).count();
        
        // Multi-GPU benchmark
        start_time = std::chrono::high_resolution_clock::now();
        multi_gpu_crossover(test_elite, test_non_elite, test_offspring);
        auto multi_gpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_time).count();
        
        double speedup = static_cast<double>(single_gpu_time) / multi_gpu_time;
        
        std::cout << "Crossover Performance:" << std::endl;
        std::cout << "  Single GPU: " << single_gpu_time << " ms" << std::endl;
        std::cout << "  Multi-GPU:  " << multi_gpu_time << " ms" << std::endl;
        std::cout << "  Speedup:    " << std::fixed << std::setprecision(2) 
                  << speedup << "x" << std::endl;
        
        // Test mutation performance
        auto test_mutants = this->generate_test_population(this->config->mutant_size);
        
        start_time = std::chrono::high_resolution_clock::now();
        this->mutate_device(test_mutants);
        single_gpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_time).count();
        
        start_time = std::chrono::high_resolution_clock::now();
        multi_gpu_mutation(test_mutants);
        multi_gpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_time).count();
        
        speedup = static_cast<double>(single_gpu_time) / multi_gpu_time;
        
        std::cout << "Mutation Performance:" << std::endl;
        std::cout << "  Single GPU: " << single_gpu_time << " ms" << std::endl;
        std::cout << "  Multi-GPU:  " << multi_gpu_time << " ms" << std::endl;
        std::cout << "  Speedup:    " << std::fixed << std::setprecision(2) 
                  << speedup << "x" << std::endl;
        
        std::cout << "=======================================" << std::endl;
    }
    
    // Memory usage optimization
    void optimize_memory_usage() {
        if (!multi_gpu_initialized) return;
        
        std::cout << "\n=== Multi-GPU Memory Optimization ===" << std::endl;
        
        for (size_t gpu_idx = 0; gpu_idx < gpu_workspaces.size(); gpu_idx++) {
            auto& workspace = *gpu_workspaces[gpu_idx];
            cudaSetDevice(workspace.device_id);
            
            size_t free_mem, total_mem;
            cudaMemGetInfo(&free_mem, &total_mem);
            
            size_t used_mem = total_mem - free_mem;
            double usage_percent = (static_cast<double>(used_mem) / total_mem) * 100;
            
            std::cout << "GPU " << workspace.device_id << " Memory:" << std::endl;
            std::cout << "  Used:  " << (used_mem / (1024*1024)) << " MB" << std::endl;
            std::cout << "  Free:  " << (free_mem / (1024*1024)) << " MB" << std::endl;
            std::cout << "  Usage: " << std::fixed << std::setprecision(1) 
                      << usage_percent << "%" << std::endl;
            
            // Warn if memory usage is high
            if (usage_percent > 90) {
                std::cout << "  WARNING: High memory usage on GPU " 
                          << workspace.device_id << std::endl;
            }
        }
        
        std::cout << "=====================================" << std::endl;
    }
    
    // Load balancing analysis
    void analyze_load_balance() {
        if (!multi_gpu_initialized) return;
        
        std::cout << "\n=== Load Balance Analysis ===" << std::endl;
        
        auto work_dist = gpu_manager.distribute_work(this->config->population_size);
        auto weights = gpu_manager.get_performance_weights();
        
        for (size_t i = 0; i < gpu_workspaces.size(); i++) {
            double work_ratio = static_cast<double>(work_dist[i]) / this->config->population_size;
            double weight_ratio = weights[i] / std::accumulate(weights.begin(), weights.end(), 0.0);
            
            std::cout << "GPU " << gpu_workspaces[i]->device_id << ":" << std::endl;
            std::cout << "  Work assigned: " << work_dist[i] << " (" 
                      << std::fixed << std::setprecision(1) << work_ratio * 100 << "%)" << std::endl;
            std::cout << "  Performance weight: " << std::setprecision(2) << weights[i] << "x" << std::endl;
            std::cout << "  Expected work ratio: " << std::setprecision(1) 
                      << weight_ratio * 100 << "%" << std::endl;
            
            double balance_factor = work_ratio / weight_ratio;
            std::cout << "  Balance factor: " << std::setprecision(2) << balance_factor;
            if (balance_factor < 0.9 || balance_factor > 1.1) {
                std::cout << " (IMBALANCED)";
            }
            std::cout << std::endl;
        }
        
        std::cout << "==============================" << std::endl;
    }
    
    // Getters and status
    bool is_multi_gpu_enabled() const { return multi_gpu_initialized; }
    size_t get_active_gpu_count() const { return gpu_workspaces.size(); }
    
    void print_multi_gpu_status() const {
        std::cout << "\n=== Multi-GPU Genetic Operators Status ===" << std::endl;
        std::cout << "Multi-GPU enabled: " << (multi_gpu_initialized ? "Yes" : "No") << std::endl;
        
        if (multi_gpu_initialized) {
            std::cout << "Active GPUs: " << gpu_workspaces.size() << std::endl;
            for (size_t i = 0; i < gpu_workspaces.size(); i++) {
                std::cout << "  GPU " << gpu_workspaces[i]->device_id 
                          << " (allocated: " << (gpu_workspaces[i]->allocated ? "Yes" : "No") << ")" << std::endl;
            }
        }
        std::cout << "===========================================" << std::endl;
    }

private:
    // Helper method to generate test populations for benchmarking
    std::vector<Individual<T>> generate_test_population(size_t size) {
        std::vector<Individual<T>> test_pop;
        test_pop.reserve(size);
        
        std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
        
        for (size_t i = 0; i < size; i++) {
            Individual<T> individual(this->config->component_lengths);
            individual.randomize(rng);
            test_pop.push_back(individual);
        }
        
        return test_pop;
    }
};

// Global instance
MultiGPUManager g_multi_gpu_manager;

#endif // MULTI_GPU_GENETIC_OPERATORS_HPP