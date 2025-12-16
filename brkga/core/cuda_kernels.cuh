#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdexcept>
#include <string>

// CUDA kernel to initialize cuRAND states only
__global__ void init_curand_states_kernel(curandState* states, int num_states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_states) {
        curand_init(seed + idx, idx, 0, &states[idx]);
    }
}

// CUDA kernel for population initialization
template<typename T>
__global__ void initialize_population_kernel(T* population, curandState* states,
                                           int pop_size, int chrom_length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pop_size) {
        curand_init(1234 + idx, 0, 0, &states[idx]);
        for (int i = 0; i < chrom_length; i++) {
            population[idx * chrom_length + i] = curand_uniform(&states[idx]);
        }
    }
}

// CUDA kernel for crossover operation
template<typename T>
__global__ void crossover_kernel(T* elite_pop, T* non_elite_pop, T* offspring, 
                               curandState* states, int num_offspring, int chrom_length, 
                               double elite_prob, int elite_size, int non_elite_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_offspring) {
        curandState localState = states[idx];
        
        int elite_parent = curand(&localState) % elite_size;
        int non_elite_parent = curand(&localState) % non_elite_size;
        
        for (int i = 0; i < chrom_length; i++) {
            if (curand_uniform(&localState) < elite_prob) {
                offspring[idx * chrom_length + i] = elite_pop[elite_parent * chrom_length + i];
            } else {
                offspring[idx * chrom_length + i] = non_elite_pop[non_elite_parent * chrom_length + i];
            }
        }
        
        states[idx] = localState;
    }
}

// CUDA kernel for mutation operation
template<typename T>
__global__ void mutation_kernel(T* population, curandState* states,
                              int num_mutants, int chrom_length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_mutants) {
        curandState localState = states[idx];
        
        for (int i = 0; i < chrom_length; i++) {
            population[idx * chrom_length + i] = curand_uniform(&localState);
        }
        
        states[idx] = localState;
    }
}

// CUDA kernel for fitness evaluation (template for device-side evaluation)
template<typename T>
__global__ void evaluate_fitness_kernel(T* population, T* fitness_values,
                                       int pop_size, int chrom_length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pop_size) {
        // This is a template - actual implementation should be provided
        // by problem-specific configurations if device-side evaluation is desired
        fitness_values[idx] = T(0);
    }
}

// =============================================================================
// GPU-Resident BRKGA Kernels
// =============================================================================

// Combined kernel: Elite copy + Crossover + Mutation in one pass
// Population is sorted by fitness, so indices 0..elite_size-1 are elite
template<typename T>
__global__ void brkga_generation_kernel(
    T* population,           // Current population (sorted by fitness)
    T* next_gen,             // Output: next generation
    int* sorted_indices,     // Indices after sorting (to map elite)
    curandState* states,
    int pop_size,
    int elite_size,
    int mutant_size,
    int chrom_length,
    float elite_prob
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    curandState localState = states[idx];
    int offspring_start = elite_size;
    int mutant_start = pop_size - mutant_size;

    if (idx < elite_size) {
        // Copy elite individual (already sorted, so idx 0..elite_size-1 are best)
        int src_idx = sorted_indices[idx];
        for (int i = 0; i < chrom_length; i++) {
            next_gen[idx * chrom_length + i] = population[src_idx * chrom_length + i];
        }
    }
    else if (idx >= mutant_start) {
        // Generate mutant (random individual)
        for (int i = 0; i < chrom_length; i++) {
            next_gen[idx * chrom_length + i] = curand_uniform(&localState);
        }
    }
    else {
        // Crossover: pick elite and non-elite parents
        int elite_parent = curand(&localState) % elite_size;
        int non_elite_parent = elite_size + (curand(&localState) % (pop_size - elite_size));

        int elite_src = sorted_indices[elite_parent];
        int non_elite_src = sorted_indices[non_elite_parent];

        for (int i = 0; i < chrom_length; i++) {
            if (curand_uniform(&localState) < elite_prob) {
                next_gen[idx * chrom_length + i] = population[elite_src * chrom_length + i];
            } else {
                next_gen[idx * chrom_length + i] = population[non_elite_src * chrom_length + i];
            }
        }
    }

    states[idx] = localState;
}

// Reorder population based on sorted indices (for in-place update after sort)
template<typename T>
__global__ void reorder_population_kernel(
    T* src,
    T* dst,
    int* indices,
    int pop_size,
    int chrom_length
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    int src_idx = indices[idx];
    for (int i = 0; i < chrom_length; i++) {
        dst[idx * chrom_length + i] = src[src_idx * chrom_length + i];
    }
}

// =============================================================================
// Random 2-opt Local Search Kernel
// =============================================================================
// Applies a single random 2-opt move to the city portion of the chromosome.
// Only modifies elite individuals to avoid wasting compute on bad solutions.
// The 2-opt reverses a random segment of the chromosome keys, which changes
// the decoded city tour order.

template<typename T>
__global__ void random_2opt_kernel(
    T* population,           // Population chromosomes
    T* backup,               // Backup for restoration if move is bad
    T* fitness,              // Current fitness values
    T* new_fitness,          // New fitness after 2-opt (computed externally)
    curandState* states,
    int pop_size,
    int num_cities,          // Number of cities (first half of chromosome)
    int chrom_length,        // Total chromosome length
    int num_to_improve,      // Number of individuals to apply local search to
    int num_moves            // Number of random 2-opt moves to apply per individual
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_to_improve) return;

    curandState localState = states[idx];
    int offset = idx * chrom_length;

    // Backup the city portion of chromosome
    for (int c = 0; c < num_cities; c++) {
        backup[offset + c] = population[offset + c];
    }

    // Apply multiple random 2-opt moves
    for (int move = 0; move < num_moves; move++) {
        // Randomly select two positions i and j in the city chromosome (0 to num_cities-1)
        int i = curand(&localState) % num_cities;
        int j = curand(&localState) % num_cities;

        // Ensure i < j
        if (i > j) {
            int temp = i;
            i = j;
            j = temp;
        }

        // If i == j or adjacent, skip this move
        if (j - i < 2) {
            continue;
        }

        // Reverse the segment [i, j] in the city keys
        // This effectively reverses that portion of the decoded tour
        while (i < j) {
            T temp = population[offset + i];
            population[offset + i] = population[offset + j];
            population[offset + j] = temp;
            i++;
            j--;
        }
    }

    states[idx] = localState;
}

// Restore chromosomes where 2-opt didn't improve fitness
template<typename T>
__global__ void restore_if_worse_kernel(
    T* population,
    T* backup,
    T* old_fitness,
    T* new_fitness,
    int num_cities,
    int chrom_length,
    int num_to_check
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_to_check) return;

    // If new fitness is worse (higher for minimization), restore from backup
    if (new_fitness[idx] >= old_fitness[idx]) {
        int offset = idx * chrom_length;
        for (int i = 0; i < num_cities; i++) {
            population[offset + i] = backup[offset + i];
        }
    } else {
        // Keep the improvement - update the fitness
        old_fitness[idx] = new_fitness[idx];
    }
}

// CUDA utility functions
namespace CudaUtils {
    // Check CUDA errors
    inline void check_cuda_error(cudaError_t error, const char* message) {
        if (error != cudaSuccess) {
            throw std::runtime_error(std::string(message) + ": " + 
                                   std::string(cudaGetErrorString(error)));
        }
    }
    
    // Synchronize device and check for errors
    inline void sync_and_check(const char* operation) {
        cudaError_t error = cudaDeviceSynchronize();
        check_cuda_error(error, operation);
    }
    
    // Calculate grid dimensions
    inline dim3 calculate_grid_size(int total_threads, int threads_per_block) {
        return dim3((total_threads + threads_per_block - 1) / threads_per_block);
    }
}

#endif // CUDA_KERNELS_CUH