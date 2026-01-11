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

// CUDA kernel for population initialization (INDIVIDUAL-MAJOR LAYOUT)
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

/**
 * @brief Population initialization kernel (GENE-MAJOR LAYOUT)
 *
 * **Phase 5 Optimization:** Coalesced memory access through gene-major layout
 *
 * Grid:  (chrom_length, (pop_size + 255) / 256)
 * Block: (256)
 *
 * Memory access pattern:
 *   - COALESCED writes: Adjacent threads write to consecutive memory locations
 *   - Expected speedup: 5-8× over individual-major version
 *
 * **Note:** This version uses a different RNG initialization strategy than individual-major.
 * Each (individual, gene) pair gets a unique random stream, ensuring statistical quality
 * while enabling parallel generation. The random values are statistically equivalent
 * but not bit-identical to the individual-major version.
 *
 * @tparam T Data type (float, double)
 * @param population Output population array in gene-major layout [chrom_length][pop_size]
 * @param states Temporary RNG states (not used, for API compatibility)
 * @param pop_size Number of individuals
 * @param chrom_length Number of genes per individual
 */
template<typename T>
__global__ void initialize_population_kernel_gene_major(
    T* population,          // Gene-major: [chrom_length][pop_size]
    curandState* states,    // RNG states [pop_size] (unused in this version)
    int pop_size,
    int chrom_length
) {
    int gene_idx = blockIdx.x;
    int ind_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (ind_idx < pop_size) {
        // Initialize RNG with unique seed for (individual, gene) pair
        // This ensures each position gets an independent random stream
        curandState local_state;
        curand_init(1234 + ind_idx * chrom_length + gene_idx, 0, 0, &local_state);

        // COALESCED write: population[gene_idx * pop_size + ind_idx]
        // Adjacent threads (ind_idx, ind_idx+1, ...) write to consecutive memory
        int offset = gene_idx * pop_size + ind_idx;
        population[offset] = curand_uniform(&local_state);
    }
}

// CUDA kernel for crossover operation (INDIVIDUAL-MAJOR LAYOUT)
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

/**
 * @brief Crossover operation kernel (GENE-MAJOR LAYOUT)
 *
 * **Phase 5 Optimization:** Coalesced memory access through gene-major layout
 *
 * Grid:  (chrom_length, (num_offspring + 255) / 256)
 * Block: (256)
 *
 * Memory access pattern:
 *   - COALESCED reads from elite and non-elite populations
 *   - COALESCED writes to offspring
 *   - Expected speedup: 5-10× over individual-major version
 *
 * **Strategy:** Each thread processes one gene for one offspring. Parent selection
 * is deterministically recomputed for each gene using the same RNG seed, ensuring
 * all genes of an offspring come from the same parents.
 *
 * @tparam T Data type (float, double)
 * @param elite_pop Elite population in gene-major layout [chrom_length][elite_size]
 * @param non_elite_pop Non-elite population in gene-major layout [chrom_length][non_elite_size]
 * @param offspring Output offspring in gene-major layout [chrom_length][num_offspring]
 * @param states RNG states [num_offspring] (not modified, for deterministic parent selection)
 * @param num_offspring Number of offspring to generate
 * @param chrom_length Number of genes per individual
 * @param elite_prob Probability of inheriting from elite parent
 * @param elite_size Number of elite individuals
 * @param non_elite_size Number of non-elite individuals
 */
template<typename T>
__global__ void crossover_kernel_gene_major(
    const T* __restrict__ elite_pop,      // Gene-major: [chrom_length][elite_size]
    const T* __restrict__ non_elite_pop,  // Gene-major: [chrom_length][non_elite_size]
    T* __restrict__ offspring,            // Gene-major: [chrom_length][num_offspring]
    curandState* states,                  // RNG states [num_offspring]
    int num_offspring,
    int chrom_length,
    T elite_prob,
    int elite_size,
    int non_elite_size
) {
    int gene_idx = blockIdx.x;
    int offspring_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (offspring_idx >= num_offspring) return;

    // Deterministically select parents using RNG seed
    // All genes of this offspring must use the same parents
    curandState local_state;
    curand_init(1234 + offspring_idx, 0, 0, &local_state);

    // Select elite and non-elite parents (same for all genes)
    int elite_parent = curand(&local_state) % elite_size;
    int non_elite_parent = curand(&local_state) % non_elite_size;

    // Advance RNG to gene position for inheritance decision
    for (int g = 0; g < gene_idx; g++) {
        curand_uniform(&local_state);  // Skip ahead
    }

    // Decide which parent this gene comes from
    T inherit_from_elite = curand_uniform(&local_state);

    // COALESCED reads and writes
    int offspring_offset = gene_idx * num_offspring + offspring_idx;

    if (inherit_from_elite < elite_prob) {
        // Inherit from elite parent - COALESCED read
        int elite_offset = gene_idx * elite_size + elite_parent;
        offspring[offspring_offset] = elite_pop[elite_offset];
    } else {
        // Inherit from non-elite parent - COALESCED read
        int non_elite_offset = gene_idx * non_elite_size + non_elite_parent;
        offspring[offspring_offset] = non_elite_pop[non_elite_offset];
    }
}

// CUDA kernel for mutation operation (INDIVIDUAL-MAJOR LAYOUT)
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

/**
 * @brief Mutation operation kernel (GENE-MAJOR LAYOUT)
 *
 * **Phase 5 Optimization:** Coalesced memory access through gene-major layout
 *
 * Grid:  (chrom_length, (num_mutants + 255) / 256)
 * Block: (256)
 *
 * Memory access pattern:
 *   - COALESCED writes: Adjacent threads write to consecutive memory locations
 *   - Expected speedup: 5-10× over individual-major version
 *
 * **Note:** Similar to initialize_population_kernel_gene_major, this generates
 * new random chromosomes for mutant individuals.
 *
 * @tparam T Data type (float, double)
 * @param population Output mutant population in gene-major layout [chrom_length][num_mutants]
 * @param states RNG states [num_mutants] (unused, for API compatibility)
 * @param num_mutants Number of mutant individuals to generate
 * @param chrom_length Number of genes per individual
 */
template<typename T>
__global__ void mutation_kernel_gene_major(
    T* population,          // Gene-major: [chrom_length][num_mutants]
    curandState* states,    // RNG states (unused in this version)
    int num_mutants,
    int chrom_length
) {
    int gene_idx = blockIdx.x;
    int mutant_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (mutant_idx < num_mutants) {
        // Initialize RNG with unique seed for (mutant, gene) pair
        curandState local_state;
        curand_init(5678 + mutant_idx * chrom_length + gene_idx, 0, 0, &local_state);

        // COALESCED write: population[gene_idx * num_mutants + mutant_idx]
        int offset = gene_idx * num_mutants + mutant_idx;
        population[offset] = curand_uniform(&local_state);
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

/**
 * @brief Combined BRKGA generation kernel (GENE-MAJOR LAYOUT)
 *
 * **Phase 5 Optimization:** Coalesced memory access through gene-major layout
 *
 * This kernel combines elite copying, crossover, and mutation in a single pass.
 * Each thread processes one gene for one individual.
 *
 * Grid:  (chrom_length, (pop_size + 255) / 256)
 * Block: (256)
 *
 * Memory access pattern:
 *   - COALESCED reads from population (elite copy, crossover)
 *   - COALESCED writes to next_gen
 *   - Expected speedup: 5-15× over individual-major version
 *
 * Population layout after sorting:
 *   - Indices 0..(elite_size-1): Elite individuals (best fitness)
 *   - Indices elite_size..(pop_size-mutant_size-1): Offspring from crossover
 *   - Indices (pop_size-mutant_size)..(pop_size-1): Mutant individuals
 *
 * @tparam T Data type (float, double)
 * @param population Current population in gene-major layout [chrom_length][pop_size]
 * @param next_gen Output next generation in gene-major layout [chrom_length][pop_size]
 * @param sorted_indices Mapping from new to old indices after fitness sorting [pop_size]
 * @param states RNG states [pop_size] (unused in gene-major version)
 * @param pop_size Total population size
 * @param elite_size Number of elite individuals to preserve
 * @param mutant_size Number of mutant individuals to generate
 * @param chrom_length Number of genes per individual
 * @param elite_prob Probability of inheriting from elite parent during crossover
 */
template<typename T>
__global__ void brkga_generation_kernel_gene_major(
    const T* __restrict__ population,  // Gene-major: [chrom_length][pop_size]
    T* __restrict__ next_gen,          // Gene-major: [chrom_length][pop_size]
    const int* __restrict__ sorted_indices,
    curandState* states,               // Unused in gene-major version
    int pop_size,
    int elite_size,
    int mutant_size,
    int chrom_length,
    T elite_prob
) {
    int gene_idx = blockIdx.x;
    int ind_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (ind_idx >= pop_size) return;

    int mutant_start = pop_size - mutant_size;
    int output_offset = gene_idx * pop_size + ind_idx;

    if (ind_idx < elite_size) {
        // ELITE COPY: Copy from sorted elite individual
        int src_ind = sorted_indices[ind_idx];
        int src_offset = gene_idx * pop_size + src_ind;
        next_gen[output_offset] = population[src_offset];
    }
    else if (ind_idx >= mutant_start) {
        // MUTANT: Generate random value
        curandState local_state;
        curand_init(5678 + ind_idx * chrom_length + gene_idx, 0, 0, &local_state);
        next_gen[output_offset] = curand_uniform(&local_state);
    }
    else {
        // CROSSOVER: Inherit from elite or non-elite parent
        // Deterministic parent selection using RNG seed
        curandState local_state;
        curand_init(1234 + ind_idx, 0, 0, &local_state);

        // Select parents (same for all genes of this individual)
        int elite_parent = curand(&local_state) % elite_size;
        int non_elite_parent = elite_size + (curand(&local_state) % (pop_size - elite_size));

        // Advance RNG to gene position for inheritance decision
        for (int g = 0; g < gene_idx; g++) {
            curand_uniform(&local_state);
        }

        // Get source indices from sorted population
        int elite_src = sorted_indices[elite_parent];
        int non_elite_src = sorted_indices[non_elite_parent];

        // Decide inheritance for this gene
        T inherit_from_elite = curand_uniform(&local_state);

        if (inherit_from_elite < elite_prob) {
            int src_offset = gene_idx * pop_size + elite_src;
            next_gen[output_offset] = population[src_offset];
        } else {
            int src_offset = gene_idx * pop_size + non_elite_src;
            next_gen[output_offset] = population[src_offset];
        }
    }
}

// Reorder population based on sorted indices (for in-place update after sort)
// INDIVIDUAL-MAJOR LAYOUT
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

/**
 * @brief Reorder population kernel (GENE-MAJOR LAYOUT)
 *
 * **Phase 5 Optimization:** Coalesced memory access through gene-major layout
 *
 * Reorders population based on sorted indices (typically from fitness sorting).
 * Each thread processes one gene for one individual.
 *
 * Grid:  (chrom_length, (pop_size + 255) / 256)
 * Block: (256)
 *
 * Memory access pattern:
 *   - COALESCED reads from src (adjacent threads read consecutive locations)
 *   - COALESCED writes to dst (adjacent threads write consecutive locations)
 *   - Expected speedup: 3-8× over individual-major version
 *
 * @tparam T Data type (float, double)
 * @param src Source population in gene-major layout [chrom_length][pop_size]
 * @param dst Destination population in gene-major layout [chrom_length][pop_size]
 * @param indices Mapping from destination to source indices [pop_size]
 * @param pop_size Number of individuals
 * @param chrom_length Number of genes per individual
 */
template<typename T>
__global__ void reorder_population_kernel_gene_major(
    const T* __restrict__ src,
    T* __restrict__ dst,
    const int* __restrict__ indices,
    int pop_size,
    int chrom_length
) {
    int gene_idx = blockIdx.x;
    int dst_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (dst_idx >= pop_size) return;

    // Get source individual index
    int src_idx = indices[dst_idx];

    // COALESCED read and write
    int src_offset = gene_idx * pop_size + src_idx;
    int dst_offset = gene_idx * pop_size + dst_idx;

    dst[dst_offset] = src[src_offset];
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