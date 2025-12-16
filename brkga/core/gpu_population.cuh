// core/gpu_population.cuh - GPU-resident population for zero-copy evolution
#ifndef GPU_POPULATION_CUH
#define GPU_POPULATION_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/functional.h>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <string>

#include "individual.hpp"
#include "config.hpp"

// Forward declare kernels
template<typename T>
__global__ void gpu_pop_initialize_kernel(T* chromosomes, curandState* states,
                                          int pop_size, int chrom_len, unsigned long seed);

template<typename T>
__global__ void gpu_pop_crossover_kernel(T* chromosomes, curandState* states,
                                         int elite_size, int non_elite_end, int offspring_start,
                                         int offspring_count, int chrom_len, T elite_prob);

template<typename T>
__global__ void gpu_pop_mutation_kernel(T* chromosomes, curandState* states,
                                        int mutant_start, int mutant_count, int chrom_len);

template<typename T>
__global__ void gpu_pop_gather_kernel(T* dest, const T* src, const int* indices,
                                      int pop_size, int chrom_len);

// Comparator for thrust sorting (minimization by default)
template<typename T>
struct FitnessComparator {
    bool minimize;

    __host__ __device__
    FitnessComparator(bool min = true) : minimize(min) {}

    __host__ __device__
    bool operator()(const T& a, const T& b) const {
        return minimize ? (a < b) : (a > b);
    }
};

template<typename T>
class GPUPopulation {
private:
    // Device memory - main population data
    T* d_chromosomes;           // [pop_size * chrom_len]
    T* d_chromosomes_temp;      // [pop_size * chrom_len] - for gather operations
    T* d_fitness;               // [pop_size]
    T* d_fitness_temp;          // [pop_size] - for sorting
    int* d_indices;             // [pop_size] - sorting indices
    curandState* d_states;      // [pop_size] - RNG states

    // Multi-objective support
    T* d_objectives;            // [pop_size * num_objectives]
    int* d_ranks;               // [pop_size]
    T* d_crowding;              // [pop_size]

    // Configuration
    int pop_size;
    int chrom_len;
    int elite_size;
    int offspring_size;
    int mutant_size;
    int num_objectives;
    bool minimize;              // True for minimization, false for maximization

    int threads_per_block;
    bool allocated;
    bool initialized;

public:
    GPUPopulation(int population_size, int chromosome_length, int n_elite,
                  int n_offspring, int n_mutants, int n_objectives = 1,
                  bool minimization = true, int tpb = 256)
        : pop_size(population_size), chrom_len(chromosome_length),
          elite_size(n_elite), offspring_size(n_offspring), mutant_size(n_mutants),
          num_objectives(n_objectives), minimize(minimization),
          threads_per_block(tpb), allocated(false), initialized(false),
          d_chromosomes(nullptr), d_chromosomes_temp(nullptr),
          d_fitness(nullptr), d_fitness_temp(nullptr),
          d_indices(nullptr), d_states(nullptr),
          d_objectives(nullptr), d_ranks(nullptr), d_crowding(nullptr) {

        allocate();
    }

    ~GPUPopulation() {
        cleanup();
    }

    // Disable copy
    GPUPopulation(const GPUPopulation&) = delete;
    GPUPopulation& operator=(const GPUPopulation&) = delete;

    // Enable move
    GPUPopulation(GPUPopulation&& other) noexcept {
        *this = std::move(other);
    }

    GPUPopulation& operator=(GPUPopulation&& other) noexcept {
        if (this != &other) {
            cleanup();
            d_chromosomes = other.d_chromosomes;
            d_chromosomes_temp = other.d_chromosomes_temp;
            d_fitness = other.d_fitness;
            d_fitness_temp = other.d_fitness_temp;
            d_indices = other.d_indices;
            d_states = other.d_states;
            d_objectives = other.d_objectives;
            d_ranks = other.d_ranks;
            d_crowding = other.d_crowding;
            pop_size = other.pop_size;
            chrom_len = other.chrom_len;
            elite_size = other.elite_size;
            offspring_size = other.offspring_size;
            mutant_size = other.mutant_size;
            num_objectives = other.num_objectives;
            minimize = other.minimize;
            threads_per_block = other.threads_per_block;
            allocated = other.allocated;
            initialized = other.initialized;

            other.d_chromosomes = nullptr;
            other.d_chromosomes_temp = nullptr;
            other.d_fitness = nullptr;
            other.d_fitness_temp = nullptr;
            other.d_indices = nullptr;
            other.d_states = nullptr;
            other.d_objectives = nullptr;
            other.d_ranks = nullptr;
            other.d_crowding = nullptr;
            other.allocated = false;
        }
        return *this;
    }

    void allocate() {
        if (allocated) return;

        cudaError_t err;

        // Main buffers
        err = cudaMalloc(&d_chromosomes, pop_size * chrom_len * sizeof(T));
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate d_chromosomes");

        err = cudaMalloc(&d_chromosomes_temp, pop_size * chrom_len * sizeof(T));
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate d_chromosomes_temp");

        err = cudaMalloc(&d_fitness, pop_size * sizeof(T));
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate d_fitness");

        err = cudaMalloc(&d_fitness_temp, pop_size * sizeof(T));
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate d_fitness_temp");

        err = cudaMalloc(&d_indices, pop_size * sizeof(int));
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate d_indices");

        err = cudaMalloc(&d_states, pop_size * sizeof(curandState));
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate d_states");

        // Multi-objective buffers
        if (num_objectives > 1) {
            err = cudaMalloc(&d_objectives, pop_size * num_objectives * sizeof(T));
            if (err != cudaSuccess) throw std::runtime_error("Failed to allocate d_objectives");

            err = cudaMalloc(&d_ranks, pop_size * sizeof(int));
            if (err != cudaSuccess) throw std::runtime_error("Failed to allocate d_ranks");

            err = cudaMalloc(&d_crowding, pop_size * sizeof(T));
            if (err != cudaSuccess) throw std::runtime_error("Failed to allocate d_crowding");
        }

        allocated = true;
    }

    void cleanup() {
        if (!allocated) return;

        if (d_chromosomes) cudaFree(d_chromosomes);
        if (d_chromosomes_temp) cudaFree(d_chromosomes_temp);
        if (d_fitness) cudaFree(d_fitness);
        if (d_fitness_temp) cudaFree(d_fitness_temp);
        if (d_indices) cudaFree(d_indices);
        if (d_states) cudaFree(d_states);
        if (d_objectives) cudaFree(d_objectives);
        if (d_ranks) cudaFree(d_ranks);
        if (d_crowding) cudaFree(d_crowding);

        d_chromosomes = nullptr;
        d_chromosomes_temp = nullptr;
        d_fitness = nullptr;
        d_fitness_temp = nullptr;
        d_indices = nullptr;
        d_states = nullptr;
        d_objectives = nullptr;
        d_ranks = nullptr;
        d_crowding = nullptr;

        allocated = false;
    }

    // Initialize population with random values and set up RNG states
    void initialize(unsigned long seed = 1234) {
        if (!allocated) {
            throw std::runtime_error("GPUPopulation not allocated");
        }

        dim3 block(threads_per_block);
        dim3 grid((pop_size + block.x - 1) / block.x);

        gpu_pop_initialize_kernel<<<grid, block>>>(
            d_chromosomes, d_states, pop_size, chrom_len, seed
        );

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to initialize GPU population: " +
                                    std::string(cudaGetErrorString(err)));
        }

        initialized = true;
    }

    // Selection: sort population by fitness (best first)
    // After this, positions [0, elite_size) are elite, [elite_size, pop_size-mutant_size) are non-elite
    void select() {
        if (!initialized) {
            throw std::runtime_error("GPUPopulation not initialized");
        }

        // 1. Initialize index array: [0, 1, 2, ..., pop_size-1]
        thrust::device_ptr<int> indices_ptr(d_indices);
        thrust::sequence(indices_ptr, indices_ptr + pop_size);

        // 2. Copy fitness to temp for sorting (preserve original order for gather)
        cudaMemcpy(d_fitness_temp, d_fitness, pop_size * sizeof(T), cudaMemcpyDeviceToDevice);

        // 3. Sort indices by fitness
        thrust::device_ptr<T> fitness_ptr(d_fitness_temp);

        if (minimize) {
            thrust::sort_by_key(fitness_ptr, fitness_ptr + pop_size, indices_ptr,
                               thrust::less<T>());
        } else {
            thrust::sort_by_key(fitness_ptr, fitness_ptr + pop_size, indices_ptr,
                               thrust::greater<T>());
        }

        // 4. Gather chromosomes into sorted order
        dim3 block(threads_per_block);
        dim3 grid((pop_size + block.x - 1) / block.x);

        gpu_pop_gather_kernel<<<grid, block>>>(
            d_chromosomes_temp, d_chromosomes, d_indices, pop_size, chrom_len
        );
        cudaDeviceSynchronize();

        // 5. Swap buffers (sorted becomes main)
        std::swap(d_chromosomes, d_chromosomes_temp);

        // 6. Also reorder fitness values
        thrust::device_ptr<T> fitness_src(d_fitness);
        thrust::device_ptr<T> fitness_dst(d_fitness_temp);
        thrust::gather(indices_ptr, indices_ptr + pop_size, fitness_src, fitness_dst);
        std::swap(d_fitness, d_fitness_temp);
    }

    // Crossover: create offspring from elite and non-elite parents
    // Elite: [0, elite_size)
    // Non-elite: [elite_size, pop_size - mutant_size)
    // Offspring written to: [elite_size, elite_size + offspring_size)
    void crossover(T elite_prob) {
        if (!initialized) {
            throw std::runtime_error("GPUPopulation not initialized");
        }

        int non_elite_end = pop_size - mutant_size;
        int offspring_start = elite_size;  // Offspring overwrite non-elite positions

        dim3 block(threads_per_block);
        dim3 grid((offspring_size + block.x - 1) / block.x);

        gpu_pop_crossover_kernel<<<grid, block>>>(
            d_chromosomes, d_states,
            elite_size, non_elite_end, offspring_start,
            offspring_size, chrom_len, elite_prob
        );

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            throw std::runtime_error("Crossover kernel failed: " +
                                    std::string(cudaGetErrorString(err)));
        }
    }

    // Mutation: generate new random chromosomes for mutant positions
    // Mutants written to: [pop_size - mutant_size, pop_size)
    void mutate() {
        if (!initialized) {
            throw std::runtime_error("GPUPopulation not initialized");
        }

        if (mutant_size <= 0) return;

        int mutant_start = pop_size - mutant_size;

        dim3 block(threads_per_block);
        dim3 grid((mutant_size + block.x - 1) / block.x);

        gpu_pop_mutation_kernel<<<grid, block>>>(
            d_chromosomes, d_states, mutant_start, mutant_size, chrom_len
        );

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            throw std::runtime_error("Mutation kernel failed: " +
                                    std::string(cudaGetErrorString(err)));
        }
    }

    // Evaluate fitness using config's GPU evaluation
    void evaluate(BRKGAConfig<T>* config) {
        if (!initialized) {
            throw std::runtime_error("GPUPopulation not initialized");
        }

        if (config->has_gpu_evaluation()) {
            config->evaluate_population_gpu(d_chromosomes, d_fitness, pop_size, chrom_len);
        } else {
            // Fallback: copy to host, evaluate, copy back
            evaluate_cpu_fallback(config);
        }
    }

    // Get device pointers for external access
    T* get_d_chromosomes() { return d_chromosomes; }
    T* get_d_fitness() { return d_fitness; }
    T* get_d_objectives() { return d_objectives; }
    int* get_d_indices() { return d_indices; }
    curandState* get_d_states() { return d_states; }

    const T* get_d_chromosomes() const { return d_chromosomes; }
    const T* get_d_fitness() const { return d_fitness; }

    // ========================================
    // Host synchronization methods
    // ========================================

    // Get best individual (copies only one individual to host)
    Individual<T> get_best(const std::vector<int>& component_lengths) const {
        Individual<T> best(component_lengths);

        // Copy first chromosome (best after selection)
        std::vector<T> chrom_data(chrom_len);
        cudaMemcpy(chrom_data.data(), d_chromosomes, chrom_len * sizeof(T), cudaMemcpyDeviceToHost);
        best.unflatten(chrom_data);

        // Copy fitness
        T fitness;
        cudaMemcpy(&fitness, d_fitness, sizeof(T), cudaMemcpyDeviceToHost);
        best.set_fitness(fitness);

        return best;
    }

    // Get elite individuals (copies elite_size individuals)
    std::vector<Individual<T>> get_elite(const std::vector<int>& component_lengths) const {
        std::vector<Individual<T>> elite;
        elite.reserve(elite_size);

        std::vector<T> chrom_data(elite_size * chrom_len);
        cudaMemcpy(chrom_data.data(), d_chromosomes, elite_size * chrom_len * sizeof(T),
                   cudaMemcpyDeviceToHost);

        std::vector<T> fitness_data(elite_size);
        cudaMemcpy(fitness_data.data(), d_fitness, elite_size * sizeof(T),
                   cudaMemcpyDeviceToHost);

        for (int i = 0; i < elite_size; i++) {
            Individual<T> ind(component_lengths);
            std::vector<T> ind_data(chrom_data.begin() + i * chrom_len,
                                   chrom_data.begin() + (i + 1) * chrom_len);
            ind.unflatten(ind_data);
            ind.set_fitness(fitness_data[i]);
            elite.push_back(ind);
        }

        return elite;
    }

    // Sync entire population to host (expensive - use sparingly)
    void sync_to_host(std::vector<Individual<T>>& population,
                     const std::vector<int>& component_lengths) const {
        if (population.size() != static_cast<size_t>(pop_size)) {
            population.resize(pop_size, Individual<T>(component_lengths));
        }

        std::vector<T> chrom_data(pop_size * chrom_len);
        cudaMemcpy(chrom_data.data(), d_chromosomes, pop_size * chrom_len * sizeof(T),
                   cudaMemcpyDeviceToHost);

        std::vector<T> fitness_data(pop_size);
        cudaMemcpy(fitness_data.data(), d_fitness, pop_size * sizeof(T),
                   cudaMemcpyDeviceToHost);

        for (int i = 0; i < pop_size; i++) {
            std::vector<T> ind_data(chrom_data.begin() + i * chrom_len,
                                   chrom_data.begin() + (i + 1) * chrom_len);
            population[i].unflatten(ind_data);
            population[i].set_fitness(fitness_data[i]);
        }
    }

    // Sync host population to device (used for local search results)
    void sync_from_host(const std::vector<Individual<T>>& population) {
        if (population.size() != static_cast<size_t>(pop_size)) {
            throw std::runtime_error("Population size mismatch in sync_from_host");
        }

        std::vector<T> chrom_data;
        chrom_data.reserve(pop_size * chrom_len);

        std::vector<T> fitness_data(pop_size);

        for (int i = 0; i < pop_size; i++) {
            auto flat = population[i].flatten();
            chrom_data.insert(chrom_data.end(), flat.begin(), flat.end());
            fitness_data[i] = population[i].fitness;
        }

        cudaMemcpy(d_chromosomes, chrom_data.data(), pop_size * chrom_len * sizeof(T),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_fitness, fitness_data.data(), pop_size * sizeof(T),
                   cudaMemcpyHostToDevice);
    }

    // Sync only specific indices from host (after local search on selected individuals)
    void sync_individuals_from_host(const std::vector<Individual<T>>& individuals,
                                    const std::vector<int>& indices) {
        for (size_t i = 0; i < indices.size(); i++) {
            int idx = indices[i];
            if (idx < 0 || idx >= pop_size) continue;

            auto flat = individuals[i].flatten();
            cudaMemcpy(d_chromosomes + idx * chrom_len, flat.data(),
                      chrom_len * sizeof(T), cudaMemcpyHostToDevice);

            T fitness = individuals[i].fitness;
            cudaMemcpy(d_fitness + idx, &fitness, sizeof(T), cudaMemcpyHostToDevice);
        }
    }

    // Get fitness of specific individual
    T get_fitness(int index) const {
        T fitness;
        cudaMemcpy(&fitness, d_fitness + index, sizeof(T), cudaMemcpyDeviceToHost);
        return fitness;
    }

    // Get best fitness (first position after selection)
    T get_best_fitness() const {
        return get_fitness(0);
    }

    // Configuration getters
    int get_pop_size() const { return pop_size; }
    int get_chrom_len() const { return chrom_len; }
    int get_elite_size() const { return elite_size; }
    int get_offspring_size() const { return offspring_size; }
    int get_mutant_size() const { return mutant_size; }
    bool is_allocated() const { return allocated; }
    bool is_initialized() const { return initialized; }

private:
    // CPU fallback for evaluation (when config doesn't have GPU evaluation)
    void evaluate_cpu_fallback(BRKGAConfig<T>* config) {
        // Copy to host
        std::vector<T> chrom_data(pop_size * chrom_len);
        cudaMemcpy(chrom_data.data(), d_chromosomes, pop_size * chrom_len * sizeof(T),
                   cudaMemcpyDeviceToHost);

        // Evaluate on CPU
        std::vector<T> fitness_data(pop_size);

        #pragma omp parallel for if(pop_size > 100)
        for (int i = 0; i < pop_size; i++) {
            Individual<T> ind(config->component_lengths);
            std::vector<T> ind_data(chrom_data.begin() + i * chrom_len,
                                   chrom_data.begin() + (i + 1) * chrom_len);
            ind.unflatten(ind_data);
            fitness_data[i] = config->fitness_function(ind);
        }

        // Copy fitness back
        cudaMemcpy(d_fitness, fitness_data.data(), pop_size * sizeof(T),
                   cudaMemcpyHostToDevice);
    }
};

// ========================================
// GPU Kernels
// ========================================

template<typename T>
__global__ void gpu_pop_initialize_kernel(T* chromosomes, curandState* states,
                                          int pop_size, int chrom_len, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    // Initialize RNG state
    curand_init(seed + idx, 0, 0, &states[idx]);

    // Generate random chromosome
    T* my_chrom = chromosomes + idx * chrom_len;
    for (int i = 0; i < chrom_len; i++) {
        my_chrom[i] = curand_uniform(&states[idx]);
    }
}

template<typename T>
__global__ void gpu_pop_crossover_kernel(T* chromosomes, curandState* states,
                                         int elite_size, int non_elite_end, int offspring_start,
                                         int offspring_count, int chrom_len, T elite_prob) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= offspring_count) return;

    curandState localState = states[idx];

    // Select random elite and non-elite parents
    int elite_parent = curand(&localState) % elite_size;
    int non_elite_size = non_elite_end - elite_size;
    int non_elite_parent = elite_size + (curand(&localState) % non_elite_size);

    // Pointers to parent and offspring chromosomes
    T* elite_chrom = chromosomes + elite_parent * chrom_len;
    T* non_elite_chrom = chromosomes + non_elite_parent * chrom_len;
    T* offspring_chrom = chromosomes + (offspring_start + idx) * chrom_len;

    // Biased crossover
    for (int i = 0; i < chrom_len; i++) {
        if (curand_uniform(&localState) < elite_prob) {
            offspring_chrom[i] = elite_chrom[i];
        } else {
            offspring_chrom[i] = non_elite_chrom[i];
        }
    }

    states[idx] = localState;
}

template<typename T>
__global__ void gpu_pop_mutation_kernel(T* chromosomes, curandState* states,
                                        int mutant_start, int mutant_count, int chrom_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= mutant_count) return;

    curandState localState = states[mutant_start + idx];

    // Generate new random chromosome
    T* my_chrom = chromosomes + (mutant_start + idx) * chrom_len;
    for (int i = 0; i < chrom_len; i++) {
        my_chrom[i] = curand_uniform(&localState);
    }

    states[mutant_start + idx] = localState;
}

template<typename T>
__global__ void gpu_pop_gather_kernel(T* dest, const T* src, const int* indices,
                                      int pop_size, int chrom_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    int src_idx = indices[idx];

    const T* src_chrom = src + src_idx * chrom_len;
    T* dest_chrom = dest + idx * chrom_len;

    for (int i = 0; i < chrom_len; i++) {
        dest_chrom[i] = src_chrom[i];
    }
}

#endif // GPU_POPULATION_CUH
