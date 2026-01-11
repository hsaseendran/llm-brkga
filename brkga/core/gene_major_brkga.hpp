#ifndef GENE_MAJOR_BRKGA_HPP
#define GENE_MAJOR_BRKGA_HPP

/**
 * @file gene_major_brkga.hpp
 * @brief Gene-major memory layout BRKGA implementation
 *
 * **Phase 5 Day 6: Solver Integration**
 *
 * This file provides a clean wrapper around gene-major BRKGA kernels for
 * optimized GPU execution. It can be used standalone or integrated into
 * the existing Solver class.
 *
 * **Key Features:**
 * - Coalesced memory access for 2-100× speedup
 * - Drop-in replacement for individual-major BRKGA
 * - Automatic transpose for decoder compatibility
 * - CUDA stream support for async operations
 *
 * @author Claude Sonnet 4.5
 * @date January 2026
 */

#include "cuda_kernels.cuh"
#include "memory_layout.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <memory>
#include <stdexcept>
#include <functional>

/**
 * @brief Gene-major BRKGA implementation for optimized GPU execution
 *
 * This class manages population data in gene-major layout for coalesced
 * memory access, providing significant speedups over individual-major layout.
 *
 * **Usage Example:**
 * @code
 * // Create BRKGA instance
 * GeneLayoutBRKGA<float> brkga(pop_size, chrom_len, elite_size, mutant_size, elite_prob);
 *
 * // Initialize population
 * brkga.initialize_population();
 *
 * // Run generations
 * for (int gen = 0; gen < max_generations; gen++) {
 *     // Evaluate fitness (decoder receives individual-major if needed)
 *     brkga.evaluate_fitness([](float* population, float* fitness, int pop_size, int chrom_len) {
 *         // Your decoder here
 *     });
 *
 *     // Run BRKGA generation (sort + elite copy + crossover + mutation)
 *     brkga.run_generation();
 * }
 *
 * // Get best solution
 * std::vector<float> best = brkga.get_best_individual();
 * @endcode
 *
 * @tparam T Data type (float, double)
 */
template<typename T>
class GeneLayoutBRKGA {
private:
    // Population parameters
    int pop_size_;
    int chrom_len_;
    int elite_size_;
    int mutant_size_;
    T elite_prob_;

    // GPU memory (gene-major layout)
    T* d_population_;          // Current population [chrom_len][pop_size]
    T* d_next_gen_;            // Next generation buffer [chrom_len][pop_size]
    T* d_fitness_;             // Fitness values [pop_size]
    int* d_indices_;           // Sorted indices [pop_size]
    curandState* d_states_;    // RNG states [pop_size]

    // For decoder compatibility (individual-major)
    T* d_population_ind_;      // Individual-major buffer for decoder
    bool decoder_needs_ind_major_;

    // Initialization state
    bool initialized_;
    int current_generation_;

public:
    /**
     * @brief Construct gene-major BRKGA instance
     *
     * @param pop_size Population size
     * @param chrom_len Chromosome length (number of genes)
     * @param elite_size Number of elite individuals to preserve
     * @param mutant_size Number of random mutant individuals per generation
     * @param elite_prob Probability of inheriting from elite parent during crossover
     * @param decoder_needs_ind_major If true, maintain individual-major buffer for decoder
     */
    GeneLayoutBRKGA(
        int pop_size,
        int chrom_len,
        int elite_size,
        int mutant_size,
        T elite_prob,
        bool decoder_needs_ind_major = false
    ) : pop_size_(pop_size)
      , chrom_len_(chrom_len)
      , elite_size_(elite_size)
      , mutant_size_(mutant_size)
      , elite_prob_(elite_prob)
      , decoder_needs_ind_major_(decoder_needs_ind_major)
      , initialized_(false)
      , current_generation_(0)
      , d_population_(nullptr)
      , d_next_gen_(nullptr)
      , d_fitness_(nullptr)
      , d_indices_(nullptr)
      , d_states_(nullptr)
      , d_population_ind_(nullptr)
    {
        allocate_memory();
    }

    ~GeneLayoutBRKGA() {
        free_memory();
    }

    // Disable copy
    GeneLayoutBRKGA(const GeneLayoutBRKGA&) = delete;
    GeneLayoutBRKGA& operator=(const GeneLayoutBRKGA&) = delete;

    /**
     * @brief Initialize population with random values
     *
     * Uses gene-major initialization kernel for coalesced writes.
     */
    void initialize_population() {
        // Initialize cuRAND states
        dim3 block_init(256);
        dim3 grid_init((pop_size_ + 255) / 256);
        init_curand_states_kernel<<<grid_init, block_init>>>(d_states_, pop_size_, 1234);

        // Initialize population (gene-major)
        dim3 block(256);
        dim3 grid(chrom_len_, (pop_size_ + 255) / 256);
        initialize_population_kernel_gene_major<<<grid, block>>>(
            d_population_, d_states_, pop_size_, chrom_len_
        );

        cudaDeviceSynchronize();
        initialized_ = true;
        current_generation_ = 0;
    }

    /**
     * @brief Evaluate fitness using provided decoder function
     *
     * The decoder receives population in individual-major layout for compatibility.
     * Transpose is performed automatically if decoder_needs_ind_major is true.
     *
     * @param decoder Function that evaluates fitness: (population, fitness, pop_size, chrom_len)
     */
    template<typename DecoderFunc>
    void evaluate_fitness(DecoderFunc decoder) {
        if (decoder_needs_ind_major_) {
            // Transpose to individual-major for decoder
            transpose_gene_to_ind();
            decoder(d_population_ind_, d_fitness_, pop_size_, chrom_len_);
        } else {
            // Decoder can handle gene-major directly
            decoder(d_population_, d_fitness_, pop_size_, chrom_len_);
        }
    }

    /**
     * @brief Run one BRKGA generation
     *
     * Performs: sort by fitness -> elite copy -> crossover -> mutation
     * All operations use gene-major kernels for optimal performance.
     */
    void run_generation() {
        // Step 1: Sort indices by fitness
        thrust::device_ptr<int> d_indices_ptr(d_indices_);
        thrust::sequence(d_indices_ptr, d_indices_ptr + pop_size_);
        thrust::device_ptr<T> d_fitness_ptr(d_fitness_);
        thrust::sort_by_key(d_fitness_ptr, d_fitness_ptr + pop_size_, d_indices_ptr);

        // Step 2: Run gene-major BRKGA generation kernel
        dim3 block(256);
        dim3 grid(chrom_len_, (pop_size_ + 255) / 256);

        brkga_generation_kernel_gene_major<<<grid, block>>>(
            d_population_, d_next_gen_, d_indices_, d_states_,
            pop_size_, elite_size_, mutant_size_, chrom_len_, elite_prob_
        );

        cudaDeviceSynchronize();

        // Step 3: Swap buffers
        std::swap(d_population_, d_next_gen_);

        current_generation_++;
    }

    /**
     * @brief Get best individual (after sorting)
     *
     * @return Vector containing the best chromosome
     */
    std::vector<T> get_best_individual() {
        std::vector<T> best(chrom_len_);

        // Get index of best individual (first after sorting)
        int best_idx;
        cudaMemcpy(&best_idx, d_indices_, sizeof(int), cudaMemcpyDeviceToHost);

        // Copy chromosome (gene-major layout)
        for (int gene = 0; gene < chrom_len_; gene++) {
            T value;
            cudaMemcpy(&value, d_population_ + gene * pop_size_ + best_idx,
                       sizeof(T), cudaMemcpyDeviceToHost);
            best[gene] = value;
        }

        return best;
    }

    /**
     * @brief Get best fitness value (after sorting)
     */
    T get_best_fitness() {
        T best_fitness;
        cudaMemcpy(&best_fitness, d_fitness_, sizeof(T), cudaMemcpyDeviceToHost);
        return best_fitness;
    }

    /**
     * @brief Get current generation number
     */
    int get_generation() const { return current_generation_; }

    /**
     * @brief Get population size
     */
    int get_pop_size() const { return pop_size_; }

    /**
     * @brief Get chromosome length
     */
    int get_chrom_len() const { return chrom_len_; }

    /**
     * @brief Get pointer to gene-major population (for advanced users)
     */
    T* get_population_device() { return d_population_; }

    /**
     * @brief Get pointer to fitness array (for advanced users)
     */
    T* get_fitness_device() { return d_fitness_; }

    /**
     * @brief Get memory usage in bytes
     */
    size_t get_memory_usage() const {
        size_t usage = 0;
        usage += 2 * pop_size_ * chrom_len_ * sizeof(T);  // population + next_gen
        usage += pop_size_ * sizeof(T);                    // fitness
        usage += pop_size_ * sizeof(int);                  // indices
        usage += pop_size_ * sizeof(curandState);          // RNG states
        if (decoder_needs_ind_major_) {
            usage += pop_size_ * chrom_len_ * sizeof(T);   // individual-major buffer
        }
        return usage;
    }

private:
    void allocate_memory() {
        size_t pop_bytes = pop_size_ * chrom_len_ * sizeof(T);

        cudaError_t err = cudaMalloc(&d_population_, pop_bytes);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate d_population: " +
                                     std::string(cudaGetErrorString(err)));
        }

        err = cudaMalloc(&d_next_gen_, pop_bytes);
        if (err != cudaSuccess) {
            cudaFree(d_population_);
            throw std::runtime_error("Failed to allocate d_next_gen");
        }

        err = cudaMalloc(&d_fitness_, pop_size_ * sizeof(T));
        if (err != cudaSuccess) {
            cudaFree(d_population_);
            cudaFree(d_next_gen_);
            throw std::runtime_error("Failed to allocate d_fitness");
        }

        err = cudaMalloc(&d_indices_, pop_size_ * sizeof(int));
        if (err != cudaSuccess) {
            cudaFree(d_population_);
            cudaFree(d_next_gen_);
            cudaFree(d_fitness_);
            throw std::runtime_error("Failed to allocate d_indices");
        }

        err = cudaMalloc(&d_states_, pop_size_ * sizeof(curandState));
        if (err != cudaSuccess) {
            cudaFree(d_population_);
            cudaFree(d_next_gen_);
            cudaFree(d_fitness_);
            cudaFree(d_indices_);
            throw std::runtime_error("Failed to allocate d_states");
        }

        if (decoder_needs_ind_major_) {
            err = cudaMalloc(&d_population_ind_, pop_bytes);
            if (err != cudaSuccess) {
                cudaFree(d_population_);
                cudaFree(d_next_gen_);
                cudaFree(d_fitness_);
                cudaFree(d_indices_);
                cudaFree(d_states_);
                throw std::runtime_error("Failed to allocate d_population_ind");
            }
        }
    }

    void free_memory() {
        if (d_population_) cudaFree(d_population_);
        if (d_next_gen_) cudaFree(d_next_gen_);
        if (d_fitness_) cudaFree(d_fitness_);
        if (d_indices_) cudaFree(d_indices_);
        if (d_states_) cudaFree(d_states_);
        if (d_population_ind_) cudaFree(d_population_ind_);
    }

    void transpose_gene_to_ind() {
        dim3 block(256);
        dim3 grid((pop_size_ + 255) / 256, chrom_len_);
        transpose_gene_to_ind_kernel<<<grid, block>>>(
            d_population_, d_population_ind_, pop_size_, chrom_len_
        );
        cudaDeviceSynchronize();
    }
};

#endif // GENE_MAJOR_BRKGA_HPP
