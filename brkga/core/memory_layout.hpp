#ifndef MEMORY_LAYOUT_HPP
#define MEMORY_LAYOUT_HPP

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

/**
 * @file memory_layout.hpp
 * @brief Memory layout abstractions for optimized GPU memory access
 *
 * This file provides infrastructure for transforming between individual-major
 * and gene-major memory layouts to enable coalesced memory access patterns.
 *
 * **Phase 5: Coalesced Memory Access Optimization**
 *
 * @author Claude Sonnet 4.5
 * @date January 2026
 */

/**
 * @brief Memory layout strategies for population storage
 *
 * Individual-major: [ind0_gene0, ind0_gene1, ..., ind0_geneN, ind1_gene0, ...]
 *   - Current default layout
 *   - Poor coalescing: adjacent threads access memory locations chrom_len apart
 *   - Memory bandwidth utilization: ~10-15%
 *
 * Gene-major: [ind0_gene0, ind1_gene0, ..., indN_gene0, ind0_gene1, ind1_gene1, ...]
 *   - Optimized layout for GPU kernels
 *   - Perfect coalescing: adjacent threads access consecutive memory locations
 *   - Memory bandwidth utilization: ~90% (theoretical)
 *   - Expected speedup: 5-15× for memory-bound kernels
 */
enum class MemoryLayout {
    INDIVIDUAL_MAJOR,  ///< Row-major: [pop_size][chrom_len]
    GENE_MAJOR         ///< Column-major: [chrom_len][pop_size]
};

/**
 * @brief Transpose kernel: Individual-major → Gene-major
 *
 * Transforms population from individual-major to gene-major layout:
 *   Input:  [ind0_g0, ind0_g1, ..., ind0_gN, ind1_g0, ind1_g1, ...]
 *   Output: [ind0_g0, ind1_g0, ..., indN_g0, ind0_g1, ind1_g1, ...]
 *
 * Launch configuration:
 *   Grid:  ((pop_size + 255) / 256, chrom_len)
 *   Block: (256)
 *
 * Memory access pattern:
 *   Read:  Scattered within block (each thread reads different individual)
 *   Write: COALESCED (adjacent threads write consecutive locations)
 *
 * @tparam T Data type (float, double, int, etc.)
 * @param src Source array in individual-major layout [pop_size * chrom_len]
 * @param dest Destination array in gene-major layout [pop_size * chrom_len]
 * @param pop_size Number of individuals (population size)
 * @param chrom_len Chromosome length (number of genes)
 */
template<typename T>
__global__ void transpose_ind_to_gene_kernel(
    const T* __restrict__ src,  // Individual-major: [pop_size][chrom_len]
    T* __restrict__ dest,       // Gene-major: [chrom_len][pop_size]
    int pop_size,
    int chrom_len
) {
    int ind_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int gene_idx = blockIdx.y;

    if (ind_idx < pop_size) {
        // Read: src[ind_idx * chrom_len + gene_idx] (strided read within block)
        // Write: dest[gene_idx * pop_size + ind_idx] (coalesced write!)
        T value = src[ind_idx * chrom_len + gene_idx];
        dest[gene_idx * pop_size + ind_idx] = value;
    }
}

/**
 * @brief Transpose kernel: Gene-major → Individual-major
 *
 * Transforms population from gene-major back to individual-major layout:
 *   Input:  [ind0_g0, ind1_g0, ..., indN_g0, ind0_g1, ind1_g1, ...]
 *   Output: [ind0_g0, ind0_g1, ..., ind0_gN, ind1_g0, ind1_g1, ...]
 *
 * Launch configuration:
 *   Grid:  ((pop_size + 255) / 256, chrom_len)
 *   Block: (256)
 *
 * Memory access pattern:
 *   Read:  COALESCED (adjacent threads read consecutive locations)
 *   Write: Scattered (each thread writes to different individual)
 *
 * @tparam T Data type (float, double, int, etc.)
 * @param src Source array in gene-major layout [pop_size * chrom_len]
 * @param dest Destination array in individual-major layout [pop_size * chrom_len]
 * @param pop_size Number of individuals (population size)
 * @param chrom_len Chromosome length (number of genes)
 */
template<typename T>
__global__ void transpose_gene_to_ind_kernel(
    const T* __restrict__ src,  // Gene-major: [chrom_len][pop_size]
    T* __restrict__ dest,       // Individual-major: [pop_size][chrom_len]
    int pop_size,
    int chrom_len
) {
    int ind_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int gene_idx = blockIdx.y;

    if (ind_idx < pop_size) {
        // Read: src[gene_idx * pop_size + ind_idx] (coalesced read!)
        // Write: dest[ind_idx * chrom_len + gene_idx] (strided write)
        T value = src[gene_idx * pop_size + ind_idx];
        dest[ind_idx * chrom_len + gene_idx] = value;
    }
}

/**
 * @brief Helper class for managing dual-layout buffers
 *
 * Maintains two copies of population data in different memory layouts:
 * 1. Individual-major (original/default layout)
 * 2. Gene-major (optimized layout)
 *
 * Provides transparent conversion between layouts on demand.
 *
 * **Memory overhead:** 2× population size (one copy per layout)
 * **Use case:** When kernels need different layouts, or when interfacing
 *              with external code that expects specific layout
 *
 * **Example usage:**
 * @code
 * // Create buffer for 1000 individuals × 500 genes
 * DualLayoutBuffer<float> buffer(1000, 500);
 *
 * // Initialize in individual-major layout
 * initialize_population_kernel<<<...>>>(buffer.get_individual_major(), ...);
 *
 * // Convert to gene-major for optimized BRKGA operations
 * buffer.transpose_to_gene_major();
 *
 * // Run gene-major optimized kernel
 * brkga_generation_kernel_gene_major<<<...>>>(buffer.get_gene_major(), ...);
 *
 * // Convert back for fitness evaluation (if decoder needs individual-major)
 * buffer.transpose_to_individual_major();
 * evaluate_fitness<<<...>>>(buffer.get_individual_major(), ...);
 * @endcode
 *
 * @tparam T Data type (float, double, int, etc.)
 */
template<typename T>
class DualLayoutBuffer {
private:
    T* d_individual_major_;  ///< Individual-major layout buffer
    T* d_gene_major_;        ///< Gene-major layout buffer
    int pop_size_;           ///< Population size
    int chrom_len_;          ///< Chromosome length
    MemoryLayout current_layout_;  ///< Current active layout

public:
    /**
     * @brief Construct dual-layout buffer
     *
     * Allocates two GPU memory buffers of size pop_size × chrom_len.
     *
     * @param pop_size Number of individuals
     * @param chrom_len Number of genes per individual
     * @throws std::runtime_error if CUDA allocation fails
     */
    DualLayoutBuffer(int pop_size, int chrom_len)
        : pop_size_(pop_size)
        , chrom_len_(chrom_len)
        , current_layout_(MemoryLayout::INDIVIDUAL_MAJOR)
        , d_individual_major_(nullptr)
        , d_gene_major_(nullptr)
    {
        size_t bytes = pop_size * chrom_len * sizeof(T);

        cudaError_t err = cudaMalloc(&d_individual_major_, bytes);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("DualLayoutBuffer: Failed to allocate individual-major buffer: ") +
                cudaGetErrorString(err)
            );
        }

        err = cudaMalloc(&d_gene_major_, bytes);
        if (err != cudaSuccess) {
            cudaFree(d_individual_major_);  // Clean up first allocation
            throw std::runtime_error(
                std::string("DualLayoutBuffer: Failed to allocate gene-major buffer: ") +
                cudaGetErrorString(err)
            );
        }
    }

    /**
     * @brief Destructor - frees GPU memory
     */
    ~DualLayoutBuffer() {
        if (d_individual_major_) cudaFree(d_individual_major_);
        if (d_gene_major_) cudaFree(d_gene_major_);
    }

    // Delete copy constructor/assignment (managing GPU pointers)
    DualLayoutBuffer(const DualLayoutBuffer&) = delete;
    DualLayoutBuffer& operator=(const DualLayoutBuffer&) = delete;

    /**
     * @brief Get pointer to individual-major buffer
     * @return Device pointer to individual-major layout
     */
    T* get_individual_major() { return d_individual_major_; }

    /**
     * @brief Get pointer to gene-major buffer
     * @return Device pointer to gene-major layout
     */
    T* get_gene_major() { return d_gene_major_; }

    /**
     * @brief Get pointer to individual-major buffer (const version)
     */
    const T* get_individual_major() const { return d_individual_major_; }

    /**
     * @brief Get pointer to gene-major buffer (const version)
     */
    const T* get_gene_major() const { return d_gene_major_; }

    /**
     * @brief Transpose from individual-major to gene-major layout
     *
     * Converts current individual-major data to gene-major layout.
     * If already in gene-major layout, does nothing (idempotent).
     *
     * @param stream CUDA stream for asynchronous execution (default: 0)
     */
    void transpose_to_gene_major(cudaStream_t stream = 0) {
        if (current_layout_ == MemoryLayout::GENE_MAJOR) {
            return;  // Already in gene-major layout
        }

        dim3 block(256);
        dim3 grid((pop_size_ + 255) / 256, chrom_len_);

        transpose_ind_to_gene_kernel<<<grid, block, 0, stream>>>(
            d_individual_major_, d_gene_major_, pop_size_, chrom_len_
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("DualLayoutBuffer: Transpose to gene-major failed: ") +
                cudaGetErrorString(err)
            );
        }

        current_layout_ = MemoryLayout::GENE_MAJOR;
    }

    /**
     * @brief Transpose from gene-major to individual-major layout
     *
     * Converts current gene-major data to individual-major layout.
     * If already in individual-major layout, does nothing (idempotent).
     *
     * @param stream CUDA stream for asynchronous execution (default: 0)
     */
    void transpose_to_individual_major(cudaStream_t stream = 0) {
        if (current_layout_ == MemoryLayout::INDIVIDUAL_MAJOR) {
            return;  // Already in individual-major layout
        }

        dim3 block(256);
        dim3 grid((pop_size_ + 255) / 256, chrom_len_);

        transpose_gene_to_ind_kernel<<<grid, block, 0, stream>>>(
            d_gene_major_, d_individual_major_, pop_size_, chrom_len_
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("DualLayoutBuffer: Transpose to individual-major failed: ") +
                cudaGetErrorString(err)
            );
        }

        current_layout_ = MemoryLayout::INDIVIDUAL_MAJOR;
    }

    /**
     * @brief Get current active layout
     * @return Current memory layout (INDIVIDUAL_MAJOR or GENE_MAJOR)
     */
    MemoryLayout get_current_layout() const {
        return current_layout_;
    }

    /**
     * @brief Get population size
     */
    int get_pop_size() const { return pop_size_; }

    /**
     * @brief Get chromosome length
     */
    int get_chrom_len() const { return chrom_len_; }

    /**
     * @brief Get total number of elements
     */
    int get_total_elements() const { return pop_size_ * chrom_len_; }

    /**
     * @brief Get memory usage in bytes
     */
    size_t get_memory_usage() const {
        return 2 * pop_size_ * chrom_len_ * sizeof(T);  // 2× for dual buffers
    }
};

#endif // MEMORY_LAYOUT_HPP
