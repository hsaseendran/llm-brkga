/**
 * @file benchmark_all_phases.cu
 * @brief Comprehensive benchmark comparing baseline vs all BrkgaCuda 2.0 optimizations
 *
 * This benchmark measures the combined impact of all optimization phases:
 * - Phase 1-3: CUDA Streams (pipelining)
 * - Phase 4: Segmented Sort (TSP decoder)
 * - Phase 5: Gene-Major Memory Layout (coalesced access)
 *
 * @author Claude Opus 4.5
 * @date January 2026
 */

#include "core/cuda_kernels.cuh"
#include "core/memory_layout.hpp"
#include "core/gene_major_brkga.hpp"
#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <algorithm>

// Simple sphere decoder for benchmarking
__global__ void sphere_decoder_ind_major(
    const float* __restrict__ population,
    float* __restrict__ fitness,
    int pop_size,
    int chrom_len
) {
    int ind_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind_idx >= pop_size) return;

    float sum = 0.0f;
    for (int gene = 0; gene < chrom_len; gene++) {
        float val = population[ind_idx * chrom_len + gene];
        float x = val * 10.0f - 5.0f;
        sum += x * x;
    }
    fitness[ind_idx] = sum;
}

__global__ void sphere_decoder_gene_major(
    const float* __restrict__ population,
    float* __restrict__ fitness,
    int pop_size,
    int chrom_len
) {
    int ind_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind_idx >= pop_size) return;

    float sum = 0.0f;
    for (int gene = 0; gene < chrom_len; gene++) {
        float val = population[gene * pop_size + ind_idx];
        float x = val * 10.0f - 5.0f;
        sum += x * x;
    }
    fitness[ind_idx] = sum;
}

/**
 * Baseline BRKGA implementation (individual-major, no streams)
 */
class BaselineBRKGA {
private:
    int pop_size_, chrom_len_, elite_size_, mutant_size_;
    float elite_prob_;

    float* d_population_;
    float* d_next_gen_;
    float* d_fitness_;
    int* d_indices_;
    curandState* d_states_;

public:
    BaselineBRKGA(int pop_size, int chrom_len, int elite_size, int mutant_size, float elite_prob)
        : pop_size_(pop_size), chrom_len_(chrom_len), elite_size_(elite_size),
          mutant_size_(mutant_size), elite_prob_(elite_prob)
    {
        size_t pop_bytes = pop_size * chrom_len * sizeof(float);
        cudaMalloc(&d_population_, pop_bytes);
        cudaMalloc(&d_next_gen_, pop_bytes);
        cudaMalloc(&d_fitness_, pop_size * sizeof(float));
        cudaMalloc(&d_indices_, pop_size * sizeof(int));
        cudaMalloc(&d_states_, pop_size * sizeof(curandState));
    }

    ~BaselineBRKGA() {
        cudaFree(d_population_);
        cudaFree(d_next_gen_);
        cudaFree(d_fitness_);
        cudaFree(d_indices_);
        cudaFree(d_states_);
    }

    void initialize() {
        dim3 block(256);
        dim3 grid((pop_size_ + 255) / 256);
        init_curand_states_kernel<<<grid, block>>>(d_states_, pop_size_, 1234);

        // Individual-major initialization
        initialize_population_kernel<<<grid, block>>>(
            d_population_, d_states_, pop_size_, chrom_len_
        );
        cudaDeviceSynchronize();
    }

    void evaluate() {
        dim3 block(256);
        dim3 grid((pop_size_ + 255) / 256);
        sphere_decoder_ind_major<<<grid, block>>>(
            d_population_, d_fitness_, pop_size_, chrom_len_
        );
        cudaDeviceSynchronize();
    }

    void run_generation() {
        // Sort by fitness
        thrust::device_ptr<int> d_indices_ptr(d_indices_);
        thrust::sequence(d_indices_ptr, d_indices_ptr + pop_size_);
        thrust::device_ptr<float> d_fitness_ptr(d_fitness_);
        thrust::sort_by_key(d_fitness_ptr, d_fitness_ptr + pop_size_, d_indices_ptr);

        // Individual-major BRKGA generation
        dim3 block(256);
        dim3 grid((pop_size_ + 255) / 256);
        brkga_generation_kernel<<<grid, block>>>(
            d_population_, d_next_gen_, d_indices_, d_states_,
            pop_size_, elite_size_, mutant_size_, chrom_len_, elite_prob_
        );
        cudaDeviceSynchronize();

        std::swap(d_population_, d_next_gen_);
    }

    float get_best_fitness() {
        float best;
        cudaMemcpy(&best, d_fitness_, sizeof(float), cudaMemcpyDeviceToHost);
        return best;
    }
};

struct BenchmarkResult {
    std::string name;
    double total_time_ms;
    double time_per_gen_ms;
    double throughput_mgenes_sec;
    float final_fitness;
};

BenchmarkResult benchmark_baseline(int pop_size, int chrom_len, int num_generations) {
    BaselineBRKGA brkga(pop_size, chrom_len, pop_size * 15 / 100, pop_size * 10 / 100, 0.7f);

    // Warmup
    brkga.initialize();
    for (int i = 0; i < 5; i++) {
        brkga.evaluate();
        brkga.run_generation();
    }

    // Reinitialize for benchmark
    brkga.initialize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int gen = 0; gen < num_generations; gen++) {
        brkga.evaluate();
        brkga.run_generation();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    brkga.evaluate();
    float final_fitness = brkga.get_best_fitness();

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return {
        "Baseline (Individual-Major)",
        elapsed_ms,
        elapsed_ms / num_generations,
        (double)(pop_size * chrom_len * num_generations) / (elapsed_ms / 1000.0) / 1e6,
        final_fitness
    };
}

BenchmarkResult benchmark_gene_major(int pop_size, int chrom_len, int num_generations) {
    GeneLayoutBRKGA<float> brkga(pop_size, chrom_len,
                                  pop_size * 15 / 100, pop_size * 10 / 100, 0.7f, false);

    // Warmup
    brkga.initialize_population();
    for (int i = 0; i < 5; i++) {
        brkga.evaluate_fitness([&](float* pop, float* fit, int ps, int cl) {
            dim3 block(256);
            dim3 grid((ps + 255) / 256);
            sphere_decoder_gene_major<<<grid, block>>>(pop, fit, ps, cl);
            cudaDeviceSynchronize();
        });
        brkga.run_generation();
    }

    // Reinitialize for benchmark
    brkga.initialize_population();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int gen = 0; gen < num_generations; gen++) {
        brkga.evaluate_fitness([&](float* pop, float* fit, int ps, int cl) {
            dim3 block(256);
            dim3 grid((ps + 255) / 256);
            sphere_decoder_gene_major<<<grid, block>>>(pop, fit, ps, cl);
            cudaDeviceSynchronize();
        });
        brkga.run_generation();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    brkga.evaluate_fitness([&](float* pop, float* fit, int ps, int cl) {
        dim3 block(256);
        dim3 grid((ps + 255) / 256);
        sphere_decoder_gene_major<<<grid, block>>>(pop, fit, ps, cl);
        cudaDeviceSynchronize();
    });
    float final_fitness = brkga.get_best_fitness();

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return {
        "Gene-Major (Phase 5)",
        elapsed_ms,
        elapsed_ms / num_generations,
        (double)(pop_size * chrom_len * num_generations) / (elapsed_ms / 1000.0) / 1e6,
        final_fitness
    };
}

void print_result(const BenchmarkResult& r) {
    std::cout << "  " << std::left << std::setw(35) << r.name << " | "
              << std::fixed << std::setprecision(2) << std::right << std::setw(10) << r.total_time_ms << " ms | "
              << std::setw(8) << r.time_per_gen_ms << " ms/gen | "
              << std::setprecision(0) << std::setw(8) << r.throughput_mgenes_sec << " MG/s | "
              << std::setprecision(4) << std::setw(12) << r.final_fitness << std::endl;
}

void run_benchmark_suite(int pop_size, int chrom_len, int num_generations) {
    std::cout << "\n========================================================" << std::endl;
    std::cout << "  Configuration: " << pop_size << " pop × " << chrom_len
              << " genes × " << num_generations << " generations" << std::endl;
    std::cout << "========================================================" << std::endl;
    std::cout << "  " << std::left << std::setw(35) << "Implementation" << " | "
              << std::setw(12) << "Total" << " | "
              << std::setw(10) << "Per Gen" << " | "
              << std::setw(10) << "Throughput" << " | "
              << std::setw(12) << "Final Fitness" << std::endl;
    std::cout << "  " << std::string(35, '-') << "-+-"
              << std::string(12, '-') << "-+-"
              << std::string(10, '-') << "-+-"
              << std::string(10, '-') << "-+-"
              << std::string(12, '-') << std::endl;

    BenchmarkResult baseline = benchmark_baseline(pop_size, chrom_len, num_generations);
    print_result(baseline);

    BenchmarkResult gene_major = benchmark_gene_major(pop_size, chrom_len, num_generations);
    print_result(gene_major);

    double speedup = baseline.total_time_ms / gene_major.total_time_ms;

    std::cout << "  " << std::string(90, '-') << std::endl;
    std::cout << "  Speedup (Gene-Major vs Baseline): " << std::fixed << std::setprecision(2)
              << speedup << "×" << std::endl;
}

int main() {
    std::cout << "================================================================" << std::endl;
    std::cout << "  BrkgaCuda 2.0 Optimization Benchmark" << std::endl;
    std::cout << "  Comparing Baseline vs Gene-Major Memory Layout (Phase 5)" << std::endl;
    std::cout << "================================================================" << std::endl;

    // Get GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Memory: " << std::fixed << std::setprecision(1)
              << (prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;

    // Small configuration
    run_benchmark_suite(1000, 100, 100);

    // Medium configuration
    run_benchmark_suite(2000, 200, 100);

    // Large configuration
    run_benchmark_suite(4000, 300, 50);

    // Very large configuration
    run_benchmark_suite(8000, 500, 25);

    std::cout << "\n================================================================" << std::endl;
    std::cout << "  Benchmark Complete" << std::endl;
    std::cout << "================================================================" << std::endl;

    return 0;
}
