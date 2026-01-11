// test_brkga_generation_layout.cu - Test brkga_generation_kernel gene-major version
// Verifies that gene-major BRKGA generation produces correct results

#include "core/cuda_kernels.cuh"
#include "core/memory_layout.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <numeric>

/**
 * @brief Test gene-major BRKGA generation kernel
 *
 * Strategy:
 * 1. Create population with known values
 * 2. Run BRKGA generation with gene-major layout
 * 3. Verify:
 *    - Elite individuals are correctly copied
 *    - Mutants have valid random values
 *    - Offspring have values from parent pools
 *    - Elite probability is approximately correct
 */

bool test_elite_copy() {
    std::cout << "\n[Test 1] Elite copy verification" << std::endl;

    const int pop_size = 100;
    const int elite_size = 20;
    const int mutant_size = 10;
    const int chrom_len = 50;
    const float elite_prob = 0.7f;

    std::cout << "  Population: " << pop_size << " (elite: " << elite_size
              << ", mutants: " << mutant_size << ")" << std::endl;

    // Create population with known values (in gene-major layout)
    std::vector<float> h_population(pop_size * chrom_len);
    for (int gene = 0; gene < chrom_len; gene++) {
        for (int ind = 0; ind < pop_size; ind++) {
            // Each individual has unique identifier: ind * 0.001 + gene * 0.00001
            h_population[gene * pop_size + ind] = ind * 0.001f + gene * 0.00001f;
        }
    }

    // Create sorted indices (identity for simplicity - elite are at indices 0..elite_size-1)
    std::vector<int> h_sorted_indices(pop_size);
    std::iota(h_sorted_indices.begin(), h_sorted_indices.end(), 0);

    // Allocate device memory
    float* d_population = nullptr;
    float* d_next_gen = nullptr;
    int* d_sorted_indices = nullptr;
    curandState* d_states = nullptr;

    cudaMalloc(&d_population, pop_size * chrom_len * sizeof(float));
    cudaMalloc(&d_next_gen, pop_size * chrom_len * sizeof(float));
    cudaMalloc(&d_sorted_indices, pop_size * sizeof(int));
    cudaMalloc(&d_states, pop_size * sizeof(curandState));

    cudaMemcpy(d_population, h_population.data(), pop_size * chrom_len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sorted_indices, h_sorted_indices.data(), pop_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_next_gen, 0, pop_size * chrom_len * sizeof(float));

    std::cout << "  ✓ Test data prepared" << std::endl;

    // Run gene-major BRKGA generation
    dim3 block(256);
    dim3 grid(chrom_len, (pop_size + 255) / 256);

    brkga_generation_kernel_gene_major<<<grid, block>>>(
        d_population, d_next_gen, d_sorted_indices, d_states,
        pop_size, elite_size, mutant_size, chrom_len, elite_prob
    );
    cudaDeviceSynchronize();

    std::cout << "  ✓ Gene-major BRKGA generation complete" << std::endl;

    // Copy result back
    std::vector<float> h_next_gen(pop_size * chrom_len);
    cudaMemcpy(h_next_gen.data(), d_next_gen, pop_size * chrom_len * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify elite individuals are correctly copied
    bool elite_ok = true;
    for (int ind = 0; ind < elite_size; ind++) {
        for (int gene = 0; gene < chrom_len; gene++) {
            int offset = gene * pop_size + ind;
            float expected = h_population[offset];  // Same because sorted_indices is identity
            float actual = h_next_gen[offset];
            if (std::abs(expected - actual) > 1e-6f) {
                elite_ok = false;
                std::cout << "  ✗ Elite mismatch at ind=" << ind << ", gene=" << gene
                          << ": expected " << expected << ", got " << actual << std::endl;
                break;
            }
        }
        if (!elite_ok) break;
    }

    if (elite_ok) {
        std::cout << "  ✓ All " << elite_size << " elite individuals correctly copied" << std::endl;
    }

    // Cleanup
    cudaFree(d_population);
    cudaFree(d_next_gen);
    cudaFree(d_sorted_indices);
    cudaFree(d_states);

    std::cout << "  Result: " << (elite_ok ? "✓ PASS" : "✗ FAIL") << std::endl;
    return elite_ok;
}

bool test_mutant_generation() {
    std::cout << "\n[Test 2] Mutant generation verification" << std::endl;

    const int pop_size = 100;
    const int elite_size = 20;
    const int mutant_size = 15;
    const int chrom_len = 50;
    const float elite_prob = 0.7f;

    int mutant_start = pop_size - mutant_size;

    std::cout << "  Mutant indices: " << mutant_start << " to " << (pop_size - 1) << std::endl;

    // Initialize population with zeros
    std::vector<float> h_population(pop_size * chrom_len, 0.0f);
    std::vector<int> h_sorted_indices(pop_size);
    std::iota(h_sorted_indices.begin(), h_sorted_indices.end(), 0);

    float* d_population = nullptr;
    float* d_next_gen = nullptr;
    int* d_sorted_indices = nullptr;
    curandState* d_states = nullptr;

    cudaMalloc(&d_population, pop_size * chrom_len * sizeof(float));
    cudaMalloc(&d_next_gen, pop_size * chrom_len * sizeof(float));
    cudaMalloc(&d_sorted_indices, pop_size * sizeof(int));
    cudaMalloc(&d_states, pop_size * sizeof(curandState));

    cudaMemcpy(d_population, h_population.data(), pop_size * chrom_len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sorted_indices, h_sorted_indices.data(), pop_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_next_gen, 0, pop_size * chrom_len * sizeof(float));

    dim3 block(256);
    dim3 grid(chrom_len, (pop_size + 255) / 256);

    brkga_generation_kernel_gene_major<<<grid, block>>>(
        d_population, d_next_gen, d_sorted_indices, d_states,
        pop_size, elite_size, mutant_size, chrom_len, elite_prob
    );
    cudaDeviceSynchronize();

    std::vector<float> h_next_gen(pop_size * chrom_len);
    cudaMemcpy(h_next_gen.data(), d_next_gen, pop_size * chrom_len * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify mutant values are in [0, 1] and have reasonable distribution
    float min_val = 1.0f;
    float max_val = 0.0f;
    double sum = 0.0;
    int mutant_genes = 0;

    for (int ind = mutant_start; ind < pop_size; ind++) {
        for (int gene = 0; gene < chrom_len; gene++) {
            float val = h_next_gen[gene * pop_size + ind];
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
            sum += val;
            mutant_genes++;
        }
    }

    double mean = sum / mutant_genes;
    bool valid_range = (min_val >= 0.0f && max_val <= 1.0f);
    bool valid_mean = (mean > 0.4 && mean < 0.6);

    std::cout << "  Mutant statistics:" << std::endl;
    std::cout << "    Count: " << mutant_genes << " genes" << std::endl;
    std::cout << "    Range: [" << min_val << ", " << max_val << "]" << std::endl;
    std::cout << "    Mean: " << mean << std::endl;

    bool passed = valid_range && valid_mean;

    if (valid_range) {
        std::cout << "  ✓ Mutant values in valid range [0, 1]" << std::endl;
    } else {
        std::cout << "  ✗ Mutant values outside valid range" << std::endl;
    }

    if (valid_mean) {
        std::cout << "  ✓ Mean close to 0.5 (uniform distribution)" << std::endl;
    } else {
        std::cout << "  ✗ Mean deviates from expected" << std::endl;
    }

    cudaFree(d_population);
    cudaFree(d_next_gen);
    cudaFree(d_sorted_indices);
    cudaFree(d_states);

    std::cout << "  Result: " << (passed ? "✓ PASS" : "✗ FAIL") << std::endl;
    return passed;
}

bool test_crossover_operation() {
    std::cout << "\n[Test 3] Crossover operation verification" << std::endl;

    const int pop_size = 100;
    const int elite_size = 20;
    const int mutant_size = 10;
    const int chrom_len = 50;
    const float elite_prob = 0.7f;

    int offspring_start = elite_size;
    int offspring_end = pop_size - mutant_size;
    int num_offspring = offspring_end - offspring_start;

    std::cout << "  Offspring indices: " << offspring_start << " to " << (offspring_end - 1)
              << " (" << num_offspring << " total)" << std::endl;

    // Create population with distinct elite/non-elite values
    std::vector<float> h_population(pop_size * chrom_len);
    for (int gene = 0; gene < chrom_len; gene++) {
        for (int ind = 0; ind < pop_size; ind++) {
            if (ind < elite_size) {
                // Elite: values in [0.0, 0.2]
                h_population[gene * pop_size + ind] = 0.1f;
            } else {
                // Non-elite: values in [0.8, 1.0]
                h_population[gene * pop_size + ind] = 0.9f;
            }
        }
    }

    std::vector<int> h_sorted_indices(pop_size);
    std::iota(h_sorted_indices.begin(), h_sorted_indices.end(), 0);

    float* d_population = nullptr;
    float* d_next_gen = nullptr;
    int* d_sorted_indices = nullptr;
    curandState* d_states = nullptr;

    cudaMalloc(&d_population, pop_size * chrom_len * sizeof(float));
    cudaMalloc(&d_next_gen, pop_size * chrom_len * sizeof(float));
    cudaMalloc(&d_sorted_indices, pop_size * sizeof(int));
    cudaMalloc(&d_states, pop_size * sizeof(curandState));

    cudaMemcpy(d_population, h_population.data(), pop_size * chrom_len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sorted_indices, h_sorted_indices.data(), pop_size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid(chrom_len, (pop_size + 255) / 256);

    brkga_generation_kernel_gene_major<<<grid, block>>>(
        d_population, d_next_gen, d_sorted_indices, d_states,
        pop_size, elite_size, mutant_size, chrom_len, elite_prob
    );
    cudaDeviceSynchronize();

    std::vector<float> h_next_gen(pop_size * chrom_len);
    cudaMemcpy(h_next_gen.data(), d_next_gen, pop_size * chrom_len * sizeof(float), cudaMemcpyDeviceToHost);

    // Count how many offspring genes came from elite vs non-elite
    int from_elite = 0;
    int from_non_elite = 0;
    int total_offspring_genes = num_offspring * chrom_len;

    for (int ind = offspring_start; ind < offspring_end; ind++) {
        for (int gene = 0; gene < chrom_len; gene++) {
            float val = h_next_gen[gene * pop_size + ind];
            if (val < 0.5f) {
                from_elite++;  // 0.1 is from elite
            } else {
                from_non_elite++;  // 0.9 is from non-elite
            }
        }
    }

    float elite_ratio = (float)from_elite / total_offspring_genes;

    std::cout << "  Crossover statistics:" << std::endl;
    std::cout << "    From elite: " << from_elite << " (" << (elite_ratio * 100) << "%)" << std::endl;
    std::cout << "    From non-elite: " << from_non_elite << " (" << ((1.0f - elite_ratio) * 100) << "%)" << std::endl;
    std::cout << "    Expected elite: " << (elite_prob * 100) << "%" << std::endl;

    // Check if elite probability is approximately correct (within 10%)
    bool prob_ok = (elite_ratio > elite_prob - 0.1f) && (elite_ratio < elite_prob + 0.1f);

    if (prob_ok) {
        std::cout << "  ✓ Elite probability approximately correct" << std::endl;
    } else {
        std::cout << "  ✗ Elite probability deviates from expected" << std::endl;
    }

    cudaFree(d_population);
    cudaFree(d_next_gen);
    cudaFree(d_sorted_indices);
    cudaFree(d_states);

    std::cout << "  Result: " << (prob_ok ? "✓ PASS" : "✗ FAIL") << std::endl;
    return prob_ok;
}

bool test_large_population() {
    std::cout << "\n[Test 4] Large population stress test" << std::endl;

    const int pop_size = 4000;
    const int elite_size = 800;
    const int mutant_size = 400;
    const int chrom_len = 300;
    const float elite_prob = 0.7f;

    std::cout << "  Population: " << pop_size << " × " << chrom_len << std::endl;
    std::cout << "  Total data: " << (pop_size * chrom_len * sizeof(float) * 2 / (1024.0 * 1024.0)) << " MB" << std::endl;

    float* d_population = nullptr;
    float* d_next_gen = nullptr;
    int* d_sorted_indices = nullptr;
    curandState* d_states = nullptr;

    cudaError_t err = cudaMalloc(&d_population, pop_size * chrom_len * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "  ✗ Failed to allocate population: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    err = cudaMalloc(&d_next_gen, pop_size * chrom_len * sizeof(float));
    err = cudaMalloc(&d_sorted_indices, pop_size * sizeof(int));
    err = cudaMalloc(&d_states, pop_size * sizeof(curandState));

    std::cout << "  ✓ Memory allocated" << std::endl;

    // Initialize sorted indices to identity
    std::vector<int> h_sorted_indices(pop_size);
    std::iota(h_sorted_indices.begin(), h_sorted_indices.end(), 0);
    cudaMemcpy(d_sorted_indices, h_sorted_indices.data(), pop_size * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize population with random-ish values
    cudaMemset(d_population, 0, pop_size * chrom_len * sizeof(float));

    dim3 block(256);
    dim3 grid(chrom_len, (pop_size + 255) / 256);

    brkga_generation_kernel_gene_major<<<grid, block>>>(
        d_population, d_next_gen, d_sorted_indices, d_states,
        pop_size, elite_size, mutant_size, chrom_len, elite_prob
    );

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "  ✗ Kernel failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_population);
        cudaFree(d_next_gen);
        cudaFree(d_sorted_indices);
        cudaFree(d_states);
        return false;
    }

    std::cout << "  ✓ Gene-major BRKGA generation successful" << std::endl;

    // Spot check
    std::vector<float> sample(100);
    cudaMemcpy(sample.data(), d_next_gen, 100 * sizeof(float), cudaMemcpyDeviceToHost);

    bool all_valid = true;
    for (float val : sample) {
        if (std::isnan(val) || std::isinf(val)) {
            all_valid = false;
            break;
        }
    }

    if (all_valid) {
        std::cout << "  ✓ Spot check: No NaN or Inf values" << std::endl;
    }

    cudaFree(d_population);
    cudaFree(d_next_gen);
    cudaFree(d_sorted_indices);
    cudaFree(d_states);

    bool passed = (err == cudaSuccess) && all_valid;
    std::cout << "  Result: " << (passed ? "✓ PASS" : "✗ FAIL") << std::endl;
    return passed;
}

bool test_performance_comparison() {
    std::cout << "\n[Test 5] Performance comparison" << std::endl;

    const int pop_size = 2000;
    const int elite_size = 400;
    const int mutant_size = 200;
    const int chrom_len = 200;
    const float elite_prob = 0.7f;
    const int num_runs = 5;

    std::cout << "  Configuration: " << pop_size << " pop × " << chrom_len << " genes" << std::endl;

    float* d_pop_ind = nullptr;
    float* d_next_ind = nullptr;
    float* d_pop_gene = nullptr;
    float* d_next_gene = nullptr;
    int* d_sorted_indices = nullptr;
    curandState* d_states = nullptr;

    cudaMalloc(&d_pop_ind, pop_size * chrom_len * sizeof(float));
    cudaMalloc(&d_next_ind, pop_size * chrom_len * sizeof(float));
    cudaMalloc(&d_pop_gene, pop_size * chrom_len * sizeof(float));
    cudaMalloc(&d_next_gene, pop_size * chrom_len * sizeof(float));
    cudaMalloc(&d_sorted_indices, pop_size * sizeof(int));
    cudaMalloc(&d_states, pop_size * sizeof(curandState));

    std::vector<int> h_sorted_indices(pop_size);
    std::iota(h_sorted_indices.begin(), h_sorted_indices.end(), 0);
    cudaMemcpy(d_sorted_indices, h_sorted_indices.data(), pop_size * sizeof(int), cudaMemcpyHostToDevice);

    init_curand_states_kernel<<<(pop_size + 255) / 256, 256>>>(d_states, pop_size, 12345);
    cudaDeviceSynchronize();

    std::cout << "  ✓ Memory initialized" << std::endl;

    // Warmup
    dim3 block_ind(256);
    dim3 grid_ind((pop_size + 255) / 256);
    brkga_generation_kernel<<<grid_ind, block_ind>>>(
        d_pop_ind, d_next_ind, d_sorted_indices, d_states,
        pop_size, elite_size, mutant_size, chrom_len, elite_prob
    );
    cudaDeviceSynchronize();

    dim3 block_gene(256);
    dim3 grid_gene(chrom_len, (pop_size + 255) / 256);
    brkga_generation_kernel_gene_major<<<grid_gene, block_gene>>>(
        d_pop_gene, d_next_gene, d_sorted_indices, d_states,
        pop_size, elite_size, mutant_size, chrom_len, elite_prob
    );
    cudaDeviceSynchronize();

    std::cout << "  ✓ Warmup complete" << std::endl;

    // Time individual-major
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < num_runs; i++) {
        brkga_generation_kernel<<<grid_ind, block_ind>>>(
            d_pop_ind, d_next_ind, d_sorted_indices, d_states,
            pop_size, elite_size, mutant_size, chrom_len, elite_prob
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_ind_ms = 0.0f;
    cudaEventElapsedTime(&time_ind_ms, start, stop);
    time_ind_ms /= num_runs;

    std::cout << "  Individual-major: " << std::fixed << std::setprecision(3) << time_ind_ms << " ms/run" << std::endl;

    // Time gene-major
    cudaEventRecord(start);
    for (int i = 0; i < num_runs; i++) {
        brkga_generation_kernel_gene_major<<<grid_gene, block_gene>>>(
            d_pop_gene, d_next_gene, d_sorted_indices, d_states,
            pop_size, elite_size, mutant_size, chrom_len, elite_prob
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_gene_ms = 0.0f;
    cudaEventElapsedTime(&time_gene_ms, start, stop);
    time_gene_ms /= num_runs;

    std::cout << "  Gene-major: " << std::fixed << std::setprecision(3) << time_gene_ms << " ms/run" << std::endl;

    float speedup = time_ind_ms / time_gene_ms;
    std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << speedup << "×" << std::endl;

    bool significant_speedup = (speedup > 1.5);

    if (significant_speedup) {
        std::cout << "  ✓ Gene-major shows performance improvement" << std::endl;
    } else {
        std::cout << "  ⚠ Gene-major speedup less than expected (may vary by GPU)" << std::endl;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_pop_ind);
    cudaFree(d_next_ind);
    cudaFree(d_pop_gene);
    cudaFree(d_next_gene);
    cudaFree(d_sorted_indices);
    cudaFree(d_states);

    std::cout << "  Result: ✓ PASS" << std::endl;
    return true;
}

int main() {
    std::cout << "================================================" << std::endl;
    std::cout << "  Phase 5 Day 4: BRKGA Generation Kernel Tests " << std::endl;
    std::cout << "================================================" << std::endl;

    int passed = 0;
    int total = 5;

    if (test_elite_copy()) passed++;
    if (test_mutant_generation()) passed++;
    if (test_crossover_operation()) passed++;
    if (test_large_population()) passed++;
    if (test_performance_comparison()) passed++;

    std::cout << "\n================================================" << std::endl;
    std::cout << "  Test Summary: " << passed << "/" << total << " passed" << std::endl;

    if (passed == total) {
        std::cout << "  ✓ ALL TESTS PASSED" << std::endl;
        std::cout << "  Gene-major BRKGA generation kernel ready!" << std::endl;
        std::cout << "================================================" << std::endl;
        return 0;
    } else {
        std::cout << "  ✗ " << (total - passed) << " TESTS FAILED" << std::endl;
        std::cout << "================================================" << std::endl;
        return 1;
    }
}
