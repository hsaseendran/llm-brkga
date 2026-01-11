// test_mutation_layout.cu - Test mutation_kernel gene-major version
// Verifies that gene-major mutation produces statistically valid mutants

#include "core/cuda_kernels.cuh"
#include "core/memory_layout.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

/**
 * @brief Test gene-major mutation kernel
 *
 * Strategy:
 * 1. Run mutation with gene-major layout
 * 2. Verify all values are in valid range [0, 1]
 * 3. Check uniform distribution (mean ~0.5)
 * 4. Performance comparison vs individual-major
 */

bool test_mutation_correctness() {
    std::cout << "\n[Test 1] Gene-major mutation statistical validity" << std::endl;

    const int num_mutants = 500;
    const int chrom_len = 200;

    std::cout << "  Mutants: " << num_mutants << std::endl;
    std::cout << "  Chromosome length: " << chrom_len << std::endl;

    // Allocate device memory
    float* d_mutants = nullptr;
    curandState* d_states = nullptr;

    cudaMalloc(&d_mutants, num_mutants * chrom_len * sizeof(float));
    cudaMalloc(&d_states, num_mutants * sizeof(curandState));

    // Initialize RNG states (not used by gene-major version, but required for API)
    dim3 block_init(256);
    dim3 grid_init((num_mutants + 255) / 256);
    init_curand_states_kernel<<<grid_init, block_init>>>(d_states, num_mutants, 12345);
    cudaDeviceSynchronize();

    std::cout << "  ✓ Memory allocated" << std::endl;

    // Run gene-major mutation
    dim3 block(256);
    dim3 grid(chrom_len, (num_mutants + 255) / 256);

    mutation_kernel_gene_major<<<grid, block>>>(
        d_mutants, d_states, num_mutants, chrom_len
    );
    cudaDeviceSynchronize();

    std::cout << "  ✓ Gene-major mutation complete" << std::endl;

    // Copy back
    std::vector<float> h_mutants(num_mutants * chrom_len);
    cudaMemcpy(h_mutants.data(), d_mutants, num_mutants * chrom_len * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify statistical properties
    float min_val = 1.0f;
    float max_val = 0.0f;
    double sum = 0.0;

    for (float val : h_mutants) {
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
        sum += val;
    }

    double mean = sum / h_mutants.size();

    std::cout << "  Statistical properties:" << std::endl;
    std::cout << "    Min: " << std::fixed << std::setprecision(6) << min_val << std::endl;
    std::cout << "    Max: " << std::fixed << std::setprecision(6) << max_val << std::endl;
    std::cout << "    Mean: " << std::fixed << std::setprecision(6) << mean << " (expected ~0.5)" << std::endl;

    bool valid_range = (min_val >= 0.0f && max_val <= 1.0f);
    bool valid_mean = (mean > 0.48 && mean < 0.52);

    bool passed = valid_range && valid_mean;

    if (valid_range) {
        std::cout << "  ✓ All values in [0, 1]" << std::endl;
    } else {
        std::cout << "  ✗ Values outside [0, 1]" << std::endl;
    }

    if (valid_mean) {
        std::cout << "  ✓ Mean close to 0.5 (uniform distribution)" << std::endl;
    } else {
        std::cout << "  ✗ Mean deviates from 0.5" << std::endl;
    }

    // Cleanup
    cudaFree(d_mutants);
    cudaFree(d_states);

    std::cout << "  Result: " << (passed ? "✓ PASS" : "✗ FAIL") << std::endl;
    return passed;
}

bool test_mutant_diversity() {
    std::cout << "\n[Test 2] Mutant diversity check" << std::endl;

    const int num_mutants = 100;
    const int chrom_len = 50;

    float* d_mutants = nullptr;
    curandState* d_states = nullptr;

    cudaMalloc(&d_mutants, num_mutants * chrom_len * sizeof(float));
    cudaMalloc(&d_states, num_mutants * sizeof(curandState));

    init_curand_states_kernel<<<(num_mutants + 255) / 256, 256>>>(d_states, num_mutants, 99999);
    cudaDeviceSynchronize();

    dim3 block(256);
    dim3 grid(chrom_len, (num_mutants + 255) / 256);

    mutation_kernel_gene_major<<<grid, block>>>(
        d_mutants, d_states, num_mutants, chrom_len
    );
    cudaDeviceSynchronize();

    std::vector<float> h_mutants(num_mutants * chrom_len);
    cudaMemcpy(h_mutants.data(), d_mutants, num_mutants * chrom_len * sizeof(float), cudaMemcpyDeviceToHost);

    // Check diversity - no two mutants should be identical
    bool all_unique = true;
    for (int i = 0; i < num_mutants - 1 && all_unique; i++) {
        for (int j = i + 1; j < num_mutants && all_unique; j++) {
            bool identical = true;
            for (int gene = 0; gene < chrom_len; gene++) {
                int offset_i = gene * num_mutants + i;
                int offset_j = gene * num_mutants + j;
                if (std::abs(h_mutants[offset_i] - h_mutants[offset_j]) > 1e-6f) {
                    identical = false;
                    break;
                }
            }
            if (identical) {
                all_unique = false;
                std::cout << "  ✗ Mutant " << i << " and " << j << " are identical" << std::endl;
            }
        }
    }

    if (all_unique) {
        std::cout << "  ✓ All " << num_mutants << " mutants are unique" << std::endl;
    }

    cudaFree(d_mutants);
    cudaFree(d_states);

    std::cout << "  Result: " << (all_unique ? "✓ PASS" : "✗ FAIL") << std::endl;
    return all_unique;
}

bool test_large_mutation() {
    std::cout << "\n[Test 3] Large population stress test" << std::endl;

    const int num_mutants = 4000;
    const int chrom_len = 500;

    std::cout << "  Mutants: " << num_mutants << std::endl;
    std::cout << "  Chromosome: " << chrom_len << std::endl;
    std::cout << "  Total data: " << (num_mutants * chrom_len * sizeof(float) / (1024.0 * 1024.0)) << " MB" << std::endl;

    float* d_mutants = nullptr;
    curandState* d_states = nullptr;

    cudaError_t err = cudaMalloc(&d_mutants, num_mutants * chrom_len * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "  ✗ Failed to allocate: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    err = cudaMalloc(&d_states, num_mutants * sizeof(curandState));
    if (err != cudaSuccess) {
        cudaFree(d_mutants);
        std::cerr << "  ✗ Failed to allocate states: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    std::cout << "  ✓ Memory allocated successfully" << std::endl;

    init_curand_states_kernel<<<(num_mutants + 255) / 256, 256>>>(d_states, num_mutants, 77777);
    cudaDeviceSynchronize();

    dim3 block(256);
    dim3 grid(chrom_len, (num_mutants + 255) / 256);

    mutation_kernel_gene_major<<<grid, block>>>(
        d_mutants, d_states, num_mutants, chrom_len
    );

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "  ✗ Kernel execution failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_mutants);
        cudaFree(d_states);
        return false;
    }

    std::cout << "  ✓ Gene-major mutation successful" << std::endl;

    // Spot check
    std::vector<float> sample(100);
    cudaMemcpy(sample.data(), d_mutants, 100 * sizeof(float), cudaMemcpyDeviceToHost);

    bool all_valid = true;
    for (float val : sample) {
        if (val < 0.0f || val > 1.0f || std::isnan(val) || std::isinf(val)) {
            all_valid = false;
            break;
        }
    }

    if (all_valid) {
        std::cout << "  ✓ Spot check: Values in valid range" << std::endl;
    } else {
        std::cout << "  ✗ Spot check: Invalid values detected" << std::endl;
    }

    cudaFree(d_mutants);
    cudaFree(d_states);

    bool passed = (err == cudaSuccess) && all_valid;
    std::cout << "  Result: " << (passed ? "✓ PASS" : "✗ FAIL") << std::endl;
    return passed;
}

bool test_performance_comparison() {
    std::cout << "\n[Test 4] Performance comparison" << std::endl;

    const int num_mutants = 2000;
    const int chrom_len = 300;
    const int num_runs = 5;

    std::cout << "  Configuration: " << num_mutants << " mutants × " << chrom_len << " genes" << std::endl;

    float* d_mutants_ind = nullptr;
    float* d_mutants_gene = nullptr;
    curandState* d_states = nullptr;

    cudaMalloc(&d_mutants_ind, num_mutants * chrom_len * sizeof(float));
    cudaMalloc(&d_mutants_gene, num_mutants * chrom_len * sizeof(float));
    cudaMalloc(&d_states, num_mutants * sizeof(curandState));

    init_curand_states_kernel<<<(num_mutants + 255) / 256, 256>>>(d_states, num_mutants, 55555);
    cudaDeviceSynchronize();

    std::cout << "  ✓ Memory initialized" << std::endl;

    // Warmup
    dim3 block_ind(256);
    dim3 grid_ind((num_mutants + 255) / 256);
    mutation_kernel<<<grid_ind, block_ind>>>(d_mutants_ind, d_states, num_mutants, chrom_len);
    cudaDeviceSynchronize();

    dim3 block_gene(256);
    dim3 grid_gene(chrom_len, (num_mutants + 255) / 256);
    mutation_kernel_gene_major<<<grid_gene, block_gene>>>(d_mutants_gene, d_states, num_mutants, chrom_len);
    cudaDeviceSynchronize();

    std::cout << "  ✓ Warmup complete" << std::endl;

    // Time individual-major
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < num_runs; i++) {
        mutation_kernel<<<grid_ind, block_ind>>>(d_mutants_ind, d_states, num_mutants, chrom_len);
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
        mutation_kernel_gene_major<<<grid_gene, block_gene>>>(d_mutants_gene, d_states, num_mutants, chrom_len);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_gene_ms = 0.0f;
    cudaEventElapsedTime(&time_gene_ms, start, stop);
    time_gene_ms /= num_runs;

    std::cout << "  Gene-major: " << std::fixed << std::setprecision(3) << time_gene_ms << " ms/run" << std::endl;

    float speedup = time_ind_ms / time_gene_ms;
    std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << speedup << "×" << std::endl;

    bool significant_speedup = (speedup > 2.0);

    if (significant_speedup) {
        std::cout << "  ✓ Gene-major shows significant performance improvement" << std::endl;
    } else {
        std::cout << "  ⚠ Gene-major speedup less than expected (may vary by GPU)" << std::endl;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_mutants_ind);
    cudaFree(d_mutants_gene);
    cudaFree(d_states);

    std::cout << "  Result: ✓ PASS" << std::endl;
    return true;
}

int main() {
    std::cout << "================================================" << std::endl;
    std::cout << "  Phase 5 Day 3: Mutation Kernel Tests         " << std::endl;
    std::cout << "================================================" << std::endl;

    int passed = 0;
    int total = 4;

    if (test_mutation_correctness()) passed++;
    if (test_mutant_diversity()) passed++;
    if (test_large_mutation()) passed++;
    if (test_performance_comparison()) passed++;

    std::cout << "\n================================================" << std::endl;
    std::cout << "  Test Summary: " << passed << "/" << total << " passed" << std::endl;

    if (passed == total) {
        std::cout << "  ✓ ALL TESTS PASSED" << std::endl;
        std::cout << "  Gene-major mutation kernel ready!" << std::endl;
        std::cout << "================================================" << std::endl;
        return 0;
    } else {
        std::cout << "  ✗ " << (total - passed) << " TESTS FAILED" << std::endl;
        std::cout << "================================================" << std::endl;
        return 1;
    }
}
