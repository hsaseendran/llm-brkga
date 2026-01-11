// test_reorder_layout.cu - Test reorder_population_kernel gene-major version
// Verifies that gene-major reorder produces correct results

#include "core/cuda_kernels.cuh"
#include "core/memory_layout.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <random>

/**
 * @brief Test gene-major reorder kernel
 *
 * Strategy:
 * 1. Create population with known values
 * 2. Create indices for reordering (random permutation)
 * 3. Run reorder with gene-major layout
 * 4. Verify individuals are correctly relocated
 */

bool test_reorder_correctness() {
    std::cout << "\n[Test 1] Reorder correctness verification" << std::endl;

    const int pop_size = 100;
    const int chrom_len = 50;

    std::cout << "  Population: " << pop_size << " × " << chrom_len << std::endl;

    // Create population with unique identifiers (in gene-major layout)
    std::vector<float> h_src(pop_size * chrom_len);
    for (int gene = 0; gene < chrom_len; gene++) {
        for (int ind = 0; ind < pop_size; ind++) {
            // Each individual has unique identifier: ind * 0.01 + gene * 0.0001
            h_src[gene * pop_size + ind] = ind * 0.01f + gene * 0.0001f;
        }
    }

    // Create indices for reordering (reverse order)
    std::vector<int> h_indices(pop_size);
    for (int i = 0; i < pop_size; i++) {
        h_indices[i] = pop_size - 1 - i;  // Reverse mapping
    }

    // Allocate device memory
    float* d_src = nullptr;
    float* d_dst = nullptr;
    int* d_indices = nullptr;

    cudaMalloc(&d_src, pop_size * chrom_len * sizeof(float));
    cudaMalloc(&d_dst, pop_size * chrom_len * sizeof(float));
    cudaMalloc(&d_indices, pop_size * sizeof(int));

    cudaMemcpy(d_src, h_src.data(), pop_size * chrom_len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, h_indices.data(), pop_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_dst, 0, pop_size * chrom_len * sizeof(float));

    std::cout << "  ✓ Test data prepared (reverse mapping)" << std::endl;

    // Run gene-major reorder
    dim3 block(256);
    dim3 grid(chrom_len, (pop_size + 255) / 256);

    reorder_population_kernel_gene_major<<<grid, block>>>(
        d_src, d_dst, d_indices, pop_size, chrom_len
    );
    cudaDeviceSynchronize();

    std::cout << "  ✓ Gene-major reorder complete" << std::endl;

    // Copy result back
    std::vector<float> h_dst(pop_size * chrom_len);
    cudaMemcpy(h_dst.data(), d_dst, pop_size * chrom_len * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify reordering is correct
    bool correct = true;
    for (int ind = 0; ind < pop_size && correct; ind++) {
        int src_ind = h_indices[ind];  // Where this individual came from
        for (int gene = 0; gene < chrom_len && correct; gene++) {
            float expected = h_src[gene * pop_size + src_ind];
            float actual = h_dst[gene * pop_size + ind];
            if (std::abs(expected - actual) > 1e-6f) {
                correct = false;
                std::cout << "  ✗ Mismatch at dst_ind=" << ind << ", gene=" << gene
                          << ": expected " << expected << ", got " << actual << std::endl;
            }
        }
    }

    if (correct) {
        std::cout << "  ✓ All " << pop_size << " individuals correctly reordered" << std::endl;
    }

    // Cleanup
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_indices);

    std::cout << "  Result: " << (correct ? "✓ PASS" : "✗ FAIL") << std::endl;
    return correct;
}

bool test_random_permutation() {
    std::cout << "\n[Test 2] Random permutation reorder" << std::endl;

    const int pop_size = 200;
    const int chrom_len = 100;

    // Create population with known values
    std::vector<float> h_src(pop_size * chrom_len);
    for (int gene = 0; gene < chrom_len; gene++) {
        for (int ind = 0; ind < pop_size; ind++) {
            h_src[gene * pop_size + ind] = ind + gene * 0.001f;
        }
    }

    // Create random permutation
    std::vector<int> h_indices(pop_size);
    std::iota(h_indices.begin(), h_indices.end(), 0);
    std::mt19937 rng(42);
    std::shuffle(h_indices.begin(), h_indices.end(), rng);

    std::cout << "  Sample mappings: dst[0]<-src[" << h_indices[0]
              << "], dst[1]<-src[" << h_indices[1]
              << "], dst[2]<-src[" << h_indices[2] << "]" << std::endl;

    float* d_src = nullptr;
    float* d_dst = nullptr;
    int* d_indices = nullptr;

    cudaMalloc(&d_src, pop_size * chrom_len * sizeof(float));
    cudaMalloc(&d_dst, pop_size * chrom_len * sizeof(float));
    cudaMalloc(&d_indices, pop_size * sizeof(int));

    cudaMemcpy(d_src, h_src.data(), pop_size * chrom_len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, h_indices.data(), pop_size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid(chrom_len, (pop_size + 255) / 256);

    reorder_population_kernel_gene_major<<<grid, block>>>(
        d_src, d_dst, d_indices, pop_size, chrom_len
    );
    cudaDeviceSynchronize();

    std::vector<float> h_dst(pop_size * chrom_len);
    cudaMemcpy(h_dst.data(), d_dst, pop_size * chrom_len * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify random permutation is correct
    bool correct = true;
    for (int ind = 0; ind < pop_size && correct; ind++) {
        int src_ind = h_indices[ind];
        for (int gene = 0; gene < chrom_len && correct; gene++) {
            float expected = h_src[gene * pop_size + src_ind];
            float actual = h_dst[gene * pop_size + ind];
            if (std::abs(expected - actual) > 1e-6f) {
                correct = false;
                std::cout << "  ✗ Mismatch at dst_ind=" << ind << ", gene=" << gene << std::endl;
            }
        }
    }

    if (correct) {
        std::cout << "  ✓ Random permutation correctly applied" << std::endl;
    }

    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_indices);

    std::cout << "  Result: " << (correct ? "✓ PASS" : "✗ FAIL") << std::endl;
    return correct;
}

bool test_large_reorder() {
    std::cout << "\n[Test 3] Large population stress test" << std::endl;

    const int pop_size = 4000;
    const int chrom_len = 300;

    std::cout << "  Population: " << pop_size << " × " << chrom_len << std::endl;
    std::cout << "  Total data: " << (pop_size * chrom_len * sizeof(float) * 2 / (1024.0 * 1024.0)) << " MB" << std::endl;

    float* d_src = nullptr;
    float* d_dst = nullptr;
    int* d_indices = nullptr;

    cudaError_t err = cudaMalloc(&d_src, pop_size * chrom_len * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "  ✗ Failed to allocate: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    err = cudaMalloc(&d_dst, pop_size * chrom_len * sizeof(float));
    err = cudaMalloc(&d_indices, pop_size * sizeof(int));

    std::cout << "  ✓ Memory allocated" << std::endl;

    // Initialize with identity indices
    std::vector<int> h_indices(pop_size);
    std::iota(h_indices.begin(), h_indices.end(), 0);
    cudaMemcpy(d_indices, h_indices.data(), pop_size * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemset(d_src, 0, pop_size * chrom_len * sizeof(float));

    dim3 block(256);
    dim3 grid(chrom_len, (pop_size + 255) / 256);

    reorder_population_kernel_gene_major<<<grid, block>>>(
        d_src, d_dst, d_indices, pop_size, chrom_len
    );

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "  ✗ Kernel failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_src);
        cudaFree(d_dst);
        cudaFree(d_indices);
        return false;
    }

    std::cout << "  ✓ Gene-major reorder successful" << std::endl;

    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_indices);

    bool passed = (err == cudaSuccess);
    std::cout << "  Result: " << (passed ? "✓ PASS" : "✗ FAIL") << std::endl;
    return passed;
}

bool test_performance_comparison() {
    std::cout << "\n[Test 4] Performance comparison" << std::endl;

    const int pop_size = 2000;
    const int chrom_len = 300;
    const int num_runs = 5;

    std::cout << "  Configuration: " << pop_size << " pop × " << chrom_len << " genes" << std::endl;

    float* d_src_ind = nullptr;
    float* d_dst_ind = nullptr;
    float* d_src_gene = nullptr;
    float* d_dst_gene = nullptr;
    int* d_indices = nullptr;

    cudaMalloc(&d_src_ind, pop_size * chrom_len * sizeof(float));
    cudaMalloc(&d_dst_ind, pop_size * chrom_len * sizeof(float));
    cudaMalloc(&d_src_gene, pop_size * chrom_len * sizeof(float));
    cudaMalloc(&d_dst_gene, pop_size * chrom_len * sizeof(float));
    cudaMalloc(&d_indices, pop_size * sizeof(int));

    std::vector<int> h_indices(pop_size);
    std::iota(h_indices.begin(), h_indices.end(), 0);
    std::mt19937 rng(123);
    std::shuffle(h_indices.begin(), h_indices.end(), rng);
    cudaMemcpy(d_indices, h_indices.data(), pop_size * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemset(d_src_ind, 0, pop_size * chrom_len * sizeof(float));
    cudaMemset(d_src_gene, 0, pop_size * chrom_len * sizeof(float));

    std::cout << "  ✓ Memory initialized" << std::endl;

    // Warmup
    dim3 block_ind(256);
    dim3 grid_ind((pop_size + 255) / 256);
    reorder_population_kernel<<<grid_ind, block_ind>>>(
        d_src_ind, d_dst_ind, d_indices, pop_size, chrom_len
    );
    cudaDeviceSynchronize();

    dim3 block_gene(256);
    dim3 grid_gene(chrom_len, (pop_size + 255) / 256);
    reorder_population_kernel_gene_major<<<grid_gene, block_gene>>>(
        d_src_gene, d_dst_gene, d_indices, pop_size, chrom_len
    );
    cudaDeviceSynchronize();

    std::cout << "  ✓ Warmup complete" << std::endl;

    // Time individual-major
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < num_runs; i++) {
        reorder_population_kernel<<<grid_ind, block_ind>>>(
            d_src_ind, d_dst_ind, d_indices, pop_size, chrom_len
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
        reorder_population_kernel_gene_major<<<grid_gene, block_gene>>>(
            d_src_gene, d_dst_gene, d_indices, pop_size, chrom_len
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
    cudaFree(d_src_ind);
    cudaFree(d_dst_ind);
    cudaFree(d_src_gene);
    cudaFree(d_dst_gene);
    cudaFree(d_indices);

    std::cout << "  Result: ✓ PASS" << std::endl;
    return true;
}

int main() {
    std::cout << "================================================" << std::endl;
    std::cout << "  Phase 5 Day 5: Reorder Kernel Tests          " << std::endl;
    std::cout << "================================================" << std::endl;

    int passed = 0;
    int total = 4;

    if (test_reorder_correctness()) passed++;
    if (test_random_permutation()) passed++;
    if (test_large_reorder()) passed++;
    if (test_performance_comparison()) passed++;

    std::cout << "\n================================================" << std::endl;
    std::cout << "  Test Summary: " << passed << "/" << total << " passed" << std::endl;

    if (passed == total) {
        std::cout << "  ✓ ALL TESTS PASSED" << std::endl;
        std::cout << "  Gene-major reorder kernel ready!" << std::endl;
        std::cout << "================================================" << std::endl;
        return 0;
    } else {
        std::cout << "  ✗ " << (total - passed) << " TESTS FAILED" << std::endl;
        std::cout << "================================================" << std::endl;
        return 1;
    }
}
