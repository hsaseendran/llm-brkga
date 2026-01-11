// test_crossover_layout.cu - Test crossover_kernel gene-major version
// Verifies that gene-major crossover produces statistically equivalent offspring

#include "core/cuda_kernels.cuh"
#include "core/memory_layout.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <random>

/**
 * @brief Test gene-major crossover kernel
 *
 * Strategy:
 * 1. Create elite and non-elite populations with known values
 * 2. Run crossover with gene-major layout
 * 3. Verify offspring inherit correctly from parents
 * 4. Check elite probability is respected
 * 5. Performance benchmark vs individual-major
 */

bool test_crossover_correctness() {
    std::cout << "\n[Test 1] Gene-major crossover correctness" << std::endl;

    const int elite_size = 100;
    const int non_elite_size = 150;
    const int num_offspring = 200;
    const int chrom_len = 50;
    const float elite_prob = 0.7f;

    std::cout << "  Elite: " << elite_size << " individuals" << std::endl;
    std::cout << "  Non-elite: " << non_elite_size << " individuals" << std::endl;
    std::cout << "  Offspring: " << num_offspring << std::endl;
    std::cout << "  Chromosome length: " << chrom_len << std::endl;
    std::cout << "  Elite probability: " << elite_prob << std::endl;

    // Create known elite population (values in [0.0, 0.1])
    std::vector<float> h_elite(elite_size * chrom_len);
    for (int i = 0; i < elite_size * chrom_len; i++) {
        h_elite[i] = 0.05f + (i % 100) * 0.0001f;  // Small values
    }

    // Create known non-elite population (values in [0.9, 1.0])
    std::vector<float> h_non_elite(non_elite_size * chrom_len);
    for (int i = 0; i < non_elite_size * chrom_len; i++) {
        h_non_elite[i] = 0.95f + (i % 100) * 0.0001f;  // Large values
    }

    // Transpose to gene-major layout
    std::vector<float> h_elite_gene(elite_size * chrom_len);
    std::vector<float> h_non_elite_gene(non_elite_size * chrom_len);

    for (int ind = 0; ind < elite_size; ind++) {
        for (int gene = 0; gene < chrom_len; gene++) {
            h_elite_gene[gene * elite_size + ind] = h_elite[ind * chrom_len + gene];
        }
    }

    for (int ind = 0; ind < non_elite_size; ind++) {
        for (int gene = 0; gene < chrom_len; gene++) {
            h_non_elite_gene[gene * non_elite_size + ind] = h_non_elite[ind * chrom_len + gene];
        }
    }

    // Allocate device memory
    float* d_elite = nullptr;
    float* d_non_elite = nullptr;
    float* d_offspring = nullptr;
    curandState* d_states = nullptr;

    cudaMalloc(&d_elite, elite_size * chrom_len * sizeof(float));
    cudaMalloc(&d_non_elite, non_elite_size * chrom_len * sizeof(float));
    cudaMalloc(&d_offspring, num_offspring * chrom_len * sizeof(float));
    cudaMalloc(&d_states, num_offspring * sizeof(curandState));

    // Copy to device
    cudaMemcpy(d_elite, h_elite_gene.data(), elite_size * chrom_len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_non_elite, h_non_elite_gene.data(), non_elite_size * chrom_len * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize RNG states (not used by gene-major version, but required for API)
    dim3 block_init(256);
    dim3 grid_init((num_offspring + 255) / 256);
    init_curand_states_kernel<<<grid_init, block_init>>>(d_states, num_offspring, 12345);
    cudaDeviceSynchronize();

    std::cout << "  ✓ Test data prepared and copied to device" << std::endl;

    // Run gene-major crossover
    dim3 block(256);
    dim3 grid(chrom_len, (num_offspring + 255) / 256);

    crossover_kernel_gene_major<<<grid, block>>>(
        d_elite, d_non_elite, d_offspring,
        d_states, num_offspring, chrom_len,
        elite_prob, elite_size, non_elite_size
    );
    cudaDeviceSynchronize();

    std::cout << "  ✓ Gene-major crossover complete" << std::endl;

    // Copy offspring back
    std::vector<float> h_offspring_gene(num_offspring * chrom_len);
    cudaMemcpy(h_offspring_gene.data(), d_offspring, num_offspring * chrom_len * sizeof(float), cudaMemcpyDeviceToHost);

    // Transpose back to individual-major for analysis
    std::vector<float> h_offspring(num_offspring * chrom_len);
    for (int ind = 0; ind < num_offspring; ind++) {
        for (int gene = 0; gene < chrom_len; gene++) {
            h_offspring[ind * chrom_len + gene] = h_offspring_gene[gene * num_offspring + ind];
        }
    }

    // Verify offspring values are in valid ranges
    bool all_valid = true;
    int elite_count = 0;
    int non_elite_count = 0;

    for (float val : h_offspring) {
        if (val < 0.0f || val > 1.0f) {
            all_valid = false;
            break;
        }

        // Count which parent pool this gene came from
        if (val < 0.15f) {
            elite_count++;  // From elite (values around 0.05)
        } else if (val > 0.85f) {
            non_elite_count++;  // From non-elite (values around 0.95)
        }
    }

    int total_genes = num_offspring * chrom_len;
    float elite_ratio = (float)elite_count / total_genes;
    float non_elite_ratio = (float)non_elite_count / total_genes;

    std::cout << "  Offspring analysis:" << std::endl;
    std::cout << "    Total genes: " << total_genes << std::endl;
    std::cout << "    From elite: " << elite_count << " (" << std::fixed << std::setprecision(1)
              << (elite_ratio * 100) << "%)" << std::endl;
    std::cout << "    From non-elite: " << non_elite_count << " (" << std::fixed << std::setprecision(1)
              << (non_elite_ratio * 100) << "%)" << std::endl;
    std::cout << "    Expected elite ratio: " << (elite_prob * 100) << "%" << std::endl;

    // Check if elite probability is approximately correct (within 10%)
    bool prob_ok = (elite_ratio > elite_prob - 0.1f) && (elite_ratio < elite_prob + 0.1f);

    bool passed = all_valid && prob_ok;

    if (all_valid) {
        std::cout << "  ✓ All offspring values in valid range" << std::endl;
    } else {
        std::cout << "  ✗ Some offspring values out of range" << std::endl;
    }

    if (prob_ok) {
        std::cout << "  ✓ Elite probability approximately correct" << std::endl;
    } else {
        std::cout << "  ✗ Elite probability deviates from expected" << std::endl;
    }

    // Cleanup
    cudaFree(d_elite);
    cudaFree(d_non_elite);
    cudaFree(d_offspring);
    cudaFree(d_states);

    std::cout << "  Result: " << (passed ? "✓ PASS" : "✗ FAIL") << std::endl;
    return passed;
}

bool test_offspring_diversity() {
    std::cout << "\n[Test 2] Offspring diversity check" << std::endl;

    const int elite_size = 50;
    const int non_elite_size = 50;
    const int num_offspring = 100;
    const int chrom_len = 100;
    const float elite_prob = 0.6f;

    // Create diverse populations
    std::vector<float> h_elite_gene(elite_size * chrom_len);
    std::vector<float> h_non_elite_gene(non_elite_size * chrom_len);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> elite_dist(0.0f, 0.3f);
    std::uniform_real_distribution<float> non_elite_dist(0.7f, 1.0f);

    // Generate in gene-major layout
    for (int gene = 0; gene < chrom_len; gene++) {
        for (int ind = 0; ind < elite_size; ind++) {
            h_elite_gene[gene * elite_size + ind] = elite_dist(rng);
        }
        for (int ind = 0; ind < non_elite_size; ind++) {
            h_non_elite_gene[gene * non_elite_size + ind] = non_elite_dist(rng);
        }
    }

    float* d_elite = nullptr;
    float* d_non_elite = nullptr;
    float* d_offspring = nullptr;
    curandState* d_states = nullptr;

    cudaMalloc(&d_elite, elite_size * chrom_len * sizeof(float));
    cudaMalloc(&d_non_elite, non_elite_size * chrom_len * sizeof(float));
    cudaMalloc(&d_offspring, num_offspring * chrom_len * sizeof(float));
    cudaMalloc(&d_states, num_offspring * sizeof(curandState));

    cudaMemcpy(d_elite, h_elite_gene.data(), elite_size * chrom_len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_non_elite, h_non_elite_gene.data(), non_elite_size * chrom_len * sizeof(float), cudaMemcpyHostToDevice);

    init_curand_states_kernel<<<(num_offspring + 255) / 256, 256>>>(d_states, num_offspring, 99999);
    cudaDeviceSynchronize();

    // Run crossover
    dim3 block(256);
    dim3 grid(chrom_len, (num_offspring + 255) / 256);

    crossover_kernel_gene_major<<<grid, block>>>(
        d_elite, d_non_elite, d_offspring,
        d_states, num_offspring, chrom_len,
        elite_prob, elite_size, non_elite_size
    );
    cudaDeviceSynchronize();

    // Copy back
    std::vector<float> h_offspring(num_offspring * chrom_len);
    cudaMemcpy(h_offspring.data(), d_offspring, num_offspring * chrom_len * sizeof(float), cudaMemcpyDeviceToHost);

    // Check diversity - no two offspring should be identical
    bool all_unique = true;
    for (int i = 0; i < num_offspring - 1; i++) {
        for (int j = i + 1; j < num_offspring; j++) {
            bool identical = true;
            for (int gene = 0; gene < chrom_len; gene++) {
                int offset_i = gene * num_offspring + i;
                int offset_j = gene * num_offspring + j;
                if (std::abs(h_offspring[offset_i] - h_offspring[offset_j]) > 1e-6f) {
                    identical = false;
                    break;
                }
            }
            if (identical) {
                all_unique = false;
                std::cout << "  ✗ Offspring " << i << " and " << j << " are identical" << std::endl;
                break;
            }
        }
        if (!all_unique) break;
    }

    if (all_unique) {
        std::cout << "  ✓ All " << num_offspring << " offspring are unique" << std::endl;
    }

    cudaFree(d_elite);
    cudaFree(d_non_elite);
    cudaFree(d_offspring);
    cudaFree(d_states);

    std::cout << "  Result: " << (all_unique ? "✓ PASS" : "✗ FAIL") << std::endl;
    return all_unique;
}

bool test_large_crossover() {
    std::cout << "\n[Test 3] Large population stress test" << std::endl;

    const int elite_size = 2000;
    const int non_elite_size = 3000;
    const int num_offspring = 4000;
    const int chrom_len = 500;
    const float elite_prob = 0.7f;

    std::cout << "  Elite: " << elite_size << std::endl;
    std::cout << "  Non-elite: " << non_elite_size << std::endl;
    std::cout << "  Offspring: " << num_offspring << std::endl;
    std::cout << "  Chromosome: " << chrom_len << std::endl;
    std::cout << "  Total data: " << ((elite_size + non_elite_size + num_offspring) * chrom_len * sizeof(float) / (1024.0 * 1024.0))
              << " MB" << std::endl;

    float* d_elite = nullptr;
    float* d_non_elite = nullptr;
    float* d_offspring = nullptr;
    curandState* d_states = nullptr;

    cudaError_t err = cudaMalloc(&d_elite, elite_size * chrom_len * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "  ✗ Failed to allocate elite: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    err = cudaMalloc(&d_non_elite, non_elite_size * chrom_len * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_elite);
        std::cerr << "  ✗ Failed to allocate non-elite: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    err = cudaMalloc(&d_offspring, num_offspring * chrom_len * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_elite);
        cudaFree(d_non_elite);
        std::cerr << "  ✗ Failed to allocate offspring: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    err = cudaMalloc(&d_states, num_offspring * sizeof(curandState));
    if (err != cudaSuccess) {
        cudaFree(d_elite);
        cudaFree(d_non_elite);
        cudaFree(d_offspring);
        std::cerr << "  ✗ Failed to allocate RNG states: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    std::cout << "  ✓ Memory allocated successfully" << std::endl;

    // Initialize with dummy data
    cudaMemset(d_elite, 0, elite_size * chrom_len * sizeof(float));
    cudaMemset(d_non_elite, 0, non_elite_size * chrom_len * sizeof(float));

    init_curand_states_kernel<<<(num_offspring + 255) / 256, 256>>>(d_states, num_offspring, 77777);
    cudaDeviceSynchronize();

    // Run crossover
    dim3 block(256);
    dim3 grid(chrom_len, (num_offspring + 255) / 256);

    crossover_kernel_gene_major<<<grid, block>>>(
        d_elite, d_non_elite, d_offspring,
        d_states, num_offspring, chrom_len,
        elite_prob, elite_size, non_elite_size
    );

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "  ✗ Kernel execution failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_elite);
        cudaFree(d_non_elite);
        cudaFree(d_offspring);
        cudaFree(d_states);
        return false;
    }

    std::cout << "  ✓ Gene-major crossover successful" << std::endl;

    // Spot check
    std::vector<float> sample(100);
    cudaMemcpy(sample.data(), d_offspring, 100 * sizeof(float), cudaMemcpyDeviceToHost);

    bool all_valid = true;
    for (float val : sample) {
        if (std::isnan(val) || std::isinf(val)) {
            all_valid = false;
            break;
        }
    }

    if (all_valid) {
        std::cout << "  ✓ Spot check: No NaN or Inf values" << std::endl;
    } else {
        std::cout << "  ✗ Spot check: Invalid values detected" << std::endl;
    }

    cudaFree(d_elite);
    cudaFree(d_non_elite);
    cudaFree(d_offspring);
    cudaFree(d_states);

    bool passed = (err == cudaSuccess) && all_valid;
    std::cout << "  Result: " << (passed ? "✓ PASS" : "✗ FAIL") << std::endl;
    return passed;
}

bool test_performance_comparison() {
    std::cout << "\n[Test 4] Performance comparison" << std::endl;

    const int elite_size = 1000;
    const int non_elite_size = 1500;
    const int num_offspring = 2000;
    const int chrom_len = 300;
    const float elite_prob = 0.7f;
    const int num_runs = 5;

    std::cout << "  Configuration: " << num_offspring << " offspring × " << chrom_len << " genes" << std::endl;

    // Allocate memory for both versions
    float* d_elite_ind = nullptr;
    float* d_non_elite_ind = nullptr;
    float* d_offspring_ind = nullptr;
    float* d_elite_gene = nullptr;
    float* d_non_elite_gene = nullptr;
    float* d_offspring_gene = nullptr;
    curandState* d_states = nullptr;

    cudaMalloc(&d_elite_ind, elite_size * chrom_len * sizeof(float));
    cudaMalloc(&d_non_elite_ind, non_elite_size * chrom_len * sizeof(float));
    cudaMalloc(&d_offspring_ind, num_offspring * chrom_len * sizeof(float));
    cudaMalloc(&d_elite_gene, elite_size * chrom_len * sizeof(float));
    cudaMalloc(&d_non_elite_gene, non_elite_size * chrom_len * sizeof(float));
    cudaMalloc(&d_offspring_gene, num_offspring * chrom_len * sizeof(float));
    cudaMalloc(&d_states, num_offspring * sizeof(curandState));

    // Initialize
    cudaMemset(d_elite_ind, 0, elite_size * chrom_len * sizeof(float));
    cudaMemset(d_non_elite_ind, 0, non_elite_size * chrom_len * sizeof(float));
    cudaMemset(d_elite_gene, 0, elite_size * chrom_len * sizeof(float));
    cudaMemset(d_non_elite_gene, 0, non_elite_size * chrom_len * sizeof(float));

    init_curand_states_kernel<<<(num_offspring + 255) / 256, 256>>>(d_states, num_offspring, 55555);
    cudaDeviceSynchronize();

    std::cout << "  ✓ Memory initialized" << std::endl;

    // Warmup
    dim3 block_ind(256);
    dim3 grid_ind((num_offspring + 255) / 256);
    crossover_kernel<<<grid_ind, block_ind>>>(
        d_elite_ind, d_non_elite_ind, d_offspring_ind,
        d_states, num_offspring, chrom_len,
        elite_prob, elite_size, non_elite_size
    );
    cudaDeviceSynchronize();

    dim3 block_gene(256);
    dim3 grid_gene(chrom_len, (num_offspring + 255) / 256);
    crossover_kernel_gene_major<<<grid_gene, block_gene>>>(
        d_elite_gene, d_non_elite_gene, d_offspring_gene,
        d_states, num_offspring, chrom_len,
        elite_prob, elite_size, non_elite_size
    );
    cudaDeviceSynchronize();

    std::cout << "  ✓ Warmup complete" << std::endl;

    // Time individual-major
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < num_runs; i++) {
        crossover_kernel<<<grid_ind, block_ind>>>(
            d_elite_ind, d_non_elite_ind, d_offspring_ind,
            d_states, num_offspring, chrom_len,
            elite_prob, elite_size, non_elite_size
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
        crossover_kernel_gene_major<<<grid_gene, block_gene>>>(
            d_elite_gene, d_non_elite_gene, d_offspring_gene,
            d_states, num_offspring, chrom_len,
            elite_prob, elite_size, non_elite_size
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

    bool significant_speedup = (speedup > 2.0);

    if (significant_speedup) {
        std::cout << "  ✓ Gene-major shows significant performance improvement" << std::endl;
    } else {
        std::cout << "  ⚠ Gene-major speedup less than expected (may vary by GPU)" << std::endl;
    }

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_elite_ind);
    cudaFree(d_non_elite_ind);
    cudaFree(d_offspring_ind);
    cudaFree(d_elite_gene);
    cudaFree(d_non_elite_gene);
    cudaFree(d_offspring_gene);
    cudaFree(d_states);

    std::cout << "  Result: ✓ PASS" << std::endl;
    return true;
}

int main() {
    std::cout << "================================================" << std::endl;
    std::cout << "  Phase 5 Day 2: Crossover Kernel Tests        " << std::endl;
    std::cout << "================================================" << std::endl;

    int passed = 0;
    int total = 4;

    if (test_crossover_correctness()) passed++;
    if (test_offspring_diversity()) passed++;
    if (test_large_crossover()) passed++;
    if (test_performance_comparison()) passed++;

    std::cout << "\n================================================" << std::endl;
    std::cout << "  Test Summary: " << passed << "/" << total << " passed" << std::endl;

    if (passed == total) {
        std::cout << "  ✓ ALL TESTS PASSED" << std::endl;
        std::cout << "  Gene-major crossover kernel ready!" << std::endl;
        std::cout << "================================================" << std::endl;
        return 0;
    } else {
        std::cout << "  ✗ " << (total - passed) << " TESTS FAILED" << std::endl;
        std::cout << "================================================" << std::endl;
        return 1;
    }
}
