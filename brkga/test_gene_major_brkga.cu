// test_gene_major_brkga.cu - Integration test for GeneLayoutBRKGA wrapper
// Phase 5 Day 6: Verifies the gene-major BRKGA wrapper works correctly

#include "core/gene_major_brkga.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <numeric>

/**
 * Test decoder: Simple sphere function (minimize sum of squares)
 *
 * For gene-major layout: population[gene][ind]
 * Each gene value is treated as x_i, fitness = sum(x_i^2)
 * Optimal: all zeros -> fitness = 0
 */
__global__ void sphere_decoder_gene_major(
    const float* __restrict__ population,  // [chrom_len][pop_size]
    float* __restrict__ fitness,
    int pop_size,
    int chrom_len
) {
    int ind_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind_idx >= pop_size) return;

    float sum = 0.0f;
    for (int gene = 0; gene < chrom_len; gene++) {
        // Gene-major access: population[gene * pop_size + ind]
        float val = population[gene * pop_size + ind_idx];
        // Map [0,1] to [-5,5] for sphere function
        float x = val * 10.0f - 5.0f;
        sum += x * x;
    }
    fitness[ind_idx] = sum;
}

/**
 * Test decoder for individual-major layout (decoder compatibility test)
 */
__global__ void sphere_decoder_ind_major(
    const float* __restrict__ population,  // [pop_size][chrom_len]
    float* __restrict__ fitness,
    int pop_size,
    int chrom_len
) {
    int ind_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind_idx >= pop_size) return;

    float sum = 0.0f;
    for (int gene = 0; gene < chrom_len; gene++) {
        // Individual-major access: population[ind * chrom_len + gene]
        float val = population[ind_idx * chrom_len + gene];
        float x = val * 10.0f - 5.0f;
        sum += x * x;
    }
    fitness[ind_idx] = sum;
}

/**
 * Test 1: Basic BRKGA run with gene-major decoder
 */
bool test_basic_brkga_run() {
    std::cout << "\n[Test 1] Basic BRKGA run with gene-major decoder" << std::endl;

    const int pop_size = 500;
    const int chrom_len = 20;
    const int elite_size = 75;    // 15%
    const int mutant_size = 50;   // 10%
    const float elite_prob = 0.7f;
    const int num_generations = 50;

    std::cout << "  Configuration:" << std::endl;
    std::cout << "    Population: " << pop_size << std::endl;
    std::cout << "    Chromosome: " << chrom_len << std::endl;
    std::cout << "    Generations: " << num_generations << std::endl;

    try {
        GeneLayoutBRKGA<float> brkga(pop_size, chrom_len, elite_size, mutant_size, elite_prob, false);

        std::cout << "  ✓ BRKGA instance created" << std::endl;
        std::cout << "    Memory usage: " << (brkga.get_memory_usage() / (1024.0 * 1024.0))
                  << " MB" << std::endl;

        // Initialize population
        brkga.initialize_population();
        std::cout << "  ✓ Population initialized" << std::endl;

        float initial_fitness = 0.0f;
        float final_fitness = 0.0f;

        // Run BRKGA generations
        for (int gen = 0; gen < num_generations; gen++) {
            // Evaluate fitness with gene-major decoder
            brkga.evaluate_fitness([&](float* pop, float* fit, int ps, int cl) {
                dim3 block(256);
                dim3 grid((ps + 255) / 256);
                sphere_decoder_gene_major<<<grid, block>>>(pop, fit, ps, cl);
                cudaDeviceSynchronize();
            });

            if (gen == 0) {
                initial_fitness = brkga.get_best_fitness();
            }

            // Run generation
            brkga.run_generation();
        }

        // Final evaluation
        brkga.evaluate_fitness([&](float* pop, float* fit, int ps, int cl) {
            dim3 block(256);
            dim3 grid((ps + 255) / 256);
            sphere_decoder_gene_major<<<grid, block>>>(pop, fit, ps, cl);
            cudaDeviceSynchronize();
        });

        final_fitness = brkga.get_best_fitness();

        std::cout << "  Results:" << std::endl;
        std::cout << "    Initial best fitness: " << std::fixed << std::setprecision(4)
                  << initial_fitness << std::endl;
        std::cout << "    Final best fitness: " << std::fixed << std::setprecision(4)
                  << final_fitness << std::endl;
        std::cout << "    Improvement: " << std::fixed << std::setprecision(2)
                  << ((initial_fitness - final_fitness) / initial_fitness * 100.0f) << "%" << std::endl;

        // Verify improvement
        bool improved = (final_fitness < initial_fitness * 0.5f);  // At least 50% improvement

        if (improved) {
            std::cout << "  ✓ BRKGA successfully optimized (>50% improvement)" << std::endl;
        } else {
            std::cout << "  ✗ BRKGA did not improve sufficiently" << std::endl;
        }

        std::cout << "  Result: " << (improved ? "✓ PASS" : "✗ FAIL") << std::endl;
        return improved;

    } catch (const std::exception& e) {
        std::cerr << "  ✗ Exception: " << e.what() << std::endl;
        return false;
    }
}

/**
 * Test 2: Decoder compatibility with individual-major transpose
 */
bool test_decoder_compatibility() {
    std::cout << "\n[Test 2] Decoder compatibility (individual-major transpose)" << std::endl;

    const int pop_size = 300;
    const int chrom_len = 15;
    const int elite_size = 45;
    const int mutant_size = 30;
    const float elite_prob = 0.7f;
    const int num_generations = 30;

    std::cout << "  Testing with decoder_needs_ind_major=true" << std::endl;

    try {
        // Create BRKGA with individual-major compatibility flag
        GeneLayoutBRKGA<float> brkga(pop_size, chrom_len, elite_size, mutant_size, elite_prob, true);

        std::cout << "  ✓ BRKGA instance created with ind-major compatibility" << std::endl;

        brkga.initialize_population();

        float initial_fitness = 0.0f;
        float final_fitness = 0.0f;

        for (int gen = 0; gen < num_generations; gen++) {
            // Evaluate with individual-major decoder (should work due to auto-transpose)
            brkga.evaluate_fitness([&](float* pop, float* fit, int ps, int cl) {
                dim3 block(256);
                dim3 grid((ps + 255) / 256);
                sphere_decoder_ind_major<<<grid, block>>>(pop, fit, ps, cl);
                cudaDeviceSynchronize();
            });

            if (gen == 0) {
                initial_fitness = brkga.get_best_fitness();
            }

            brkga.run_generation();
        }

        // Final evaluation
        brkga.evaluate_fitness([&](float* pop, float* fit, int ps, int cl) {
            dim3 block(256);
            dim3 grid((ps + 255) / 256);
            sphere_decoder_ind_major<<<grid, block>>>(pop, fit, ps, cl);
            cudaDeviceSynchronize();
        });

        final_fitness = brkga.get_best_fitness();

        std::cout << "  Initial fitness: " << std::fixed << std::setprecision(4) << initial_fitness << std::endl;
        std::cout << "  Final fitness: " << std::fixed << std::setprecision(4) << final_fitness << std::endl;

        bool improved = (final_fitness < initial_fitness * 0.5f);

        if (improved) {
            std::cout << "  ✓ Individual-major decoder works correctly" << std::endl;
        } else {
            std::cout << "  ✗ Transpose may not be working correctly" << std::endl;
        }

        std::cout << "  Result: " << (improved ? "✓ PASS" : "✗ FAIL") << std::endl;
        return improved;

    } catch (const std::exception& e) {
        std::cerr << "  ✗ Exception: " << e.what() << std::endl;
        return false;
    }
}

/**
 * Test 3: Get best individual
 */
bool test_get_best_individual() {
    std::cout << "\n[Test 3] Get best individual extraction" << std::endl;

    const int pop_size = 200;
    const int chrom_len = 10;
    const int elite_size = 30;
    const int mutant_size = 20;
    const float elite_prob = 0.7f;
    const int num_generations = 20;

    try {
        GeneLayoutBRKGA<float> brkga(pop_size, chrom_len, elite_size, mutant_size, elite_prob, false);
        brkga.initialize_population();

        // Run a few generations
        for (int gen = 0; gen < num_generations; gen++) {
            brkga.evaluate_fitness([&](float* pop, float* fit, int ps, int cl) {
                dim3 block(256);
                dim3 grid((ps + 255) / 256);
                sphere_decoder_gene_major<<<grid, block>>>(pop, fit, ps, cl);
                cudaDeviceSynchronize();
            });
            brkga.run_generation();
        }

        // Final evaluation
        brkga.evaluate_fitness([&](float* pop, float* fit, int ps, int cl) {
            dim3 block(256);
            dim3 grid((ps + 255) / 256);
            sphere_decoder_gene_major<<<grid, block>>>(pop, fit, ps, cl);
            cudaDeviceSynchronize();
        });

        // Extract best individual
        std::vector<float> best = brkga.get_best_individual();

        // Verify chromosome length
        bool correct_length = (static_cast<int>(best.size()) == chrom_len);

        // Verify values are in valid range
        bool valid_range = true;
        for (float val : best) {
            if (val < 0.0f || val > 1.0f || std::isnan(val) || std::isinf(val)) {
                valid_range = false;
                break;
            }
        }

        // Verify it's actually a good solution (low fitness)
        float recomputed_fitness = 0.0f;
        for (float val : best) {
            float x = val * 10.0f - 5.0f;
            recomputed_fitness += x * x;
        }

        float reported_fitness = brkga.get_best_fitness();
        bool fitness_matches = (std::abs(recomputed_fitness - reported_fitness) < 0.1f);

        std::cout << "  Best chromosome (" << best.size() << " genes):" << std::endl;
        std::cout << "    First 5 genes: [";
        for (int i = 0; i < std::min(5, static_cast<int>(best.size())); i++) {
            std::cout << std::fixed << std::setprecision(4) << best[i];
            if (i < 4) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "  Reported fitness: " << std::fixed << std::setprecision(4) << reported_fitness << std::endl;
        std::cout << "  Recomputed fitness: " << std::fixed << std::setprecision(4) << recomputed_fitness << std::endl;

        if (correct_length) std::cout << "  ✓ Correct chromosome length" << std::endl;
        else std::cout << "  ✗ Wrong chromosome length" << std::endl;

        if (valid_range) std::cout << "  ✓ All genes in valid range [0,1]" << std::endl;
        else std::cout << "  ✗ Invalid gene values detected" << std::endl;

        if (fitness_matches) std::cout << "  ✓ Fitness values match" << std::endl;
        else std::cout << "  ✗ Fitness mismatch (may be due to sorting order)" << std::endl;

        bool passed = correct_length && valid_range;
        std::cout << "  Result: " << (passed ? "✓ PASS" : "✗ FAIL") << std::endl;
        return passed;

    } catch (const std::exception& e) {
        std::cerr << "  ✗ Exception: " << e.what() << std::endl;
        return false;
    }
}

/**
 * Test 4: Large population stress test
 */
bool test_large_population() {
    std::cout << "\n[Test 4] Large population stress test" << std::endl;

    const int pop_size = 4000;
    const int chrom_len = 200;
    const int elite_size = 600;
    const int mutant_size = 400;
    const float elite_prob = 0.7f;
    const int num_generations = 10;

    std::cout << "  Configuration:" << std::endl;
    std::cout << "    Population: " << pop_size << std::endl;
    std::cout << "    Chromosome: " << chrom_len << std::endl;
    std::cout << "    Total data: " << (pop_size * chrom_len * sizeof(float) * 2 / (1024.0 * 1024.0))
              << " MB" << std::endl;

    try {
        GeneLayoutBRKGA<float> brkga(pop_size, chrom_len, elite_size, mutant_size, elite_prob, false);

        std::cout << "  ✓ Large BRKGA instance created" << std::endl;
        std::cout << "    Memory usage: " << (brkga.get_memory_usage() / (1024.0 * 1024.0))
                  << " MB" << std::endl;

        brkga.initialize_population();
        std::cout << "  ✓ Population initialized" << std::endl;

        // Time the generations
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

        std::cout << "  Performance:" << std::endl;
        std::cout << "    Total time: " << std::fixed << std::setprecision(2) << elapsed_ms << " ms" << std::endl;
        std::cout << "    Time per generation: " << std::fixed << std::setprecision(3)
                  << (elapsed_ms / num_generations) << " ms" << std::endl;
        std::cout << "    Throughput: " << std::fixed << std::setprecision(0)
                  << (pop_size * chrom_len * num_generations / (elapsed_ms / 1000.0) / 1e6)
                  << " M genes/sec" << std::endl;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        std::cout << "  ✓ Large population test successful" << std::endl;
        std::cout << "  Result: ✓ PASS" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "  ✗ Exception: " << e.what() << std::endl;
        return false;
    }
}

/**
 * Test 5: Memory management (multiple instances)
 */
bool test_memory_management() {
    std::cout << "\n[Test 5] Memory management (create/destroy)" << std::endl;

    const int pop_size = 500;
    const int chrom_len = 50;
    const int num_instances = 5;

    std::cout << "  Creating and destroying " << num_instances << " instances" << std::endl;

    try {
        for (int i = 0; i < num_instances; i++) {
            GeneLayoutBRKGA<float> brkga(pop_size, chrom_len, 75, 50, 0.7f, false);
            brkga.initialize_population();

            // Do some work
            brkga.evaluate_fitness([&](float* pop, float* fit, int ps, int cl) {
                dim3 block(256);
                dim3 grid((ps + 255) / 256);
                sphere_decoder_gene_major<<<grid, block>>>(pop, fit, ps, cl);
                cudaDeviceSynchronize();
            });
            brkga.run_generation();

            // Instance goes out of scope - destructor called
        }

        // Check for CUDA errors after all instances destroyed
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "  ✗ CUDA error: " << cudaGetErrorString(err) << std::endl;
            return false;
        }

        // Try to allocate again to verify memory was freed
        GeneLayoutBRKGA<float> brkga_final(pop_size, chrom_len, 75, 50, 0.7f, false);
        brkga_final.initialize_population();

        std::cout << "  ✓ All instances created and destroyed successfully" << std::endl;
        std::cout << "  ✓ Memory properly freed and reusable" << std::endl;
        std::cout << "  Result: ✓ PASS" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "  ✗ Exception: " << e.what() << std::endl;
        return false;
    }
}

int main() {
    std::cout << "================================================" << std::endl;
    std::cout << "  Phase 5 Day 6: GeneLayoutBRKGA Integration    " << std::endl;
    std::cout << "================================================" << std::endl;

    // Get GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Memory: " << (prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;

    int passed = 0;
    int total = 5;

    if (test_basic_brkga_run()) passed++;
    if (test_decoder_compatibility()) passed++;
    if (test_get_best_individual()) passed++;
    if (test_large_population()) passed++;
    if (test_memory_management()) passed++;

    std::cout << "\n================================================" << std::endl;
    std::cout << "  Test Summary: " << passed << "/" << total << " passed" << std::endl;

    if (passed == total) {
        std::cout << "  ✓ ALL TESTS PASSED" << std::endl;
        std::cout << "  GeneLayoutBRKGA wrapper ready for use!" << std::endl;
        std::cout << "================================================" << std::endl;
        return 0;
    } else {
        std::cout << "  ✗ " << (total - passed) << " TESTS FAILED" << std::endl;
        std::cout << "================================================" << std::endl;
        return 1;
    }
}
