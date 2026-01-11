// test_transpose.cu - Unit tests for memory layout transpose operations
// Tests that transpose operations correctly convert between individual-major
// and gene-major layouts without data loss or corruption.

#include "core/memory_layout.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>

/**
 * @brief Test transpose operations for correctness
 *
 * Verifies:
 * 1. Individual-major → Gene-major transpose
 * 2. Gene-major → Individual-major transpose
 * 3. Round-trip consistency (ind→gene→ind preserves data)
 * 4. DualLayoutBuffer class functionality
 */

// Helper function to initialize test data
template<typename T>
void generate_test_population(std::vector<T>& population, int pop_size, int chrom_len, int seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<T> dist(0.0, 1.0);

    population.resize(pop_size * chrom_len);
    for (int i = 0; i < pop_size * chrom_len; i++) {
        population[i] = dist(rng);
    }
}

// Helper function to manually transpose (CPU reference)
template<typename T>
void cpu_transpose_ind_to_gene(const std::vector<T>& src, std::vector<T>& dest, int pop_size, int chrom_len) {
    dest.resize(pop_size * chrom_len);
    for (int ind = 0; ind < pop_size; ind++) {
        for (int gene = 0; gene < chrom_len; gene++) {
            // Individual-major: src[ind * chrom_len + gene]
            // Gene-major: dest[gene * pop_size + ind]
            dest[gene * pop_size + ind] = src[ind * chrom_len + gene];
        }
    }
}

// Helper function to compare two arrays
template<typename T>
bool arrays_match(const std::vector<T>& a, const std::vector<T>& b, T tolerance = 1e-6) {
    if (a.size() != b.size()) return false;

    for (size_t i = 0; i < a.size(); i++) {
        if (std::abs(a[i] - b[i]) > tolerance) {
            std::cerr << "Mismatch at index " << i << ": " << a[i] << " vs " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}

// Test 1: Basic transpose individual-major → gene-major
bool test_ind_to_gene_transpose() {
    std::cout << "\n[Test 1] Individual-major → Gene-major transpose" << std::endl;

    const int pop_size = 100;
    const int chrom_len = 50;

    // Generate test data
    std::vector<float> h_population_ind;
    generate_test_population(h_population_ind, pop_size, chrom_len);

    // CPU reference transpose
    std::vector<float> h_expected_gene;
    cpu_transpose_ind_to_gene(h_population_ind, h_expected_gene, pop_size, chrom_len);

    // GPU transpose
    float* d_ind = nullptr;
    float* d_gene = nullptr;
    cudaMalloc(&d_ind, pop_size * chrom_len * sizeof(float));
    cudaMalloc(&d_gene, pop_size * chrom_len * sizeof(float));

    cudaMemcpy(d_ind, h_population_ind.data(), pop_size * chrom_len * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((pop_size + 255) / 256, chrom_len);
    transpose_ind_to_gene_kernel<<<grid, block>>>(d_ind, d_gene, pop_size, chrom_len);
    cudaDeviceSynchronize();

    std::vector<float> h_result_gene(pop_size * chrom_len);
    cudaMemcpy(h_result_gene.data(), d_gene, pop_size * chrom_len * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_ind);
    cudaFree(d_gene);

    bool passed = arrays_match(h_expected_gene, h_result_gene);
    std::cout << "  Result: " << (passed ? "✓ PASS" : "✗ FAIL") << std::endl;

    return passed;
}

// Test 2: Basic transpose gene-major → individual-major
bool test_gene_to_ind_transpose() {
    std::cout << "\n[Test 2] Gene-major → Individual-major transpose" << std::endl;

    const int pop_size = 100;
    const int chrom_len = 50;

    // Generate test data in gene-major layout
    std::vector<float> h_population_gene;
    generate_test_population(h_population_gene, pop_size, chrom_len);

    // CPU reference transpose (gene → ind is reverse of ind → gene)
    std::vector<float> h_expected_ind(pop_size * chrom_len);
    for (int gene = 0; gene < chrom_len; gene++) {
        for (int ind = 0; ind < pop_size; ind++) {
            // Gene-major: src[gene * pop_size + ind]
            // Individual-major: dest[ind * chrom_len + gene]
            h_expected_ind[ind * chrom_len + gene] = h_population_gene[gene * pop_size + ind];
        }
    }

    // GPU transpose
    float* d_gene = nullptr;
    float* d_ind = nullptr;
    cudaMalloc(&d_gene, pop_size * chrom_len * sizeof(float));
    cudaMalloc(&d_ind, pop_size * chrom_len * sizeof(float));

    cudaMemcpy(d_gene, h_population_gene.data(), pop_size * chrom_len * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((pop_size + 255) / 256, chrom_len);
    transpose_gene_to_ind_kernel<<<grid, block>>>(d_gene, d_ind, pop_size, chrom_len);
    cudaDeviceSynchronize();

    std::vector<float> h_result_ind(pop_size * chrom_len);
    cudaMemcpy(h_result_ind.data(), d_ind, pop_size * chrom_len * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_gene);
    cudaFree(d_ind);

    bool passed = arrays_match(h_expected_ind, h_result_ind);
    std::cout << "  Result: " << (passed ? "✓ PASS" : "✗ FAIL") << std::endl;

    return passed;
}

// Test 3: Round-trip consistency (ind → gene → ind)
bool test_round_trip_consistency() {
    std::cout << "\n[Test 3] Round-trip consistency (ind → gene → ind)" << std::endl;

    const int pop_size = 200;
    const int chrom_len = 100;

    // Generate test data
    std::vector<float> h_original;
    generate_test_population(h_original, pop_size, chrom_len);

    // GPU round-trip: ind → gene → ind
    float* d_ind1 = nullptr;
    float* d_gene = nullptr;
    float* d_ind2 = nullptr;
    cudaMalloc(&d_ind1, pop_size * chrom_len * sizeof(float));
    cudaMalloc(&d_gene, pop_size * chrom_len * sizeof(float));
    cudaMalloc(&d_ind2, pop_size * chrom_len * sizeof(float));

    cudaMemcpy(d_ind1, h_original.data(), pop_size * chrom_len * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((pop_size + 255) / 256, chrom_len);

    // ind → gene
    transpose_ind_to_gene_kernel<<<grid, block>>>(d_ind1, d_gene, pop_size, chrom_len);

    // gene → ind
    transpose_gene_to_ind_kernel<<<grid, block>>>(d_gene, d_ind2, pop_size, chrom_len);

    cudaDeviceSynchronize();

    std::vector<float> h_result(pop_size * chrom_len);
    cudaMemcpy(h_result.data(), d_ind2, pop_size * chrom_len * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_ind1);
    cudaFree(d_gene);
    cudaFree(d_ind2);

    bool passed = arrays_match(h_original, h_result);
    std::cout << "  Result: " << (passed ? "✓ PASS" : "✗ FAIL") << std::endl;

    return passed;
}

// Test 4: DualLayoutBuffer class functionality
bool test_dual_layout_buffer() {
    std::cout << "\n[Test 4] DualLayoutBuffer class functionality" << std::endl;

    const int pop_size = 150;
    const int chrom_len = 75;

    // Generate test data
    std::vector<float> h_population;
    generate_test_population(h_population, pop_size, chrom_len);

    try {
        // Create dual-layout buffer
        DualLayoutBuffer<float> buffer(pop_size, chrom_len);

        std::cout << "  ✓ Buffer created successfully" << std::endl;
        std::cout << "    Memory usage: " << (buffer.get_memory_usage() / (1024.0 * 1024.0))
                  << " MB" << std::endl;

        // Initialize individual-major buffer
        cudaMemcpy(buffer.get_individual_major(), h_population.data(),
                   pop_size * chrom_len * sizeof(float), cudaMemcpyHostToDevice);

        // Check initial layout
        if (buffer.get_current_layout() != MemoryLayout::INDIVIDUAL_MAJOR) {
            std::cout << "  ✗ Initial layout is not INDIVIDUAL_MAJOR" << std::endl;
            return false;
        }
        std::cout << "  ✓ Initial layout: INDIVIDUAL_MAJOR" << std::endl;

        // Transpose to gene-major
        buffer.transpose_to_gene_major();

        if (buffer.get_current_layout() != MemoryLayout::GENE_MAJOR) {
            std::cout << "  ✗ Layout not updated to GENE_MAJOR" << std::endl;
            return false;
        }
        std::cout << "  ✓ Transposed to GENE_MAJOR" << std::endl;

        // Transpose back to individual-major
        buffer.transpose_to_individual_major();

        if (buffer.get_current_layout() != MemoryLayout::INDIVIDUAL_MAJOR) {
            std::cout << "  ✗ Layout not updated to INDIVIDUAL_MAJOR" << std::endl;
            return false;
        }
        std::cout << "  ✓ Transposed back to INDIVIDUAL_MAJOR" << std::endl;

        // Verify data integrity after round-trip
        std::vector<float> h_result(pop_size * chrom_len);
        cudaMemcpy(h_result.data(), buffer.get_individual_major(),
                   pop_size * chrom_len * sizeof(float), cudaMemcpyDeviceToHost);

        bool data_intact = arrays_match(h_population, h_result);
        if (!data_intact) {
            std::cout << "  ✗ Data corrupted after round-trip" << std::endl;
            return false;
        }
        std::cout << "  ✓ Data integrity preserved after round-trip" << std::endl;

        // Test idempotent transpose (should do nothing if already in target layout)
        buffer.transpose_to_individual_major();  // Already in individual-major
        if (buffer.get_current_layout() != MemoryLayout::INDIVIDUAL_MAJOR) {
            std::cout << "  ✗ Idempotent transpose failed" << std::endl;
            return false;
        }
        std::cout << "  ✓ Idempotent transpose works correctly" << std::endl;

        std::cout << "  Result: ✓ PASS" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "  ✗ Exception: " << e.what() << std::endl;
        std::cout << "  Result: ✗ FAIL" << std::endl;
        return false;
    }
}

// Test 5: Large population stress test
bool test_large_population() {
    std::cout << "\n[Test 5] Large population stress test" << std::endl;

    const int pop_size = 8000;
    const int chrom_len = 1000;

    std::cout << "  Population: " << pop_size << " individuals" << std::endl;
    std::cout << "  Chromosome: " << chrom_len << " genes" << std::endl;
    std::cout << "  Total data: " << (pop_size * chrom_len * sizeof(float) / (1024.0 * 1024.0))
              << " MB" << std::endl;

    // Generate test data
    std::vector<float> h_population;
    generate_test_population(h_population, pop_size, chrom_len);

    try {
        DualLayoutBuffer<float> buffer(pop_size, chrom_len);

        std::cout << "  Buffer memory: " << (buffer.get_memory_usage() / (1024.0 * 1024.0))
                  << " MB" << std::endl;

        // Initialize
        cudaMemcpy(buffer.get_individual_major(), h_population.data(),
                   pop_size * chrom_len * sizeof(float), cudaMemcpyHostToDevice);

        // Round-trip
        buffer.transpose_to_gene_major();
        buffer.transpose_to_individual_major();

        // Verify
        std::vector<float> h_result(pop_size * chrom_len);
        cudaMemcpy(h_result.data(), buffer.get_individual_major(),
                   pop_size * chrom_len * sizeof(float), cudaMemcpyDeviceToHost);

        bool passed = arrays_match(h_population, h_result);
        std::cout << "  Result: " << (passed ? "✓ PASS" : "✗ FAIL") << std::endl;

        return passed;

    } catch (const std::exception& e) {
        std::cerr << "  ✗ Exception: " << e.what() << std::endl;
        std::cout << "  Result: ✗ FAIL" << std::endl;
        return false;
    }
}

// Test 6: Spot check values to verify correct transpose
bool test_spot_check_values() {
    std::cout << "\n[Test 6] Spot check transpose values" << std::endl;

    const int pop_size = 5;
    const int chrom_len = 4;

    // Create simple test data for easy verification
    std::vector<float> h_ind(pop_size * chrom_len);
    for (int ind = 0; ind < pop_size; ind++) {
        for (int gene = 0; gene < chrom_len; gene++) {
            h_ind[ind * chrom_len + gene] = ind * 10.0f + gene;
        }
    }

    // Display individual-major layout
    std::cout << "\n  Individual-major layout:" << std::endl;
    for (int ind = 0; ind < pop_size; ind++) {
        std::cout << "    Ind" << ind << ": [";
        for (int gene = 0; gene < chrom_len; gene++) {
            std::cout << std::setw(5) << std::fixed << std::setprecision(1)
                      << h_ind[ind * chrom_len + gene];
            if (gene < chrom_len - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    // GPU transpose
    float* d_ind = nullptr;
    float* d_gene = nullptr;
    cudaMalloc(&d_ind, pop_size * chrom_len * sizeof(float));
    cudaMalloc(&d_gene, pop_size * chrom_len * sizeof(float));

    cudaMemcpy(d_ind, h_ind.data(), pop_size * chrom_len * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((pop_size + 255) / 256, chrom_len);
    transpose_ind_to_gene_kernel<<<grid, block>>>(d_ind, d_gene, pop_size, chrom_len);
    cudaDeviceSynchronize();

    std::vector<float> h_gene(pop_size * chrom_len);
    cudaMemcpy(h_gene.data(), d_gene, pop_size * chrom_len * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_ind);
    cudaFree(d_gene);

    // Display gene-major layout
    std::cout << "\n  Gene-major layout:" << std::endl;
    for (int gene = 0; gene < chrom_len; gene++) {
        std::cout << "    Gene" << gene << ": [";
        for (int ind = 0; ind < pop_size; ind++) {
            std::cout << std::setw(5) << std::fixed << std::setprecision(1)
                      << h_gene[gene * pop_size + ind];
            if (ind < pop_size - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    // Verify: Each gene should have values [0, 10, 20, 30, 40] + gene_offset
    bool passed = true;
    for (int gene = 0; gene < chrom_len; gene++) {
        for (int ind = 0; ind < pop_size; ind++) {
            float expected = ind * 10.0f + gene;
            float actual = h_gene[gene * pop_size + ind];
            if (std::abs(expected - actual) > 1e-5) {
                std::cerr << "  ✗ Mismatch at gene=" << gene << ", ind=" << ind
                          << ": expected " << expected << ", got " << actual << std::endl;
                passed = false;
            }
        }
    }

    std::cout << "\n  Result: " << (passed ? "✓ PASS" : "✗ FAIL") << std::endl;
    return passed;
}

int main() {
    std::cout << "===========================================" << std::endl;
    std::cout << "  Phase 5: Memory Layout Transpose Tests  " << std::endl;
    std::cout << "===========================================" << std::endl;

    int passed = 0;
    int total = 6;

    if (test_spot_check_values()) passed++;
    if (test_ind_to_gene_transpose()) passed++;
    if (test_gene_to_ind_transpose()) passed++;
    if (test_round_trip_consistency()) passed++;
    if (test_dual_layout_buffer()) passed++;
    if (test_large_population()) passed++;

    std::cout << "\n===========================================" << std::endl;
    std::cout << "  Test Summary: " << passed << "/" << total << " passed" << std::endl;

    if (passed == total) {
        std::cout << "  ✓ ALL TESTS PASSED" << std::endl;
        std::cout << "  Memory layout infrastructure ready!" << std::endl;
        std::cout << "===========================================" << std::endl;
        return 0;
    } else {
        std::cout << "  ✗ " << (total - passed) << " TESTS FAILED" << std::endl;
        std::cout << "===========================================" << std::endl;
        return 1;
    }
}
