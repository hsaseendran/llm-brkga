// test_segmented_sort.cu - Test Phase 4 segmented sorting implementation
// Verifies that parallel segmented sort produces identical results to sequential thrust::sort

#include "utils/segmented_sort.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>

template<typename T>
bool arrays_equal(const std::vector<T>& a, const std::vector<T>& b, T tolerance = 1e-5) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        if (std::abs(a[i] - b[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

int main() {
    std::cout << "==========================================" << std::endl;
    std::cout << "  Phase 4: Segmented Sort Test           " << std::endl;
    std::cout << "==========================================" << std::endl;

    try {
        const int num_segments = 100;    // Simulate 100 TSP tours
        const int segment_size = 500;    // Each with 500 cities
        const int total_items = num_segments * segment_size;

        std::cout << "\n[Test Setup]" << std::endl;
        std::cout << "  Segments (tours): " << num_segments << std::endl;
        std::cout << "  Segment size (cities): " << segment_size << std::endl;
        std::cout << "  Total items: " << total_items << std::endl;

        // Generate random keys (chromosomes)
        std::cout << "\n[Generating Random Data]" << std::endl;
        std::vector<float> h_keys(total_items);
        std::vector<int> h_values(total_items);

        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        for (int i = 0; i < total_items; i++) {
            h_keys[i] = dist(rng);
            h_values[i] = i % segment_size;  // Tour indices
        }

        // Reference: Sequential thrust::sort for each segment
        std::cout << "  Running reference sequential sort..." << std::endl;
        std::vector<float> ref_keys = h_keys;
        std::vector<int> ref_values = h_values;

        for (int seg = 0; seg < num_segments; seg++) {
            int offset = seg * segment_size;

            // Create pairs for sorting
            std::vector<std::pair<float, int>> pairs(segment_size);
            for (int i = 0; i < segment_size; i++) {
                pairs[i] = {ref_keys[offset + i], ref_values[offset + i]};
            }

            // CPU sort
            std::sort(pairs.begin(), pairs.end());

            // Write back
            for (int i = 0; i < segment_size; i++) {
                ref_keys[offset + i] = pairs[i].first;
                ref_values[offset + i] = pairs[i].second;
            }
        }
        std::cout << "  ✓ Reference sort complete" << std::endl;

        // Test: Segmented sort using CUB
        std::cout << "\n[Testing Segmented Sort]" << std::endl;

        // Allocate device memory
        float* d_keys;
        int* d_values;
        cudaMalloc(&d_keys, total_items * sizeof(float));
        cudaMalloc(&d_values, total_items * sizeof(int));

        cudaMemcpy(d_keys, h_keys.data(), total_items * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_values, h_values.data(), total_items * sizeof(int), cudaMemcpyHostToDevice);

        // Create segmented sorter
        SegmentedSorter<float, int> sorter(num_segments);
        std::cout << "  ✓ SegmentedSorter created" << std::endl;

        // Prepare segment offsets
        std::vector<int> offsets(num_segments + 1);
        for (int i = 0; i <= num_segments; i++) {
            offsets[i] = i * segment_size;
        }

        // Run segmented sort
        std::cout << "  Running parallel segmented sort..." << std::endl;
        sorter.sort_segments(d_keys, d_values, offsets, num_segments);
        cudaDeviceSynchronize();
        std::cout << "  ✓ Segmented sort complete" << std::endl;

        // Copy results back
        std::vector<float> test_keys(total_items);
        std::vector<int> test_values(total_items);
        cudaMemcpy(test_keys.data(), d_keys, total_items * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(test_values.data(), d_values, total_items * sizeof(int), cudaMemcpyDeviceToHost);

        // Cleanup
        cudaFree(d_keys);
        cudaFree(d_values);

        // Verification
        std::cout << "\n[Verification]" << std::endl;

        bool keys_match = arrays_equal(ref_keys, test_keys);
        bool values_match = arrays_equal(ref_values, test_values);

        std::cout << "  Keys match: " << (keys_match ? "✓ YES" : "✗ NO") << std::endl;
        std::cout << "  Values match: " << (values_match ? "✓ YES" : "✗ NO") << std::endl;

        // Detailed verification: Check a few random segments
        std::cout << "\n[Spot Check]" << std::endl;
        std::vector<int> check_segments = {0, num_segments / 2, num_segments - 1};
        bool all_segments_ok = true;

        for (int seg : check_segments) {
            int offset = seg * segment_size;
            bool segment_sorted = true;

            // Check if segment is sorted
            for (int i = 0; i < segment_size - 1; i++) {
                if (test_keys[offset + i] > test_keys[offset + i + 1]) {
                    segment_sorted = false;
                    break;
                }
            }

            std::cout << "  Segment " << seg << " sorted: "
                      << (segment_sorted ? "✓" : "✗") << std::endl;
            all_segments_ok &= segment_sorted;
        }

        // Performance info
        std::cout << "\n[Performance Info]" << std::endl;
        std::cout << "  Temp storage: "
                  << sorter.get_temp_storage_bytes() / (1024.0 * 1024.0)
                  << " MB" << std::endl;
        std::cout << "  Total data: "
                  << (total_items * (sizeof(float) + sizeof(int))) / (1024.0 * 1024.0)
                  << " MB" << std::endl;

        // Final result
        std::cout << "\n==========================================" << std::endl;
        if (keys_match && values_match && all_segments_ok) {
            std::cout << "✓ ALL TESTS PASSED" << std::endl;
            std::cout << "Phase 4 segmented sort working correctly!" << std::endl;
            std::cout << "Ready for TSP/TSPJ integration." << std::endl;
            std::cout << "==========================================" << std::endl;
            return 0;
        } else {
            std::cout << "✗ SOME TESTS FAILED" << std::endl;
            std::cout << "Review verification checks above." << std::endl;
            std::cout << "==========================================" << std::endl;
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "\n✗ TEST FAILED WITH EXCEPTION:" << std::endl;
        std::cerr << "  " << e.what() << std::endl;
        std::cerr << "==========================================" << std::endl;
        return 1;
    }
}
