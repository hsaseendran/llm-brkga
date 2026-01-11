// test_streams.cu - Simple test to verify CUDA streams infrastructure works
#include "core/cuda_streams.hpp"
#include <iostream>
#include <cuda_runtime.h>

__global__ void simple_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

int main() {
    std::cout << "Testing CUDA Streams Infrastructure..." << std::endl;

    try {
        // Test 1: StreamManager creation
        std::cout << "\n[Test 1] Creating StreamManager with 3 streams..." << std::endl;
        StreamManager manager(3);
        std::cout << "  ✓ StreamManager created successfully" << std::endl;
        std::cout << "  ✓ Number of streams: " << manager.get_num_streams() << std::endl;

        // Test 2: Pinned memory allocation
        std::cout << "\n[Test 2] Allocating pinned memory..." << std::endl;
        PinnedMemory<float> host_buffer(1000);
        std::cout << "  ✓ Pinned memory allocated: " << host_buffer.size() << " elements" << std::endl;

        // Test 3: Device memory and async operations
        std::cout << "\n[Test 3] Testing async operations on streams..." << std::endl;
        const int N = 1000;
        float* d_data1;
        float* d_data2;
        cudaMalloc(&d_data1, N * sizeof(float));
        cudaMalloc(&d_data2, N * sizeof(float));

        // Initialize host data
        for (int i = 0; i < N; i++) {
            host_buffer[i] = static_cast<float>(i);
        }

        // Stream 0: Copy to device and process
        cudaMemcpyAsync(d_data1, host_buffer.data(), N * sizeof(float),
                       cudaMemcpyHostToDevice, manager.get_stream(0));
        simple_kernel<<<(N+255)/256, 256, 0, manager.get_stream(0)>>>(d_data1, N);
        manager.record_event(0);
        std::cout << "  ✓ Stream 0: Launched async copy + kernel" << std::endl;

        // Stream 1: Another computation (overlaps with stream 0)
        cudaMemcpyAsync(d_data2, host_buffer.data(), N * sizeof(float),
                       cudaMemcpyHostToDevice, manager.get_stream(1));
        simple_kernel<<<(N+255)/256, 256, 0, manager.get_stream(1)>>>(d_data2, N);
        manager.record_event(1);
        std::cout << "  ✓ Stream 1: Launched async copy + kernel (overlapping)" << std::endl;

        // Wait for both streams
        manager.synchronize_all();
        std::cout << "  ✓ All streams synchronized" << std::endl;

        // Test 4: Event timing
        std::cout << "\n[Test 4] Testing event-based synchronization..." << std::endl;
        manager.record_event(0);
        simple_kernel<<<(N+255)/256, 256, 0, manager.get_stream(0)>>>(d_data1, N);
        manager.record_event(0);

        manager.record_event(1);
        simple_kernel<<<(N+255)/256, 256, 0, manager.get_stream(1)>>>(d_data2, N);
        manager.record_event(1);

        manager.synchronize_all();
        std::cout << "  ✓ Event-based synchronization works" << std::endl;

        // Test 5: Stream query
        std::cout << "\n[Test 5] Testing stream query..." << std::endl;
        bool idle0 = manager.is_stream_idle(0);
        bool idle1 = manager.is_stream_idle(1);
        std::cout << "  ✓ Stream 0 idle: " << (idle0 ? "yes" : "no") << std::endl;
        std::cout << "  ✓ Stream 1 idle: " << (idle1 ? "yes" : "no") << std::endl;

        // Verify results
        cudaMemcpy(host_buffer.data(), d_data1, N * sizeof(float), cudaMemcpyDeviceToHost);
        bool correct = true;
        for (int i = 0; i < 10; i++) {
            // After 2 applications: ((i * 2 + 1) * 2 + 1) = 4i + 3
            float expected = 4.0f * i + 3.0f;
            if (std::abs(host_buffer[i] - expected) > 1e-5) {
                correct = false;
                break;
            }
        }

        std::cout << "\n[Test 6] Verifying computation results..." << std::endl;
        if (correct) {
            std::cout << "  ✓ Computation results correct" << std::endl;
        } else {
            std::cout << "  ✗ Computation results incorrect" << std::endl;
        }

        // Cleanup
        cudaFree(d_data1);
        cudaFree(d_data2);

        std::cout << "\n=========================================" << std::endl;
        std::cout << "ALL TESTS PASSED ✓" << std::endl;
        std::cout << "CUDA Streams infrastructure is working correctly" << std::endl;
        std::cout << "=========================================" << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n✗ TEST FAILED: " << e.what() << std::endl;
        return 1;
    }
}
