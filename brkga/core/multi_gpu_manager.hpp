#ifndef MULTI_GPU_MANAGER_HPP
#define MULTI_GPU_MANAGER_HPP

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <iostream>
#include <algorithm>
#include <numeric>

struct GPUInfo {
    int device_id;
    std::string name;
    size_t total_memory;
    size_t free_memory;
    int compute_capability_major;
    int compute_capability_minor;
    int multiprocessor_count;
    int max_threads_per_block;
    int max_blocks_per_multiprocessor;
    bool is_available;
    double relative_performance; // Performance factor relative to weakest GPU
};

class MultiGPUManager {
private:
    std::vector<GPUInfo> gpu_info;
    std::vector<int> active_devices;
    bool multi_gpu_enabled;
    int primary_device;
    
    // Load balancing
    std::vector<double> performance_weights;
    std::vector<size_t> work_distribution;
    
public:
    MultiGPUManager() : multi_gpu_enabled(false), primary_device(0) {
        detect_gpus();
        calculate_performance_weights();
    }
    
    ~MultiGPUManager() {
        cleanup();
    }
    
    // GPU detection and initialization
    void detect_gpus() {
        int device_count;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        
        if (error != cudaSuccess || device_count == 0) {
            std::cout << "No CUDA devices found or CUDA not available." << std::endl;
            return;
        }
        
        std::cout << "Detected " << device_count << " CUDA device(s):" << std::endl;
        
        for (int i = 0; i < device_count; i++) {
            cudaDeviceProp prop;
            error = cudaGetDeviceProperties(&prop, i);
            
            if (error != cudaSuccess) {
                continue;
            }
            
            GPUInfo info;
            info.device_id = i;
            info.name = prop.name;
            info.total_memory = prop.totalGlobalMem;
            info.compute_capability_major = prop.major;
            info.compute_capability_minor = prop.minor;
            info.multiprocessor_count = prop.multiProcessorCount;
            info.max_threads_per_block = prop.maxThreadsPerBlock;
            info.max_blocks_per_multiprocessor = prop.maxBlocksPerMultiProcessor;
            
            // Get available memory
            cudaSetDevice(i);
            cudaMemGetInfo(&info.free_memory, &info.total_memory);
            
            // Check if device is suitable (compute capability >= 3.5)
            info.is_available = (info.compute_capability_major > 3 || 
                               (info.compute_capability_major == 3 && info.compute_capability_minor >= 5)) &&
                               info.free_memory > 100 * 1024 * 1024; // At least 100MB free
            
            gpu_info.push_back(info);
            
            std::cout << "  GPU " << i << ": " << info.name 
                      << " (CC " << info.compute_capability_major << "." << info.compute_capability_minor << ")"
                      << " - " << (info.free_memory / (1024*1024)) << " MB available"
                      << " - " << (info.is_available ? "Available" : "Not suitable") << std::endl;
        }
        
        // Select available devices
        for (const auto& info : gpu_info) {
            if (info.is_available) {
                active_devices.push_back(info.device_id);
            }
        }
        
        if (active_devices.size() > 1) {
            multi_gpu_enabled = true;
            std::cout << "Multi-GPU support enabled with " << active_devices.size() << " devices." << std::endl;
        } else if (active_devices.size() == 1) {
            primary_device = active_devices[0];
            std::cout << "Single GPU mode enabled." << std::endl;
        }
    }
    
    // Calculate relative performance weights for load balancing
    void calculate_performance_weights() {
        if (active_devices.empty()) return;
        
        performance_weights.clear();
        performance_weights.reserve(active_devices.size());
        
        // Simple performance estimation based on multiprocessor count
        double min_performance = std::numeric_limits<double>::max();
        
        for (int device_id : active_devices) {
            const auto& info = gpu_info[device_id];
            double performance = info.multiprocessor_count * 
                               (info.compute_capability_major * 10 + info.compute_capability_minor);
            gpu_info[device_id].relative_performance = performance;
            min_performance = std::min(min_performance, performance);
        }
        
        // Normalize weights (weakest GPU = 1.0)
        for (int device_id : active_devices) {
            double normalized_weight = gpu_info[device_id].relative_performance / min_performance;
            performance_weights.push_back(normalized_weight);
        }
        
        std::cout << "Performance weights calculated:" << std::endl;
        for (size_t i = 0; i < active_devices.size(); i++) {
            std::cout << "  GPU " << active_devices[i] << ": " 
                      << std::fixed << std::setprecision(2) << performance_weights[i] << "x" << std::endl;
        }
    }
    
    // Distribute work across GPUs based on performance
    std::vector<size_t> distribute_work(size_t total_work) {
        work_distribution.clear();
        
        if (!multi_gpu_enabled || active_devices.size() <= 1) {
            work_distribution.push_back(total_work);
            return work_distribution;
        }
        
        // Calculate work distribution based on performance weights
        double total_weight = std::accumulate(performance_weights.begin(), performance_weights.end(), 0.0);
        size_t assigned_work = 0;
        
        for (size_t i = 0; i < performance_weights.size() - 1; i++) {
            size_t gpu_work = static_cast<size_t>((performance_weights[i] / total_weight) * total_work);
            work_distribution.push_back(gpu_work);
            assigned_work += gpu_work;
        }
        
        // Assign remaining work to last GPU
        work_distribution.push_back(total_work - assigned_work);
        
        return work_distribution;
    }
    
    // Set device for current thread
    void set_device(size_t gpu_index) {
        if (gpu_index < active_devices.size()) {
            cudaSetDevice(active_devices[gpu_index]);
        }
    }
    
    // Synchronize all active devices
    void synchronize_all() {
        for (int device_id : active_devices) {
            cudaSetDevice(device_id);
            cudaDeviceSynchronize();
        }
    }
    
    // Enable peer-to-peer access between GPUs
    void enable_peer_access() {
        if (!multi_gpu_enabled) return;
        
        for (size_t i = 0; i < active_devices.size(); i++) {
            cudaSetDevice(active_devices[i]);
            
            for (size_t j = 0; j < active_devices.size(); j++) {
                if (i != j) {
                    int can_access_peer;
                    cudaDeviceCanAccessPeer(&can_access_peer, active_devices[i], active_devices[j]);
                    
                    if (can_access_peer) {
                        cudaError_t error = cudaDeviceEnablePeerAccess(active_devices[j], 0);
                        if (error == cudaSuccess) {
                            std::cout << "Enabled P2P access: GPU " << active_devices[i] 
                                      << " -> GPU " << active_devices[j] << std::endl;
                        }
                    }
                }
            }
        }
    }
    
    // Disable peer-to-peer access
    void disable_peer_access() {
        for (size_t i = 0; i < active_devices.size(); i++) {
            cudaSetDevice(active_devices[i]);
            
            for (size_t j = 0; j < active_devices.size(); j++) {
                if (i != j) {
                    cudaDeviceDisablePeerAccess(active_devices[j]);
                }
            }
        }
    }
    
    // Memory management helpers
    size_t get_available_memory(size_t gpu_index) {
        if (gpu_index >= active_devices.size()) return 0;
        
        cudaSetDevice(active_devices[gpu_index]);
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        return free_mem;
    }
    
    size_t get_total_available_memory() {
        size_t total = 0;
        for (size_t i = 0; i < active_devices.size(); i++) {
            total += get_available_memory(i);
        }
        return total;
    }
    
    // Getters
    bool is_multi_gpu_enabled() const { return multi_gpu_enabled; }
    size_t get_gpu_count() const { return active_devices.size(); }
    int get_device_id(size_t index) const { 
        return (index < active_devices.size()) ? active_devices[index] : -1; 
    }
    const std::vector<int>& get_active_devices() const { return active_devices; }
    const std::vector<double>& get_performance_weights() const { return performance_weights; }
    const std::vector<GPUInfo>& get_gpu_info() const { return gpu_info; }
    
    // Force single/multi GPU mode
    void force_single_gpu(int device_id = 0) {
        if (device_id >= 0 && device_id < static_cast<int>(gpu_info.size()) && 
            gpu_info[device_id].is_available) {
            active_devices = {device_id};
            primary_device = device_id;
            multi_gpu_enabled = false;
            performance_weights = {1.0};
            std::cout << "Forced single GPU mode on device " << device_id << std::endl;
        }
    }
    
    void enable_multi_gpu() {
        if (gpu_info.size() > 1) {
            active_devices.clear();
            for (const auto& info : gpu_info) {
                if (info.is_available) {
                    active_devices.push_back(info.device_id);
                }
            }
            multi_gpu_enabled = (active_devices.size() > 1);
            calculate_performance_weights();
        }
    }
    
    // Print detailed GPU information
    void print_gpu_info() const {
        std::cout << "\n=== GPU Configuration ===" << std::endl;
        std::cout << "Total GPUs detected: " << gpu_info.size() << std::endl;
        std::cout << "Active GPUs: " << active_devices.size() << std::endl;
        std::cout << "Multi-GPU enabled: " << (multi_gpu_enabled ? "Yes" : "No") << std::endl;
        
        for (const auto& info : gpu_info) {
            std::cout << "\nGPU " << info.device_id << ": " << info.name << std::endl;
            std::cout << "  Compute Capability: " << info.compute_capability_major 
                      << "." << info.compute_capability_minor << std::endl;
            std::cout << "  Total Memory: " << (info.total_memory / (1024*1024)) << " MB" << std::endl;
            std::cout << "  Free Memory: " << (info.free_memory / (1024*1024)) << " MB" << std::endl;
            std::cout << "  Multiprocessors: " << info.multiprocessor_count << std::endl;
            std::cout << "  Max Threads/Block: " << info.max_threads_per_block << std::endl;
            std::cout << "  Status: " << (info.is_available ? "Available" : "Not suitable") << std::endl;
            
            if (std::find(active_devices.begin(), active_devices.end(), info.device_id) != active_devices.end()) {
                auto it = std::find(active_devices.begin(), active_devices.end(), info.device_id);
                size_t index = std::distance(active_devices.begin(), it);
                std::cout << "  Performance Weight: " << std::fixed << std::setprecision(2) 
                          << performance_weights[index] << "x" << std::endl;
            }
        }
        std::cout << "=========================" << std::endl;
    }
    
    // Cleanup
    void cleanup() {
        if (multi_gpu_enabled) {
            disable_peer_access();
        }
    }
    
    // Benchmark GPUs (optional - for fine-tuning performance weights)
    void benchmark_gpus(size_t work_size = 1000000) {
        if (active_devices.empty()) return;
        
        std::cout << "\nBenchmarking GPUs..." << std::endl;
        
        std::vector<double> benchmark_times;
        
        for (size_t i = 0; i < active_devices.size(); i++) {
            cudaSetDevice(active_devices[i]);
            
            // Allocate test data
            float* d_data;
            cudaMalloc(&d_data, work_size * sizeof(float));
            
            // Create events for timing
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            
            // Warm up
            for (int w = 0; w < 3; w++) {
                cudaMemset(d_data, 0, work_size * sizeof(float));
                cudaDeviceSynchronize();
            }
            
            // Benchmark kernel execution time
            cudaEventRecord(start);
            for (int iter = 0; iter < 100; iter++) {
                cudaMemset(d_data, iter % 256, work_size * sizeof(float));
            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float elapsed_ms;
            cudaEventElapsedTime(&elapsed_ms, start, stop);
            benchmark_times.push_back(elapsed_ms);
            
            // Cleanup
            cudaFree(d_data);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            
            std::cout << "  GPU " << active_devices[i] << " benchmark: " 
                      << std::fixed << std::setprecision(2) << elapsed_ms << " ms" << std::endl;
        }
        
        // Update performance weights based on benchmark results
        double fastest_time = *std::min_element(benchmark_times.begin(), benchmark_times.end());
        
        for (size_t i = 0; i < benchmark_times.size(); i++) {
            performance_weights[i] = fastest_time / benchmark_times[i];
            std::cout << "  GPU " << active_devices[i] << " updated weight: " 
                      << std::fixed << std::setprecision(2) << performance_weights[i] << "x" << std::endl;
        }
    }
};

// Global multi-GPU manager instance
extern MultiGPUManager g_multi_gpu_manager;

#endif // MULTI_GPU_MANAGER_HPP