#ifndef KNAPSACK_GPU_CONFIG_HPP
#define KNAPSACK_GPU_CONFIG_HPP

#include "../core/config.hpp"
#include <vector>
#include <memory>
#include <iostream>
#include <cuda_runtime.h>

// Forward declaration of GPU kernel
template<typename T>
__global__ void knapsack_fitness_kernel(T* population, T* fitness, T* weights, T* values,
                                       T capacity, int pop_size, int chrom_len);

template<typename T>
class KnapsackGPUConfig : public BRKGAConfig<T> {
private:
    std::vector<T> weights;
    std::vector<T> values;
    T capacity;
    std::string instance_name;

    // GPU-specific members
    T* d_weights;
    T* d_values;
    bool gpu_memory_allocated;
    bool gpu_available;

public:
    KnapsackGPUConfig(const std::vector<T>& w, const std::vector<T>& v, T cap,
                      const std::string& name = "Knapsack_GPU")
        : BRKGAConfig<T>({static_cast<int>(w.size())}),
          weights(w), values(v), capacity(cap), instance_name(name),
          d_weights(nullptr), d_values(nullptr),
          gpu_memory_allocated(false), gpu_available(false) {

        if (weights.size() != values.size()) {
            throw std::invalid_argument("Weights and values must have same size");
        }

        // CPU fallback fitness function
        this->fitness_function = [this](const Individual<T>& individual) {
            return calculate_knapsack_fitness(individual);
        };

        this->decoder = [this](const Individual<T>& individual) {
            return decode_to_solution(individual);
        };

        this->comparator = [](T a, T b) { return a > b; };  // Maximization

        this->threads_per_block = 256;
        this->update_cuda_grid_size();

        // Check GPU availability
        check_gpu_availability();
    }

    ~KnapsackGPUConfig() {
        cleanup_gpu_memory();
    }

    // GPU evaluation interface
    bool has_gpu_evaluation() const override { return gpu_available; }

    void evaluate_population_gpu(T* d_population, T* d_fitness,
                                int pop_size, int chrom_len) override {
        if (!gpu_available) {
            return;  // Fallback to CPU
        }

        if (!gpu_memory_allocated) {
            allocate_gpu_memory();
        }

        dim3 block(this->threads_per_block);
        dim3 grid((pop_size + block.x - 1) / block.x);

        knapsack_fitness_kernel<<<grid, block>>>(
            d_population, d_fitness, d_weights, d_values,
            capacity, pop_size, chrom_len
        );

        cudaError_t error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            std::cout << "GPU fitness kernel failed: " << cudaGetErrorString(error) << std::endl;
        }
    }

private:
    void check_gpu_availability() {
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);

        if (error == cudaSuccess && device_count > 0) {
            gpu_available = true;
            std::cout << "GPU available for Knapsack fitness evaluation" << std::endl;
        } else {
            gpu_available = false;
            std::cout << "No GPU available, using CPU fitness evaluation" << std::endl;
        }
    }

    void allocate_gpu_memory() {
        if (!gpu_available || gpu_memory_allocated) return;

        int num_items = weights.size();

        // Allocate weight array
        cudaError_t error = cudaMalloc(&d_weights, num_items * sizeof(T));
        if (error != cudaSuccess) {
            std::cout << "GPU memory allocation failed for weights: " << cudaGetErrorString(error) << std::endl;
            gpu_available = false;
            return;
        }

        // Allocate value array
        error = cudaMalloc(&d_values, num_items * sizeof(T));
        if (error != cudaSuccess) {
            std::cout << "GPU memory allocation failed for values: " << cudaGetErrorString(error) << std::endl;
            cudaFree(d_weights);
            gpu_available = false;
            return;
        }

        // Copy data to GPU
        error = cudaMemcpy(d_weights, weights.data(), num_items * sizeof(T), cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            std::cout << "GPU memory copy failed for weights: " << cudaGetErrorString(error) << std::endl;
            cudaFree(d_weights);
            cudaFree(d_values);
            gpu_available = false;
            return;
        }

        error = cudaMemcpy(d_values, values.data(), num_items * sizeof(T), cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            std::cout << "GPU memory copy failed for values: " << cudaGetErrorString(error) << std::endl;
            cudaFree(d_weights);
            cudaFree(d_values);
            gpu_available = false;
            return;
        }

        gpu_memory_allocated = true;
        std::cout << "GPU data allocated for " << num_items << " items" << std::endl;
    }

    void cleanup_gpu_memory() {
        if (gpu_memory_allocated) {
            if (d_weights) cudaFree(d_weights);
            if (d_values) cudaFree(d_values);
            gpu_memory_allocated = false;
            d_weights = nullptr;
            d_values = nullptr;
        }
    }

    T calculate_knapsack_fitness(const Individual<T>& individual) {
        const auto& chromosome = individual.get_chromosome();
        T total_weight = 0;
        T total_value = 0;

        for (size_t i = 0; i < chromosome.size(); i++) {
            if (chromosome[i] > 0.5) {  // Item is selected
                total_weight += weights[i];
                total_value += values[i];
            }
        }

        // Penalty for exceeding capacity
        if (total_weight > capacity) {
            T penalty = (total_weight - capacity) * 1000;
            return total_value - penalty;
        }

        return total_value;
    }

    std::vector<std::vector<T>> decode_to_solution(const Individual<T>& individual) {
        const auto& chromosome = individual.get_chromosome();

        std::vector<std::vector<T>> result(1);
        result[0].reserve(chromosome.size());

        for (T gene : chromosome) {
            result[0].push_back(gene > 0.5 ? T(1) : T(0));
        }

        return result;
    }

public:
    void print_solution(const Individual<T>& individual) override {
        const auto& chromosome = individual.get_chromosome();
        T total_weight = 0;
        T total_value = 0;
        int selected_items = 0;

        std::cout << "\n=== Knapsack GPU Solution ===" << std::endl;
        std::cout << "Instance: " << instance_name << std::endl;
        std::cout << "Selected items: ";
        for (size_t i = 0; i < chromosome.size(); i++) {
            if (chromosome[i] > 0.5) {
                std::cout << i << " ";
                total_weight += weights[i];
                total_value += values[i];
                selected_items++;
            }
        }
        std::cout << std::endl;
        std::cout << "Total weight: " << total_weight << "/" << capacity << std::endl;
        std::cout << "Total value: " << total_value << std::endl;
        std::cout << "Items selected: " << selected_items << "/" << weights.size() << std::endl;
        std::cout << "GPU Evaluation: " << (gpu_available ? "Enabled" : "Disabled") << std::endl;
        std::cout << "=============================" << std::endl;
    }

    static std::unique_ptr<KnapsackGPUConfig<T>> create_test_instance(int num_items = 100) {
        std::vector<T> w, v;
        w.reserve(num_items);
        v.reserve(num_items);

        T total_weight = 0;
        for (int i = 0; i < num_items; i++) {
            T weight = static_cast<T>(10 + (i % 50));
            T value = static_cast<T>(20 + (i % 80));
            w.push_back(weight);
            v.push_back(value);
            total_weight += weight;
        }

        T cap = total_weight * 0.5;  // 50% capacity

        return std::make_unique<KnapsackGPUConfig<T>>(w, v, cap, "Test_GPU_" + std::to_string(num_items));
    }

    bool is_gpu_available() const { return gpu_available; }
};

// GPU kernel implementation
template<typename T>
__global__ void knapsack_fitness_kernel(T* population, T* fitness, T* weights, T* values,
                                       T capacity, int pop_size, int chrom_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    T* chromosome = population + idx * chrom_len;

    T total_weight = 0;
    T total_value = 0;

    // Calculate total weight and value
    for (int i = 0; i < chrom_len; i++) {
        if (chromosome[i] > 0.5) {  // Item is selected
            total_weight += weights[i];
            total_value += values[i];
        }
    }

    // Apply penalty if over capacity
    if (total_weight > capacity) {
        T penalty = (total_weight - capacity) * 1000;
        fitness[idx] = total_value - penalty;
    } else {
        fitness[idx] = total_value;
    }
}

#endif // KNAPSACK_GPU_CONFIG_HPP
