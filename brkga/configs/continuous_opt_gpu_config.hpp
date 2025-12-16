#ifndef CONTINUOUS_OPT_GPU_CONFIG_HPP
#define CONTINUOUS_OPT_GPU_CONFIG_HPP

#include "../core/config.hpp"
#include <vector>
#include <memory>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

// Forward declaration of GPU kernels
template<typename T>
__global__ void sphere_fitness_kernel(T* population, T* fitness, T lower_bound, T upper_bound,
                                     int pop_size, int chrom_len);

template<typename T>
__global__ void rastrigin_fitness_kernel(T* population, T* fitness, T lower_bound, T upper_bound,
                                        int pop_size, int chrom_len);

enum class OptimizationFunction {
    SPHERE,
    RASTRIGIN,
    ROSENBROCK
};

template<typename T>
class ContinuousOptGPUConfig : public BRKGAConfig<T> {
private:
    int num_variables;
    T lower_bound;
    T upper_bound;
    OptimizationFunction function_type;
    std::string instance_name;
    bool gpu_available;

public:
    ContinuousOptGPUConfig(int num_vars, T lower, T upper,
                          OptimizationFunction func = OptimizationFunction::SPHERE,
                          const std::string& name = "ContinuousOpt_GPU")
        : BRKGAConfig<T>({num_vars}),
          num_variables(num_vars), lower_bound(lower), upper_bound(upper),
          function_type(func), instance_name(name), gpu_available(false) {

        // CPU fallback fitness function
        this->fitness_function = [this](const Individual<T>& individual) {
            return calculate_continuous_fitness(individual);
        };

        this->decoder = [this](const Individual<T>& individual) {
            return decode_to_solution(individual);
        };

        this->comparator = [](T a, T b) { return a < b; };  // Minimization

        this->threads_per_block = 256;
        this->update_cuda_grid_size();

        // Check GPU availability
        check_gpu_availability();
    }

    // GPU evaluation interface
    bool has_gpu_evaluation() const override { return gpu_available; }

    void evaluate_population_gpu(T* d_population, T* d_fitness,
                                int pop_size, int chrom_len) override {
        if (!gpu_available) {
            return;  // Fallback to CPU
        }

        dim3 block(this->threads_per_block);
        dim3 grid((pop_size + block.x - 1) / block.x);

        switch (function_type) {
            case OptimizationFunction::SPHERE:
                sphere_fitness_kernel<<<grid, block>>>(
                    d_population, d_fitness, lower_bound, upper_bound,
                    pop_size, chrom_len
                );
                break;

            case OptimizationFunction::RASTRIGIN:
                rastrigin_fitness_kernel<<<grid, block>>>(
                    d_population, d_fitness, lower_bound, upper_bound,
                    pop_size, chrom_len
                );
                break;

            default:
                // Fallback to CPU for unsupported functions
                return;
        }

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
            std::cout << "GPU available for Continuous Optimization fitness evaluation" << std::endl;
        } else {
            gpu_available = false;
            std::cout << "No GPU available, using CPU fitness evaluation" << std::endl;
        }
    }

    T calculate_continuous_fitness(const Individual<T>& individual) {
        const auto& chromosome = individual.get_chromosome();

        switch (function_type) {
            case OptimizationFunction::SPHERE:
                return calculate_sphere(chromosome);

            case OptimizationFunction::RASTRIGIN:
                return calculate_rastrigin(chromosome);

            case OptimizationFunction::ROSENBROCK:
                return calculate_rosenbrock(chromosome);

            default:
                return 0;
        }
    }

    T calculate_sphere(const std::vector<T>& chromosome) {
        T sum = 0;
        for (T gene : chromosome) {
            // Scale from [0,1] to [lower_bound, upper_bound]
            T x = gene * (upper_bound - lower_bound) + lower_bound;
            sum += x * x;
        }
        return sum;
    }

    T calculate_rastrigin(const std::vector<T>& chromosome) {
        const T A = 10.0;
        const T PI = 3.14159265358979323846;
        T sum = A * chromosome.size();

        for (T gene : chromosome) {
            // Scale from [0,1] to [lower_bound, upper_bound]
            T x = gene * (upper_bound - lower_bound) + lower_bound;
            sum += x * x - A * std::cos(2 * PI * x);
        }
        return sum;
    }

    T calculate_rosenbrock(const std::vector<T>& chromosome) {
        T sum = 0;
        for (size_t i = 0; i < chromosome.size() - 1; i++) {
            // Scale from [0,1] to [lower_bound, upper_bound]
            T x = chromosome[i] * (upper_bound - lower_bound) + lower_bound;
            T x_next = chromosome[i + 1] * (upper_bound - lower_bound) + lower_bound;

            T term1 = x_next - x * x;
            T term2 = 1 - x;
            sum += 100 * term1 * term1 + term2 * term2;
        }
        return sum;
    }

    std::vector<std::vector<T>> decode_to_solution(const Individual<T>& individual) {
        const auto& chromosome = individual.get_chromosome();

        std::vector<std::vector<T>> result(1);
        result[0].reserve(chromosome.size());

        // Scale genes to actual decision variable range
        for (T gene : chromosome) {
            T value = gene * (upper_bound - lower_bound) + lower_bound;
            result[0].push_back(value);
        }

        return result;
    }

public:
    void print_solution(const Individual<T>& individual) override {
        const auto& chromosome = individual.get_chromosome();

        std::string func_name;
        switch (function_type) {
            case OptimizationFunction::SPHERE: func_name = "Sphere"; break;
            case OptimizationFunction::RASTRIGIN: func_name = "Rastrigin"; break;
            case OptimizationFunction::ROSENBROCK: func_name = "Rosenbrock"; break;
        }

        std::cout << "\n=== " << func_name << " GPU Solution ===" << std::endl;
        std::cout << "Instance: " << instance_name << std::endl;
        std::cout << "Variables: " << chromosome.size() << std::endl;
        std::cout << "Range: [" << lower_bound << ", " << upper_bound << "]" << std::endl;
        std::cout << "Fitness: " << individual.fitness << std::endl;

        std::cout << "\nDecision variables (first 10):" << std::endl;
        for (size_t i = 0; i < std::min(size_t(10), chromosome.size()); i++) {
            T value = chromosome[i] * (upper_bound - lower_bound) + lower_bound;
            std::cout << "  x[" << i << "] = " << value << std::endl;
        }

        if (chromosome.size() > 10) {
            std::cout << "  ... (" << (chromosome.size() - 10) << " more variables)" << std::endl;
        }

        std::cout << "GPU Evaluation: " << (gpu_available ? "Enabled" : "Disabled") << std::endl;
        std::cout << "=====================================" << std::endl;
    }

    static std::unique_ptr<ContinuousOptGPUConfig<T>> create_sphere(int num_vars = 30) {
        return std::make_unique<ContinuousOptGPUConfig<T>>(
            num_vars, -5.12, 5.12,
            OptimizationFunction::SPHERE,
            "Sphere_GPU_" + std::to_string(num_vars)
        );
    }

    static std::unique_ptr<ContinuousOptGPUConfig<T>> create_rastrigin(int num_vars = 30) {
        return std::make_unique<ContinuousOptGPUConfig<T>>(
            num_vars, -5.12, 5.12,
            OptimizationFunction::RASTRIGIN,
            "Rastrigin_GPU_" + std::to_string(num_vars)
        );
    }

    static std::unique_ptr<ContinuousOptGPUConfig<T>> create_rosenbrock(int num_vars = 30) {
        return std::make_unique<ContinuousOptGPUConfig<T>>(
            num_vars, -5.0, 10.0,
            OptimizationFunction::ROSENBROCK,
            "Rosenbrock_GPU_" + std::to_string(num_vars)
        );
    }

    bool is_gpu_available() const { return gpu_available; }
};

// GPU kernel for Sphere function
template<typename T>
__global__ void sphere_fitness_kernel(T* population, T* fitness, T lower_bound, T upper_bound,
                                     int pop_size, int chrom_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    T* chromosome = population + idx * chrom_len;

    T sum = 0;
    for (int i = 0; i < chrom_len; i++) {
        // Scale from [0,1] to [lower_bound, upper_bound]
        T x = chromosome[i] * (upper_bound - lower_bound) + lower_bound;
        sum += x * x;
    }

    fitness[idx] = sum;
}

// GPU kernel for Rastrigin function
template<typename T>
__global__ void rastrigin_fitness_kernel(T* population, T* fitness, T lower_bound, T upper_bound,
                                        int pop_size, int chrom_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    T* chromosome = population + idx * chrom_len;

    const T A = 10.0;
    const T PI = 3.14159265358979323846;

    T sum = A * chrom_len;
    for (int i = 0; i < chrom_len; i++) {
        // Scale from [0,1] to [lower_bound, upper_bound]
        T x = chromosome[i] * (upper_bound - lower_bound) + lower_bound;
        sum += x * x - A * cos(2 * PI * x);
    }

    fitness[idx] = sum;
}

#endif // CONTINUOUS_OPT_GPU_CONFIG_HPP
