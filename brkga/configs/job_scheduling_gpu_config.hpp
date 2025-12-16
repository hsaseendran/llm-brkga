#ifndef JOB_SCHEDULING_GPU_CONFIG_HPP
#define JOB_SCHEDULING_GPU_CONFIG_HPP

#include "../core/config.hpp"
#include <vector>
#include <memory>
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>

// Forward declaration of GPU kernel
template<typename T>
__global__ void job_scheduling_fitness_kernel(T* population, T* fitness, T* job_times,
                                             int num_machines, int pop_size, int chrom_len);

template<typename T>
class JobSchedulingGPUConfig : public BRKGAConfig<T> {
private:
    std::vector<T> job_times;
    int num_machines;
    std::string instance_name;

    // GPU-specific members
    T* d_job_times;
    bool gpu_memory_allocated;
    bool gpu_available;

public:
    JobSchedulingGPUConfig(const std::vector<T>& jobs, int machines,
                          const std::string& name = "JobScheduling_GPU")
        : BRKGAConfig<T>({static_cast<int>(jobs.size())}),
          job_times(jobs), num_machines(machines), instance_name(name),
          d_job_times(nullptr),
          gpu_memory_allocated(false), gpu_available(false) {

        if (num_machines <= 0) {
            throw std::invalid_argument("Number of machines must be positive");
        }

        // CPU fallback fitness function
        this->fitness_function = [this](const Individual<T>& individual) {
            return calculate_scheduling_fitness(individual);
        };

        this->decoder = [this](const Individual<T>& individual) {
            return decode_to_solution(individual);
        };

        this->comparator = [](T a, T b) { return a < b; };  // Minimization (makespan)

        this->threads_per_block = 256;
        this->update_cuda_grid_size();

        // Check GPU availability
        check_gpu_availability();
    }

    ~JobSchedulingGPUConfig() {
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

        job_scheduling_fitness_kernel<<<grid, block>>>(
            d_population, d_fitness, d_job_times,
            num_machines, pop_size, chrom_len
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
            std::cout << "GPU available for Job Scheduling fitness evaluation" << std::endl;
        } else {
            gpu_available = false;
            std::cout << "No GPU available, using CPU fitness evaluation" << std::endl;
        }
    }

    void allocate_gpu_memory() {
        if (!gpu_available || gpu_memory_allocated) return;

        int num_jobs = job_times.size();

        // Allocate job times array
        cudaError_t error = cudaMalloc(&d_job_times, num_jobs * sizeof(T));
        if (error != cudaSuccess) {
            std::cout << "GPU memory allocation failed: " << cudaGetErrorString(error) << std::endl;
            gpu_available = false;
            return;
        }

        // Copy data to GPU
        error = cudaMemcpy(d_job_times, job_times.data(), num_jobs * sizeof(T), cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            std::cout << "GPU memory copy failed: " << cudaGetErrorString(error) << std::endl;
            cudaFree(d_job_times);
            gpu_available = false;
            return;
        }

        gpu_memory_allocated = true;
        std::cout << "GPU data allocated for " << num_jobs << " jobs on " << num_machines << " machines" << std::endl;
    }

    void cleanup_gpu_memory() {
        if (gpu_memory_allocated && d_job_times) {
            cudaFree(d_job_times);
            gpu_memory_allocated = false;
            d_job_times = nullptr;
        }
    }

    T calculate_scheduling_fitness(const Individual<T>& individual) {
        const auto& chromosome = individual.get_chromosome();

        // Initialize machine loads
        std::vector<T> machine_load(num_machines, 0);

        // Assign jobs to machines based on chromosome values
        for (size_t i = 0; i < chromosome.size(); i++) {
            // Map gene value [0,1] to machine index
            int machine = static_cast<int>(chromosome[i] * num_machines);
            if (machine >= num_machines) machine = num_machines - 1;

            machine_load[machine] += job_times[i];
        }

        // Makespan is the maximum machine load
        T makespan = *std::max_element(machine_load.begin(), machine_load.end());

        return makespan;
    }

    std::vector<std::vector<T>> decode_to_solution(const Individual<T>& individual) {
        const auto& chromosome = individual.get_chromosome();

        std::vector<std::vector<T>> result(1);
        result[0].reserve(chromosome.size());

        // Map each gene to a machine assignment
        for (T gene : chromosome) {
            int machine = static_cast<int>(gene * num_machines);
            if (machine >= num_machines) machine = num_machines - 1;
            result[0].push_back(static_cast<T>(machine));
        }

        return result;
    }

public:
    void print_solution(const Individual<T>& individual) override {
        const auto& chromosome = individual.get_chromosome();

        std::cout << "\n=== Job Scheduling GPU Solution ===" << std::endl;
        std::cout << "Instance: " << instance_name << std::endl;
        std::cout << "Jobs: " << chromosome.size() << ", Machines: " << num_machines << std::endl;

        // Calculate machine assignments and loads
        std::vector<T> machine_load(num_machines, 0);
        std::vector<std::vector<int>> machine_jobs(num_machines);

        for (size_t i = 0; i < chromosome.size(); i++) {
            int machine = static_cast<int>(chromosome[i] * num_machines);
            if (machine >= num_machines) machine = num_machines - 1;

            machine_load[machine] += job_times[i];
            machine_jobs[machine].push_back(i);
        }

        T makespan = *std::max_element(machine_load.begin(), machine_load.end());

        std::cout << "\nMachine assignments:" << std::endl;
        for (int m = 0; m < num_machines; m++) {
            std::cout << "Machine " << m << " (load: " << machine_load[m] << "): ";
            for (size_t j = 0; j < machine_jobs[m].size(); j++) {
                std::cout << "J" << machine_jobs[m][j];
                if (j < machine_jobs[m].size() - 1) std::cout << ", ";
            }
            if (machine_load[m] == makespan) std::cout << " <- CRITICAL";
            std::cout << std::endl;
        }

        std::cout << "\nMakespan: " << makespan << std::endl;
        std::cout << "GPU Evaluation: " << (gpu_available ? "Enabled" : "Disabled") << std::endl;
        std::cout << "====================================" << std::endl;
    }

    static std::unique_ptr<JobSchedulingGPUConfig<T>> create_test_instance(int num_jobs = 50, int num_machines = 5) {
        std::vector<T> jobs;
        jobs.reserve(num_jobs);

        for (int i = 0; i < num_jobs; i++) {
            T job_time = static_cast<T>(10 + (i % 40));
            jobs.push_back(job_time);
        }

        return std::make_unique<JobSchedulingGPUConfig<T>>(
            jobs, num_machines,
            "Test_GPU_J" + std::to_string(num_jobs) + "_M" + std::to_string(num_machines)
        );
    }

    bool is_gpu_available() const { return gpu_available; }
    int get_num_machines() const { return num_machines; }
};

// GPU kernel implementation
template<typename T>
__global__ void job_scheduling_fitness_kernel(T* population, T* fitness, T* job_times,
                                             int num_machines, int pop_size, int chrom_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    T* chromosome = population + idx * chrom_len;

    // Local array for machine loads (limit to 32 machines for GPU)
    T machine_load[32];
    int actual_machines = min(num_machines, 32);

    // Initialize machine loads
    for (int m = 0; m < actual_machines; m++) {
        machine_load[m] = 0;
    }

    // Assign jobs to machines
    for (int i = 0; i < chrom_len; i++) {
        // Map gene value [0,1] to machine index
        int machine = static_cast<int>(chromosome[i] * actual_machines);
        if (machine >= actual_machines) machine = actual_machines - 1;

        machine_load[machine] += job_times[i];
    }

    // Find makespan (maximum load)
    T makespan = machine_load[0];
    for (int m = 1; m < actual_machines; m++) {
        if (machine_load[m] > makespan) {
            makespan = machine_load[m];
        }
    }

    fitness[idx] = makespan;
}

#endif // JOB_SCHEDULING_GPU_CONFIG_HPP
