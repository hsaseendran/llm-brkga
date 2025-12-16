#ifndef TSPJ_CONFIG_HPP
#define TSPJ_CONFIG_HPP

#include "../core/config.hpp"
#include "../core/local_search.hpp"
#include "../core/cuda_kernels.cuh"
#include "../utils/file_utils.hpp"
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <chrono>
#include <random>
#include <map>
#include <mutex>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Forward declaration of GPU kernel
template<typename T>
__global__ void tspj_fitness_kernel(T* population, T* fitness, T* travel_matrix, T* job_matrix,
                                    int pop_size, int chrom_len, int num_cities, int matrix_dim);

template<typename T>
class TSPJConfig : public BRKGAConfig<T> {
private:
    std::vector<std::vector<T>> travel_times;
    std::vector<std::vector<T>> job_durations;
    int num_cities;
    int num_jobs;
    std::string instance_name;

    // GPU-specific members (per-GPU storage)
    std::map<int, T*> d_travel_matrices;  // device_id -> matrix
    std::map<int, T*> d_job_matrices;     // device_id -> matrix
    bool gpu_available;
    mutable std::mutex gpu_mutex;  // Thread safety for multi-GPU allocation
    
public:
    TSPJConfig(const std::vector<std::vector<T>>& travel,
               const std::vector<std::vector<T>>& jobs,
               const std::string& name = "TSPJ")
        : BRKGAConfig<T>({static_cast<int>(travel.size() - 1), static_cast<int>(travel.size() - 1)}),
          travel_times(travel), job_durations(jobs),
          num_cities(travel.size() - 1), num_jobs(travel.size() - 1), instance_name(name),
          gpu_available(false) {

        this->fitness_function = [this](const Individual<T>& individual) {
            return calculate_tspj_fitness(individual);
        };

        this->decoder = [this](const Individual<T>& individual) {
            return decode_to_solution(individual);
        };

        this->comparator = [](T a, T b) { return a < b; };

        this->threads_per_block = 256;
        this->update_cuda_grid_size();

        // Check GPU availability
        check_gpu_availability();
    }

    ~TSPJConfig() {
        cleanup_all_gpu_memory();
    }

    // GPU evaluation interface
    bool has_gpu_evaluation() const override { return gpu_available; }

    // Local search interface - TSPJ uses random 2-opt on the city tour
    bool has_local_search() const override { return gpu_available; }

    int apply_local_search_gpu(
        T* d_population,
        T* d_backup,
        T* d_fitness,
        T* d_fitness_backup,
        void* d_rng_states,
        int pop_size,
        int chrom_len,
        int num_to_improve,
        int num_moves
    ) override {
        if (!gpu_available || num_to_improve <= 0) return 0;

        int device_id;
        cudaGetDevice(&device_id);

        // TSPJ has 2 components: cities + jobs, each of size num_cities
        int num_cities_local = chrom_len / 2;

        int threads = 256;
        int blocks = (num_to_improve + threads - 1) / threads;

        // Step 1: Backup fitness
        cudaMemcpy(d_fitness_backup, d_fitness,
                   num_to_improve * sizeof(T), cudaMemcpyDeviceToDevice);

        // Step 2: Apply random 2-opt to city portion
        random_2opt_kernel<<<blocks, threads>>>(
            d_population,
            d_backup,
            d_fitness,
            d_fitness_backup,
            static_cast<curandState*>(d_rng_states),
            pop_size,
            num_cities_local,
            chrom_len,
            num_to_improve,
            num_moves
        );
        cudaDeviceSynchronize();

        // Step 3: Re-evaluate fitness
        evaluate_population_gpu(d_population, d_fitness, num_to_improve, chrom_len);
        cudaDeviceSynchronize();

        // Step 4: Restore if worse
        restore_if_worse_kernel<<<blocks, threads>>>(
            d_population,
            d_backup,
            d_fitness_backup,
            d_fitness,
            num_cities_local,
            chrom_len,
            num_to_improve
        );
        cudaDeviceSynchronize();

        return 0;  // Improvement tracking done in solver
    }

    void evaluate_population_gpu(T* d_population, T* d_fitness,
                                 int pop_size, int chrom_len) override {
        if (!gpu_available) return;

        // Get current device (set by caller)
        int device_id;
        cudaGetDevice(&device_id);

        // Ensure matrices are allocated on this device
        ensure_gpu_memory(device_id);

        // CRITICAL: Re-set device after ensure_gpu_memory() because
        // another thread might have changed it while we waited on mutex
        cudaSetDevice(device_id);

        // Get device-specific pointers
        T* d_travel = nullptr;
        T* d_job = nullptr;
        {
            std::lock_guard<std::mutex> lock(gpu_mutex);
            d_travel = d_travel_matrices[device_id];
            d_job = d_job_matrices[device_id];
        }

        if (!d_travel || !d_job) return;

        dim3 block(this->threads_per_block);
        dim3 grid((pop_size + block.x - 1) / block.x);

        int matrix_dim = num_cities + 1;  // Include depot (row/col 0)

        tspj_fitness_kernel<<<grid, block>>>(
            d_population, d_fitness, d_travel, d_job,
            pop_size, chrom_len, num_cities, matrix_dim
        );

        cudaError_t error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            std::cerr << "TSPJ GPU kernel failed on device " << device_id
                      << ": " << cudaGetErrorString(error) << std::endl;
        }
    }

private:
    void check_gpu_availability() {
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);

        if (error == cudaSuccess && device_count > 0) {
            gpu_available = true;
        } else {
            gpu_available = false;
        }
    }

    void ensure_gpu_memory(int device_id) {
        std::lock_guard<std::mutex> lock(gpu_mutex);

        // Check if already allocated on this device
        if (d_travel_matrices.find(device_id) != d_travel_matrices.end()) {
            return;
        }

        // Allocate on specified device
        cudaSetDevice(device_id);

        int matrix_dim = num_cities + 1;
        int matrix_size = matrix_dim * matrix_dim;

        T* d_travel = nullptr;
        T* d_job = nullptr;

        cudaError_t err1 = cudaMalloc(&d_travel, matrix_size * sizeof(T));
        cudaError_t err2 = cudaMalloc(&d_job, matrix_size * sizeof(T));

        if (err1 != cudaSuccess || err2 != cudaSuccess) {
            if (d_travel) cudaFree(d_travel);
            if (d_job) cudaFree(d_job);
            std::cerr << "Failed to allocate GPU memory on device " << device_id
                      << ": err1=" << cudaGetErrorString(err1)
                      << ", err2=" << cudaGetErrorString(err2)
                      << ", size=" << (matrix_size * sizeof(T) / 1024 / 1024) << "MB" << std::endl;
            return;
        }

        // Flatten and copy matrices to GPU
        std::vector<T> flat_travel(matrix_size);
        std::vector<T> flat_jobs(matrix_size);

        for (int i = 0; i < matrix_dim; i++) {
            for (int j = 0; j < matrix_dim; j++) {
                flat_travel[i * matrix_dim + j] = travel_times[i][j];
                flat_jobs[i * matrix_dim + j] = job_durations[i][j];
            }
        }

        cudaMemcpy(d_travel, flat_travel.data(), matrix_size * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(d_job, flat_jobs.data(), matrix_size * sizeof(T), cudaMemcpyHostToDevice);

        d_travel_matrices[device_id] = d_travel;
        d_job_matrices[device_id] = d_job;
    }

    void cleanup_all_gpu_memory() {
        std::lock_guard<std::mutex> lock(gpu_mutex);

        for (auto& pair : d_travel_matrices) {
            if (pair.second) {
                cudaSetDevice(pair.first);
                cudaFree(pair.second);
            }
        }
        for (auto& pair : d_job_matrices) {
            if (pair.second) {
                cudaSetDevice(pair.first);
                cudaFree(pair.second);
            }
        }
        d_travel_matrices.clear();
        d_job_matrices.clear();
    }

    // Local search methods - uncomment when LocalSearchManager is available in BRKGAConfig
    /*
    void add_tspj_local_searches() {
        auto city_swap = std::make_unique<CitySwapTSPJ>(travel_times, job_durations, num_cities, num_jobs);
        this->add_local_search(std::move(city_swap));

        auto job_reassign = std::make_unique<JobReassignTSPJ>(travel_times, job_durations, num_cities, num_jobs);
        this->add_local_search(std::move(job_reassign));

        LocalSearchConfig<T> ls_config;
        ls_config.strategy = LocalSearchStrategy::ELITE_ONLY;
        ls_config.frequency = 2;
        ls_config.probability = 0.8;
        ls_config.apply_to_best = true;
        this->set_local_search_config(ls_config);
    }
    */

private:
    class CitySwapTSPJ : public LocalSearch<T> {
    private:
        const std::vector<std::vector<T>>& travel_times;
        const std::vector<std::vector<T>>& job_durations;
        int num_cities, num_jobs;
        
    public:
        CitySwapTSPJ(const std::vector<std::vector<T>>& travel, const std::vector<std::vector<T>>& jobs,
                     int cities, int job_types) 
            : LocalSearch<T>("TSPJ-CitySwap"), travel_times(travel), job_durations(jobs), 
              num_cities(cities), num_jobs(job_types) {}
        
        Individual<T> improve(const Individual<T>& individual) override {
            Individual<T> best = individual;
            auto solution = decode_solution(individual);
            
            bool improved = true;
            int iterations = 0;
            
            while (improved && iterations < this->max_iterations) {
                improved = false;
                
                for (int i = 0; i < num_cities - 1 && !improved; i++) {
                    for (int j = i + 1; j < num_cities && !improved; j++) {
                        auto new_solution = solution;
                        std::swap(new_solution.first[i], new_solution.first[j]);
                        
                        T new_fitness = calculate_makespan(new_solution);
                        if (new_fitness < best.fitness) {
                            solution = new_solution;
                            best.set_fitness(new_fitness);
                            improved = true;
                        }
                    }
                }
                iterations++;
            }
            
            if (best.fitness < individual.fitness) {
                encode_solution(best, solution);
            }
            
            return best;
        }
        
        bool should_apply(int generation, const Individual<T>& individual, 
                         const std::vector<Individual<T>>& population) override {
            return true;
        }
        
        void configure(const std::map<std::string, std::string>& params) override {
            this->parse_basic_config(params);
        }
        
        LocalSearch<T>* clone() const override {
            return new CitySwapTSPJ(travel_times, job_durations, num_cities, num_jobs);
        }
        
    protected:
        bool is_better(T fitness1, T fitness2) const override {
            return fitness1 < fitness2;
        }
        
    private:
        std::pair<std::vector<int>, std::vector<int>> decode_solution(const Individual<T>& individual) {
            const auto& city_chromosome = individual.get_component(0);
            const auto& job_chromosome = individual.get_component(1);
            
            std::vector<std::pair<T, int>> keyed_cities;
            for (int i = 0; i < num_cities; i++) {
                keyed_cities.emplace_back(city_chromosome[i], i + 1);
            }
            std::sort(keyed_cities.begin(), keyed_cities.end());
            
            std::vector<int> city_tour;
            for (const auto& pair : keyed_cities) {
                city_tour.push_back(pair.second);
            }
            
            std::vector<std::pair<T, int>> keyed_jobs;
            for (int i = 0; i < num_jobs; i++) {
                keyed_jobs.emplace_back(job_chromosome[i], i + 1);
            }
            std::sort(keyed_jobs.begin(), keyed_jobs.end());
            
            std::vector<int> job_assignment;
            for (int i = 0; i < num_jobs; i++) {
                job_assignment.push_back(keyed_jobs[i].second);
            }
            
            return {city_tour, job_assignment};
        }
        
        void encode_solution(Individual<T>& individual, const std::pair<std::vector<int>, std::vector<int>>& solution) {
            auto& city_chromosome = individual.get_component(0);
            auto& job_chromosome = individual.get_component(1);
            
            for (int i = 0; i < num_cities; i++) {
                city_chromosome[solution.first[i] - 1] = static_cast<T>(i) / static_cast<T>(num_cities);
            }
            
            for (int i = 0; i < num_jobs; i++) {
                job_chromosome[solution.second[i] - 1] = static_cast<T>(i) / static_cast<T>(num_jobs);
            }
            
            individual.reset_evaluation();
        }
        
        T calculate_makespan(const std::pair<std::vector<int>, std::vector<int>>& solution) {
            const auto& city_tour = solution.first;
            const auto& job_assignment = solution.second;
            
            T travel_time = 0;
            int current_city = 0;
            for (int city : city_tour) {
                travel_time += travel_times[current_city][city];
                current_city = city;
            }
            travel_time += travel_times[current_city][0];
            
            T max_job_completion = 0;
            T current_time = 0;
            current_city = 0;
            
            for (size_t i = 0; i < city_tour.size(); i++) {
                int city = city_tour[i];
                int job_type = job_assignment[i];
                
                current_time += travel_times[current_city][city];
                T job_completion = current_time + job_durations[city][job_type];
                max_job_completion = std::max(max_job_completion, job_completion);
                current_city = city;
            }
            
            return std::max(travel_time, max_job_completion);
        }
    };
    
    class JobReassignTSPJ : public LocalSearch<T> {
    private:
        const std::vector<std::vector<T>>& travel_times;
        const std::vector<std::vector<T>>& job_durations;
        int num_cities, num_jobs;
        
    public:
        JobReassignTSPJ(const std::vector<std::vector<T>>& travel, const std::vector<std::vector<T>>& jobs,
                        int cities, int job_types) 
            : LocalSearch<T>("TSPJ-JobReassign"), travel_times(travel), job_durations(jobs), 
              num_cities(cities), num_jobs(job_types) {}
        
        Individual<T> improve(const Individual<T>& individual) override {
            Individual<T> best = individual;
            auto solution = decode_solution(individual);
            
            bool improved = true;
            int iterations = 0;
            
            while (improved && iterations < this->max_iterations) {
                improved = false;
                
                for (int i = 0; i < num_jobs - 1 && !improved; i++) {
                    for (int j = i + 1; j < num_jobs && !improved; j++) {
                        auto new_solution = solution;
                        std::swap(new_solution.second[i], new_solution.second[j]);
                        
                        T new_fitness = calculate_makespan(new_solution);
                        if (new_fitness < best.fitness) {
                            solution = new_solution;
                            best.set_fitness(new_fitness);
                            improved = true;
                        }
                    }
                }
                iterations++;
            }
            
            if (best.fitness < individual.fitness) {
                encode_solution(best, solution);
            }
            
            return best;
        }
        
        bool should_apply(int generation, const Individual<T>& individual, 
                         const std::vector<Individual<T>>& population) override {
            return generation % 2 == 0;
        }
        
        void configure(const std::map<std::string, std::string>& params) override {
            this->parse_basic_config(params);
        }
        
        LocalSearch<T>* clone() const override {
            return new JobReassignTSPJ(travel_times, job_durations, num_cities, num_jobs);
        }
        
    protected:
        bool is_better(T fitness1, T fitness2) const override {
            return fitness1 < fitness2;
        }
        
    private:
        std::pair<std::vector<int>, std::vector<int>> decode_solution(const Individual<T>& individual) {
            const auto& city_chromosome = individual.get_component(0);
            const auto& job_chromosome = individual.get_component(1);
            
            std::vector<std::pair<T, int>> keyed_cities;
            for (int i = 0; i < num_cities; i++) {
                keyed_cities.emplace_back(city_chromosome[i], i + 1);
            }
            std::sort(keyed_cities.begin(), keyed_cities.end());
            
            std::vector<int> city_tour;
            for (const auto& pair : keyed_cities) {
                city_tour.push_back(pair.second);
            }
            
            std::vector<std::pair<T, int>> keyed_jobs;
            for (int i = 0; i < num_jobs; i++) {
                keyed_jobs.emplace_back(job_chromosome[i], i + 1);
            }
            std::sort(keyed_jobs.begin(), keyed_jobs.end());
            
            std::vector<int> job_assignment;
            for (int i = 0; i < num_jobs; i++) {
                job_assignment.push_back(keyed_jobs[i].second);
            }
            
            return {city_tour, job_assignment};
        }
        
        void encode_solution(Individual<T>& individual, const std::pair<std::vector<int>, std::vector<int>>& solution) {
            auto& city_chromosome = individual.get_component(0);
            auto& job_chromosome = individual.get_component(1);
            
            for (int i = 0; i < num_cities; i++) {
                city_chromosome[solution.first[i] - 1] = static_cast<T>(i) / static_cast<T>(num_cities);
            }
            
            for (int i = 0; i < num_jobs; i++) {
                job_chromosome[solution.second[i] - 1] = static_cast<T>(i) / static_cast<T>(num_jobs);
            }
            
            individual.reset_evaluation();
        }
        
        T calculate_makespan(const std::pair<std::vector<int>, std::vector<int>>& solution) {
            const auto& city_tour = solution.first;
            const auto& job_assignment = solution.second;
            
            T travel_time = 0;
            int current_city = 0;
            for (int city : city_tour) {
                travel_time += travel_times[current_city][city];
                current_city = city;
            }
            travel_time += travel_times[current_city][0];
            
            T max_job_completion = 0;
            T current_time = 0;
            current_city = 0;
            
            for (size_t i = 0; i < city_tour.size(); i++) {
                int city = city_tour[i];
                int job_type = job_assignment[i];
                
                current_time += travel_times[current_city][city];
                T job_completion = current_time + job_durations[city][job_type];
                max_job_completion = std::max(max_job_completion, job_completion);
                current_city = city;
            }
            
            return std::max(travel_time, max_job_completion);
        }
    };

public:
    T calculate_tspj_fitness(const Individual<T>& individual) {
        auto solution = decode_solution_internal(individual);
        auto& city_tour = solution.first;
        auto& job_assignment = solution.second;
        
        T travel_time = 0;
        int current_city = 0;
        
        for (int city : city_tour) {
            travel_time += travel_times[current_city][city];
            current_city = city;
        }
        travel_time += travel_times[current_city][0];
        
        T max_job_completion = 0;
        T current_time = 0;
        current_city = 0;
        
        for (size_t i = 0; i < city_tour.size(); i++) {
            int city = city_tour[i];
            int job_type = job_assignment[i];
            
            current_time += travel_times[current_city][city];
            T job_completion_time = current_time + job_durations[city][job_type];
            max_job_completion = std::max(max_job_completion, job_completion_time);
            
            current_city = city;
        }
        
        return std::max(travel_time, max_job_completion);
    }
    
    std::vector<std::vector<T>> decode_to_solution(const Individual<T>& individual) {
        auto solution = decode_solution_internal(individual);
        
        std::vector<std::vector<T>> result(2);
        
        result[0].reserve(solution.first.size());
        for (int city : solution.first) {
            result[0].push_back(static_cast<T>(city));
        }
        
        result[1].reserve(solution.second.size());
        for (int job : solution.second) {
            result[1].push_back(static_cast<T>(job));
        }
        
        return result;
    }
    
private:
    std::pair<std::vector<int>, std::vector<int>> decode_solution_internal(const Individual<T>& individual) {
        const auto& city_chromosome = individual.get_component(0);
        const auto& job_chromosome = individual.get_component(1);
        
        std::vector<std::pair<T, int>> keyed_cities;
        keyed_cities.reserve(num_cities);
        
        for (int i = 0; i < num_cities; i++) {
            keyed_cities.emplace_back(city_chromosome[i], i + 1);
        }
        
        std::sort(keyed_cities.begin(), keyed_cities.end());
        
        std::vector<int> city_tour;
        city_tour.reserve(num_cities);
        for (const auto& pair : keyed_cities) {
            city_tour.push_back(pair.second);
        }
        
        std::vector<std::pair<T, int>> keyed_jobs;
        keyed_jobs.reserve(num_jobs);
        
        for (int i = 0; i < num_jobs; i++) {
            keyed_jobs.emplace_back(job_chromosome[i], i + 1);
        }
        
        std::sort(keyed_jobs.begin(), keyed_jobs.end());
        
        std::vector<int> job_assignment;
        job_assignment.reserve(num_jobs);
        
        for (int i = 0; i < num_jobs; i++) {
            job_assignment.push_back(keyed_jobs[i].second);
        }
        
        return {city_tour, job_assignment};
    }
    
public:
    void print_solution(const Individual<T>& individual) override {
        auto solution = decode_solution_internal(individual);
        auto& city_tour = solution.first;
        auto& job_assignment = solution.second;
        
        std::cout << "city sequence: ";
        for (size_t i = 0; i < city_tour.size(); i++) {
            std::cout << city_tour[i];
            if (i < city_tour.size() - 1) {
                std::cout << " ";
            }
        }
        std::cout << std::endl;
        
        std::cout << "job sequence: ";
        for (size_t i = 0; i < job_assignment.size(); i++) {
            std::cout << job_assignment[i];
            if (i < job_assignment.size() - 1) {
                std::cout << " ";
            }
        }
        std::cout << std::endl;
        
        std::cout << "makespan: " << std::fixed << std::setprecision(2) << individual.fitness << std::endl;
    }
    
    bool validate_solution(const Individual<T>& individual) override {
        auto solution = decode_solution_internal(individual);
        auto& city_tour = solution.first;
        auto& job_assignment = solution.second;
        
        std::vector<bool> city_visited(num_cities + 1, false);
        for (int city : city_tour) {
            if (city < 1 || city > num_cities || city_visited[city]) {
                return false;
            }
            city_visited[city] = true;
        }
        
        std::vector<bool> job_assigned(num_jobs + 1, false);
        for (int job : job_assignment) {
            if (job < 1 || job > num_jobs || job_assigned[job]) {
                return false;
            }
            job_assigned[job] = true;
        }
        
        return true;
    }
    
    void export_solution(const Individual<T>& individual, const std::string& filename) override {
        auto solution = decode_solution_internal(individual);
        auto& city_tour = solution.first;
        auto& job_assignment = solution.second;
        
        FileUtils::ensure_directory(FileUtils::get_directory(filename));
        
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot create solution file: " + filename);
        }
        
        file << "TSPJ Solution for " << instance_name << std::endl;
        file << "Makespan: " << std::fixed << std::setprecision(6) << individual.fitness << std::endl;
        
        file << "City sequence: ";
        for (size_t i = 0; i < city_tour.size(); i++) {
            file << city_tour[i];
            if (i < city_tour.size() - 1) file << " ";
        }
        file << std::endl;
        
        file << "Job sequence: ";
        for (size_t i = 0; i < job_assignment.size(); i++) {
            file << job_assignment[i];
            if (i < job_assignment.size() - 1) file << " ";
        }
        file << std::endl;
        
        file.close();
    }
    
    static std::unique_ptr<TSPJConfig<T>> load_from_file(const std::string& filename) {
        std::string basename = FileUtils::get_basename(filename);
        std::string directory = FileUtils::get_directory(filename);

        std::string tt_file, jt_file;
        std::string instance_name;

        // Support original format: *_TT.csv and *_JT.csv (e.g., berlin52)
        if (basename.find("_TT") != std::string::npos) {
            tt_file = filename;
            jt_file = directory + "/" + basename.substr(0, basename.find("_TT")) + "_JT.csv";
            instance_name = basename.substr(0, basename.find("_TT"));
        } else if (basename.find("_JT") != std::string::npos) {
            jt_file = filename;
            tt_file = directory + "/" + basename.substr(0, basename.find("_JT")) + "_TT.csv";
            instance_name = basename.substr(0, basename.find("_JT"));
        }
        // Support Medium_problems format: *_cost_table_by_coordinates.csv and *_tasktime_table.csv
        else if (basename.find("_cost_table_by_coordinates") != std::string::npos) {
            tt_file = filename;
            instance_name = basename.substr(0, basename.find("_cost_table_by_coordinates"));
            jt_file = directory + "/" + instance_name + "_tasktime_table.csv";
        } else if (basename.find("_tasktime_table") != std::string::npos) {
            jt_file = filename;
            instance_name = basename.substr(0, basename.find("_tasktime_table"));
            tt_file = directory + "/" + instance_name + "_cost_table_by_coordinates.csv";
        } else {
            throw std::runtime_error("TSPJ CSV files must be named with '_TT'/'_JT' or '_cost_table_by_coordinates'/'_tasktime_table' suffix");
        }

        if (!FileUtils::file_exists(tt_file)) {
            throw std::runtime_error("Travel times file not found: " + tt_file);
        }
        if (!FileUtils::file_exists(jt_file)) {
            throw std::runtime_error("Job times file not found: " + jt_file);
        }

        auto travel_matrix = load_csv_matrix(tt_file);
        auto job_matrix = load_csv_matrix(jt_file);

        return std::make_unique<TSPJConfig<T>>(travel_matrix, job_matrix, instance_name);
    }
    
private:
    static std::vector<std::vector<T>> load_csv_matrix(const std::string& filename) {
        auto lines = FileUtils::read_lines(filename);
        std::vector<std::vector<T>> matrix;
        
        for (const auto& line : lines) {
            if (line.empty() || line[0] == '#') continue;
            
            std::vector<T> row;
            std::stringstream ss(line);
            std::string cell;
            
            while (std::getline(ss, cell, ',')) {
                cell.erase(0, cell.find_first_not_of(" \t\r\n"));
                cell.erase(cell.find_last_not_of(" \t\r\n") + 1);
                
                if (!cell.empty()) {
                    try {
                        T value = static_cast<T>(std::stod(cell));
                        row.push_back(value);
                    } catch (const std::exception& e) {
                        row.push_back(T(0));
                    }
                }
            }
            
            if (!row.empty()) {
                matrix.push_back(row);
            }
        }
        
        return matrix;
    }

public:
    static void configure_for_size(TSPJConfig<T>* config, int num_cities) {
        if (num_cities <= 10) {
            config->population_size = 300;
            config->elite_size = 60;
            config->mutant_size = 30;
            config->max_generations = 500;
        } else if (num_cities <= 20) {
            config->population_size = 600;
            config->elite_size = 120;
            config->mutant_size = 60;
            config->max_generations = 1000;
        } else if (num_cities <= 50) {
            config->population_size = 1000;
            config->elite_size = 200;
            config->mutant_size = 100;
            config->max_generations = 2000;
        } else {
            config->population_size = 2000;
            config->elite_size = 350;
            config->mutant_size = 200;
            config->max_generations = 30000;
        }
        config->elite_prob = 0.75;
        config->update_cuda_grid_size();
    }

    static void configure_for_multi_gpu(TSPJConfig<T>* config, int num_cities, int num_gpus) {
        // Scale population based on problem size and GPU count
        int base_pop = (num_cities <= 100) ? 1000 :
                       (num_cities <= 300) ? 2000 :
                       (num_cities <= 500) ? 3000 : 4000;

        config->population_size = base_pop * num_gpus;
        config->elite_size = config->population_size / 5;      // 20% elite
        config->mutant_size = config->population_size / 10;    // 10% mutants

        // Scale generations based on problem size
        if (num_cities <= 100) {
            config->max_generations = 1000;
        } else if (num_cities <= 300) {
            config->max_generations = 2000;
        } else if (num_cities <= 500) {
            config->max_generations = 5000;
        } else {
            config->max_generations = 10000;
        }

        config->elite_prob = 0.75;
        config->update_cuda_grid_size();
    }
    
    int get_num_cities() const { return num_cities; }
    int get_num_jobs() const { return num_jobs; }
    const std::string& get_instance_name() const { return instance_name; }
    
    void print_instance_info() const {
        std::cout << "TSPJ Instance: " << instance_name << std::endl;
        std::cout << "Cities to visit: " << num_cities << " (excluding depot)" << std::endl;
        std::cout << "Job types: " << num_jobs << " (excluding depot)" << std::endl;
        std::cout << "Depot: city 0 (no job processing)" << std::endl;
    }
};

// =============================================================================
// GPU Fitness Kernel for TSPJ
// =============================================================================
// This kernel evaluates multiple individuals in parallel on the GPU.
// Each thread processes one individual's chromosome to compute its makespan.

template<typename T>
__global__ void tspj_fitness_kernel(T* population, T* fitness,
                                    T* travel_matrix, T* job_matrix,
                                    int pop_size, int chrom_len, int num_cities, int matrix_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    // Each individual has 2 components: cities (num_cities) + jobs (num_cities)
    int offset = idx * chrom_len;
    T* city_keys = population + offset;
    T* job_keys = population + offset + num_cities;

    // Decode city tour using selection sort on keys
    // city_order[i] = the city visited at position i (1-indexed cities)
    // We use local memory arrays (limited to reasonable sizes)
    // For very large problems (>1500 cities), GPU evaluation may not be efficient

    int city_order[1500];  // Max 1500 cities per thread
    int job_order[1500];

    // Bounds check - skip if problem too large
    if (num_cities > 1500) {
        fitness[idx] = 1e9f;  // Mark as invalid
        return;
    }

    // Initialize with city indices (1-indexed: cities 1 to num_cities)
    for (int i = 0; i < num_cities; i++) {
        city_order[i] = i + 1;
        job_order[i] = i + 1;
    }

    // Selection sort for cities based on random keys
    for (int i = 0; i < num_cities - 1; i++) {
        int min_idx = i;
        T min_key = city_keys[city_order[i] - 1];

        for (int j = i + 1; j < num_cities; j++) {
            T key = city_keys[city_order[j] - 1];
            if (key < min_key) {
                min_key = key;
                min_idx = j;
            }
        }

        if (min_idx != i) {
            int temp = city_order[i];
            city_order[i] = city_order[min_idx];
            city_order[min_idx] = temp;
        }
    }

    // Selection sort for jobs based on random keys
    for (int i = 0; i < num_cities - 1; i++) {
        int min_idx = i;
        T min_key = job_keys[job_order[i] - 1];

        for (int j = i + 1; j < num_cities; j++) {
            T key = job_keys[job_order[j] - 1];
            if (key < min_key) {
                min_key = key;
                min_idx = j;
            }
        }

        if (min_idx != i) {
            int temp = job_order[i];
            job_order[i] = job_order[min_idx];
            job_order[min_idx] = temp;
        }
    }

    // Calculate total travel time (depot -> all cities -> depot)
    T total_travel = 0;
    int current_city = 0;  // Start at depot

    for (int i = 0; i < num_cities; i++) {
        int next_city = city_order[i];
        total_travel += travel_matrix[current_city * matrix_dim + next_city];
        current_city = next_city;
    }
    // Return to depot
    total_travel += travel_matrix[current_city * matrix_dim + 0];

    // Calculate max job completion time
    T max_job_completion = 0;
    T current_time = 0;
    current_city = 0;  // Start at depot

    for (int i = 0; i < num_cities; i++) {
        int city = city_order[i];
        int job_type = job_order[i];

        // Travel to city
        current_time += travel_matrix[current_city * matrix_dim + city];

        // Job completion = arrival time + job duration
        T job_completion = current_time + job_matrix[city * matrix_dim + job_type];

        if (job_completion > max_job_completion) {
            max_job_completion = job_completion;
        }

        current_city = city;
    }

    // Makespan = max(total_travel, max_job_completion)
    fitness[idx] = (total_travel > max_job_completion) ? total_travel : max_job_completion;
}

#endif // TSPJ_CONFIG_HPP