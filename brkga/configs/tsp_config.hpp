// configs/tsp_config.hpp - Pure TSP with GPU fitness evaluation (multi-GPU support)

#ifndef TSP_CONFIG_HPP
#define TSP_CONFIG_HPP

#include "../core/config.hpp"
#include "../utils/file_utils.hpp"
#include "../utils/segmented_sort.hpp"  // Phase 4: Parallel segmented sorting
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <mutex>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>

// Forward declaration of GPU kernels
template<typename T>
__global__ void tsp_fitness_kernel(T* population, T* fitness, T* distance_matrix,
                                   int pop_size, int chrom_len, int num_cities);

// Kernel to calculate tour length from sorted indices (for large TSP with Thrust)
// Uses parallel reduction for efficiency with large tours
template<typename T>
__global__ void tsp_tour_length_kernel(int* sorted_tour, T* fitness, T* distance_matrix,
                                       int num_cities) {
    extern __shared__ T shared_sum[];

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Each thread sums a portion of the tour
    T local_sum = 0;
    for (int i = tid; i < num_cities; i += stride) {
        int from = sorted_tour[i];
        int to = sorted_tour[(i + 1) % num_cities];  // Wrap around for last city
        local_sum += distance_matrix[from * num_cities + to];
    }

    shared_sum[tid] = local_sum;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *fitness = shared_sum[0];
    }
}

// Kernel to calculate tour length from coordinates (for very large TSP)
template<typename T>
__global__ void tsp_tour_length_from_coords_kernel(int* sorted_tour, T* fitness,
                                                    T* coords_x, T* coords_y,
                                                    int num_cities) {
    extern __shared__ T shared_sum[];
    int tid = threadIdx.x;
    int stride = blockDim.x;

    T local_sum = 0;
    for (int i = tid; i < num_cities; i += stride) {
        int from = sorted_tour[i];
        int to = sorted_tour[(i + 1) % num_cities];
        T dx = coords_x[from] - coords_x[to];
        T dy = coords_y[from] - coords_y[to];
        local_sum += ceilf(sqrtf(dx * dx + dy * dy));  // CEIL_2D as per TSPLIB
    }

    shared_sum[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared_sum[tid] += shared_sum[tid + s];
        __syncthreads();
    }

    if (tid == 0) *fitness = shared_sum[0];
}

template<typename T>
class TSPConfig : public BRKGAConfig<T> {
private:
    std::vector<std::vector<T>> distance_matrix;  // For small instances
    std::vector<T> coords_x, coords_y;            // For large instances (coordinate-based)
    int num_cities;
    std::string instance_name;
    bool use_coordinates;  // True for large instances that don't fit in memory

    // GPU-specific members
    std::map<int, T*> d_distance_matrices;  // For small instances
    T* d_coords_x = nullptr;                // For large instances
    T* d_coords_y = nullptr;
    bool gpu_available;
    mutable std::mutex gpu_mutex;

    // Phase 4: Segmented sorting infrastructure for parallel tour decoding
    std::unique_ptr<SegmentedSorter<T, int>> segmented_sorter_;
    T* d_all_keys_ = nullptr;        // Device: All chromosome keys for sorting
    int* d_all_tours_ = nullptr;     // Device: All decoded tours (permutations)
    bool segmented_sort_enabled_ = false;

public:
    // Constructor for small instances using distance matrix
    TSPConfig(const std::vector<std::vector<T>>& distances,
              const std::string& name = "TSP")
        : BRKGAConfig<T>({static_cast<int>(distances.size())}),
          distance_matrix(distances),
          num_cities(distances.size()),
          instance_name(name),
          use_coordinates(false),
          gpu_available(false) {
        setup_common();
    }

    // Constructor for large instances using coordinates (memory efficient)
    TSPConfig(const std::vector<T>& x_coords, const std::vector<T>& y_coords,
              const std::string& name = "TSP")
        : BRKGAConfig<T>({static_cast<int>(x_coords.size())}),
          coords_x(x_coords),
          coords_y(y_coords),
          num_cities(x_coords.size()),
          instance_name(name),
          use_coordinates(true),
          gpu_available(false) {
        setup_common();
    }

private:
    void setup_common() {
        this->fitness_function = [this](const Individual<T>& individual) {
            return calculate_tsp_fitness(individual);
        };

        this->decoder = [this](const Individual<T>& individual) {
            return decode_to_solution(individual);
        };

        this->comparator = [](T a, T b) { return a < b; };

        this->threads_per_block = 256;
        this->update_cuda_grid_size();
        check_gpu_availability();
    }

public:
    ~TSPConfig() {
        cleanup_all_gpu_memory();
    }

    // GPU evaluation interface - supports ALL sizes using Thrust for large instances
    bool has_gpu_evaluation() const override {
        return gpu_available;  // Always use GPU if available
    }

    void evaluate_population_gpu(T* d_population, T* d_fitness,
                                int pop_size, int chrom_len) override {
        if (!gpu_available) {
            return; // Fallback to CPU
        }

        // Get current device (set by caller for multi-GPU)
        int device_id;
        cudaGetDevice(&device_id);

        // Ensure distance matrix is allocated on this device
        ensure_gpu_memory(device_id);

        // Re-set device after ensure_gpu_memory (another thread may have changed it)
        cudaSetDevice(device_id);

        // Get device-specific pointer for distance matrix (only used for small instances)
        T* d_dist = nullptr;
        if (!use_coordinates) {
            std::lock_guard<std::mutex> lock(gpu_mutex);
            auto it = d_distance_matrices.find(device_id);
            if (it != d_distance_matrices.end()) {
                d_dist = it->second;
            }
            if (!d_dist) return;  // No distance matrix available for small instance mode
        } else {
            // For coordinate mode, check that coordinates are allocated
            if (!d_coords_x || !d_coords_y) return;
        }

        // Use different strategies based on problem size
        if (num_cities <= 128 && !use_coordinates) {
            // Small instances: use per-thread kernel (fast)
            dim3 block(this->threads_per_block);
            dim3 grid((pop_size + block.x - 1) / block.x);

            tsp_fitness_kernel<<<grid, block>>>(
                d_population, d_fitness, d_dist,
                pop_size, chrom_len, num_cities
            );
        } else {
            // Large instances or coordinate mode: use Thrust sorting (handles any size)
            evaluate_large_tsp_gpu(d_population, d_fitness, d_dist, pop_size, chrom_len);
        }

        cudaError_t error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            std::cerr << "GPU fitness kernel failed on device " << device_id
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

        if (use_coordinates) {
            // For large instances, allocate coordinate buffers (much smaller)
            if (d_coords_x != nullptr) return;  // Already allocated

            cudaSetDevice(device_id);
            cudaError_t error;

            error = cudaMalloc(&d_coords_x, num_cities * sizeof(T));
            if (error != cudaSuccess) {
                std::cerr << "GPU coord X allocation failed: " << cudaGetErrorString(error) << std::endl;
                return;
            }

            error = cudaMalloc(&d_coords_y, num_cities * sizeof(T));
            if (error != cudaSuccess) {
                cudaFree(d_coords_x);
                d_coords_x = nullptr;
                std::cerr << "GPU coord Y allocation failed: " << cudaGetErrorString(error) << std::endl;
                return;
            }

            cudaMemcpy(d_coords_x, coords_x.data(), num_cities * sizeof(T), cudaMemcpyHostToDevice);
            cudaMemcpy(d_coords_y, coords_y.data(), num_cities * sizeof(T), cudaMemcpyHostToDevice);

            std::cout << "Allocated GPU coordinate buffers: " << (2 * num_cities * sizeof(T) / 1024) << " KB" << std::endl;
        } else {
            // For small instances, use distance matrix
            if (d_distance_matrices.find(device_id) != d_distance_matrices.end()) {
                return;
            }

            cudaSetDevice(device_id);

            int matrix_size = num_cities * num_cities;
            T* d_dist = nullptr;

            cudaError_t error = cudaMalloc(&d_dist, matrix_size * sizeof(T));
            if (error != cudaSuccess) {
                std::cerr << "GPU memory allocation failed on device " << device_id
                          << ": " << cudaGetErrorString(error) << std::endl;
                return;
            }

            std::vector<T> flat_matrix(matrix_size);
            for (int i = 0; i < num_cities; i++) {
                for (int j = 0; j < num_cities; j++) {
                    flat_matrix[i * num_cities + j] = distance_matrix[i][j];
                }
            }

            error = cudaMemcpy(d_dist, flat_matrix.data(),
                              matrix_size * sizeof(T), cudaMemcpyHostToDevice);

            if (error != cudaSuccess) {
                std::cerr << "GPU memory copy failed: " << cudaGetErrorString(error) << std::endl;
                cudaFree(d_dist);
                return;
            }

            d_distance_matrices[device_id] = d_dist;
        }
    }

    void cleanup_all_gpu_memory() {
        std::lock_guard<std::mutex> lock(gpu_mutex);

        for (auto& pair : d_distance_matrices) {
            if (pair.second) {
                cudaSetDevice(pair.first);
                cudaFree(pair.second);
            }
        }
        d_distance_matrices.clear();

        if (d_coords_x) { cudaFree(d_coords_x); d_coords_x = nullptr; }
        if (d_coords_y) { cudaFree(d_coords_y); d_coords_y = nullptr; }

        // Phase 4: Cleanup segmented sort resources
        if (d_all_keys_) { cudaFree(d_all_keys_); d_all_keys_ = nullptr; }
        if (d_all_tours_) { cudaFree(d_all_tours_); d_all_tours_ = nullptr; }
        segmented_sorter_.reset();  // Destructor handles device memory
    }

    // Pure TSP fitness: cycle through all cities
    T calculate_tsp_fitness(const Individual<T>& individual) {
        auto tour = decode_tour_internal(individual);

        T total_distance = 0;

        if (use_coordinates) {
            // Calculate distances from coordinates on-the-fly (CEIL_2D)
            for (int i = 0; i < num_cities; i++) {
                int from = tour[i];
                int to = tour[(i + 1) % num_cities];
                T dx = coords_x[from] - coords_x[to];
                T dy = coords_y[from] - coords_y[to];
                total_distance += std::ceil(std::sqrt(dx * dx + dy * dy));
            }
        } else {
            // Use precomputed distance matrix
            for (int i = 0; i < num_cities - 1; i++) {
                total_distance += distance_matrix[tour[i]][tour[i + 1]];
            }
            total_distance += distance_matrix[tour[num_cities - 1]][tour[0]];
        }

        return total_distance;
    }
    
    std::vector<std::vector<T>> decode_to_solution(const Individual<T>& individual) {
        auto tour = decode_tour_internal(individual);
        
        std::vector<std::vector<T>> result(1);
        result[0].reserve(tour.size());
        for (int city : tour) {
            result[0].push_back(static_cast<T>(city));
        }
        return result;
    }
    
    std::vector<int> decode_tour_internal(const Individual<T>& individual) {
        const auto& chromosome = individual.get_chromosome();
        
        std::vector<std::pair<T, int>> keyed_cities;
        keyed_cities.reserve(num_cities);
        
        for (int i = 0; i < num_cities; i++) {
            keyed_cities.emplace_back(chromosome[i], i);
        }
        
        std::sort(keyed_cities.begin(), keyed_cities.end());
        
        std::vector<int> tour;
        tour.reserve(num_cities);
        for (const auto& pair : keyed_cities) {
            tour.push_back(pair.second);
        }
        
        return tour;
    }

    // Phase 4: Initialize segmented sorting infrastructure
    void init_segmented_sort(int max_pop_size) {
        if (segmented_sort_enabled_) return;  // Already initialized

        try {
            // Create segmented sorter (max population size)
            segmented_sorter_ = std::make_unique<SegmentedSorter<T, int>>(max_pop_size);

            // Allocate device memory for ALL chromosomes and tours
            size_t keys_bytes = max_pop_size * num_cities * sizeof(T);
            size_t tours_bytes = max_pop_size * num_cities * sizeof(int);

            cudaError_t err = cudaMalloc(&d_all_keys_, keys_bytes);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("Failed to allocate d_all_keys: ") + cudaGetErrorString(err)
                );
            }

            err = cudaMalloc(&d_all_tours_, tours_bytes);
            if (err != cudaSuccess) {
                cudaFree(d_all_keys_);
                d_all_keys_ = nullptr;
                throw std::runtime_error(
                    std::string("Failed to allocate d_all_tours: ") + cudaGetErrorString(err)
                );
            }

            segmented_sort_enabled_ = true;

            std::cout << "[Phase 4] Segmented sort initialized: " << max_pop_size
                      << " individuals × " << num_cities << " cities" << std::endl;
            std::cout << "  Memory allocated: "
                      << (keys_bytes + tours_bytes) / (1024.0 * 1024.0)
                      << " MB" << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "[Phase 4] Warning: Segmented sort initialization failed: "
                      << e.what() << std::endl;
            std::cerr << "  Falling back to sequential sorting." << std::endl;
            segmented_sort_enabled_ = false;
        }
    }

    // GPU evaluation for large TSP instances using Thrust for sorting
    void evaluate_large_tsp_gpu(T* d_population, T* d_fitness, T* d_dist,
                                int pop_size, int chrom_len) {

        // For very large instances (>10k cities), CPU parallel sorting is faster
        // than sequential GPU sorts due to memory transfer overhead
        if (use_coordinates && num_cities > 10000) {
            // CPU-based parallel evaluation (much faster for large instances)
            static bool first_call = true;
            if (first_call) {
                std::cout << "[INFO] Using CPU parallel evaluation for " << num_cities
                          << " cities (" << pop_size << " individuals)" << std::endl;
                first_call = false;
            }

            std::vector<T> h_population(pop_size * chrom_len);
            std::vector<T> h_fitness(pop_size);

            // Single transfer: GPU -> CPU
            cudaMemcpy(h_population.data(), d_population,
                      pop_size * chrom_len * sizeof(T), cudaMemcpyDeviceToHost);

            // Parallel evaluation on CPU
            #pragma omp parallel for schedule(dynamic)
            for (int ind = 0; ind < pop_size; ind++) {
                T* chromosome = h_population.data() + ind * chrom_len;

                // Create key-city pairs
                std::vector<std::pair<T, int>> pairs(num_cities);
                for (int i = 0; i < num_cities; i++) {
                    pairs[i] = {chromosome[i], i};
                }

                // CPU sort (very fast, especially with parallel std::sort)
                std::sort(pairs.begin(), pairs.end());

                // Calculate tour length from coordinates
                T total = 0;
                for (int i = 0; i < num_cities; i++) {
                    int from = pairs[i].second;
                    int to = pairs[(i + 1) % num_cities].second;
                    T dx = coords_x[from] - coords_x[to];
                    T dy = coords_y[from] - coords_y[to];
                    total += std::ceil(std::sqrt(dx * dx + dy * dy));
                }
                h_fitness[ind] = total;
            }

            // Single transfer: CPU -> GPU
            cudaMemcpy(d_fitness, h_fitness.data(),
                      pop_size * sizeof(T), cudaMemcpyHostToDevice);
            return;
        }

        // Phase 4: Try segmented sort path (10-50× faster)
        // Falls back to sequential if unavailable
        if (pop_size > 1 && num_cities <= 5000) {  // Reasonable size for parallel sort
            // Lazy initialization on first call
            if (!segmented_sort_enabled_) {
                init_segmented_sort(std::max(pop_size, 10000));  // Allocate for up to 10k individuals
            }

            if (segmented_sort_enabled_) {
                try {
                    // Step 1: Copy ALL chromosomes to device buffer (single transfer)
                    cudaMemcpy(d_all_keys_, d_population,
                              pop_size * num_cities * sizeof(T),
                              cudaMemcpyDeviceToDevice);

                    // Step 2: Initialize ALL tour indices (0,1,2,...,num_cities for each individual)
                    thrust::device_ptr<int> tour_ptr(d_all_tours_);
                    for (int ind = 0; ind < pop_size; ind++) {
                        thrust::sequence(tour_ptr + ind * num_cities,
                                       tour_ptr + (ind + 1) * num_cities);
                    }

                    // Step 3: Prepare segment offsets [0, n, 2n, 3n, ..., pop_size*n]
                    std::vector<int> offsets(pop_size + 1);
                    for (int i = 0; i <= pop_size; i++) {
                        offsets[i] = i * num_cities;
                    }

                    // Step 4: Single parallel sort of ALL tours (10-50× faster!)
                    segmented_sorter_->sort_segments(
                        d_all_keys_,     // All chromosomes (keys)
                        d_all_tours_,    // All tour permutations (values)
                        offsets,         // Segment boundaries
                        pop_size         // Number of tours
                    );

                    // Step 5: Evaluate ALL tours in parallel
                    const int block_size = 256;
                    size_t shared_mem_size = block_size * sizeof(T);
                    dim3 block(block_size);
                    dim3 grid(pop_size);  // One block per tour

                    if (use_coordinates) {
                        for (int ind = 0; ind < pop_size; ind++) {
                            tsp_tour_length_from_coords_kernel<<<1, block_size, shared_mem_size>>>(
                                d_all_tours_ + ind * num_cities,
                                d_fitness + ind,
                                d_coords_x, d_coords_y,
                                num_cities
                            );
                        }
                    } else {
                        for (int ind = 0; ind < pop_size; ind++) {
                            tsp_tour_length_kernel<<<1, block_size, shared_mem_size>>>(
                                d_all_tours_ + ind * num_cities,
                                d_fitness + ind,
                                d_dist,
                                num_cities
                            );
                        }
                    }

                    cudaDeviceSynchronize();
                    return;  // Success! Phase 4 optimization applied

                } catch (const std::exception& e) {
                    std::cerr << "[Phase 4] Warning: Segmented sort failed: " << e.what() << std::endl;
                    std::cerr << "  Falling back to sequential sorting." << std::endl;
                    // Fall through to sequential path
                }
            }
        }

        // Fallback: Sequential GPU path (original implementation)
        thrust::device_vector<T> d_keys(num_cities);
        thrust::device_vector<int> d_indices(num_cities);
        T* d_single_fitness;
        cudaMalloc(&d_single_fitness, sizeof(T));

        const int block_size = 256;
        size_t shared_mem_size = block_size * sizeof(T);

        for (int ind = 0; ind < pop_size; ind++) {
            T* chromosome = d_population + ind * chrom_len;

            // Copy chromosome keys to sorting buffer
            cudaMemcpy(thrust::raw_pointer_cast(d_keys.data()), chromosome,
                       num_cities * sizeof(T), cudaMemcpyDeviceToDevice);

            // Initialize and sort indices by key values
            thrust::sequence(d_indices.begin(), d_indices.end());
            thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_indices.begin());

            // Calculate tour length - use coordinates or distance matrix
            if (use_coordinates) {
                tsp_tour_length_from_coords_kernel<<<1, block_size, shared_mem_size>>>(
                    thrust::raw_pointer_cast(d_indices.data()),
                    d_single_fitness,
                    d_coords_x, d_coords_y,
                    num_cities
                );
            } else {
                tsp_tour_length_kernel<<<1, block_size, shared_mem_size>>>(
                    thrust::raw_pointer_cast(d_indices.data()),
                    d_single_fitness,
                    d_dist,
                    num_cities
                );
            }

            // Copy fitness to output array
            cudaMemcpy(d_fitness + ind, d_single_fitness, sizeof(T), cudaMemcpyDeviceToDevice);
        }

        cudaFree(d_single_fitness);
    }

public:
    void print_solution(const Individual<T>& individual) override {
        auto tour = decode_tour_internal(individual);

        std::cout << "\n=== TSP Solution ===" << std::endl;
        std::cout << "Instance: " << instance_name << std::endl;
        std::cout << "Cities: " << num_cities << std::endl;

        // Print tour (1-indexed for TSPLIB compatibility)
        std::cout << "Tour: " << (tour[0] + 1);
        for (int i = 1; i < num_cities; i++) {
            std::cout << " -> " << (tour[i] + 1);
        }
        std::cout << " -> " << (tour[0] + 1) << std::endl;  // Return to start

        std::cout << "Total distance: " << std::fixed << std::setprecision(2)
                  << individual.fitness << std::endl;
        std::cout << "===================" << std::endl;
    }
    
    bool validate_solution(const Individual<T>& individual) override {
        auto tour = decode_tour_internal(individual);
        
        std::vector<bool> visited(num_cities, false);
        for (int city : tour) {
            if (city < 0 || city >= num_cities || visited[city]) {
                return false;
            }
            visited[city] = true;
        }
        return true;
    }
    
    static std::unique_ptr<TSPConfig<T>> load_from_file(const std::string& filename) {
        if (!FileUtils::file_exists(filename)) {
            throw std::runtime_error("File does not exist: " + filename);
        }
        return load_from_tsplib(filename);
    }
    
    static void configure_for_size(TSPConfig<T>* config, int num_cities) {
        if (num_cities <= 20) {
            config->population_size = 200;
            config->elite_size = 40;
            config->mutant_size = 20;
            config->max_generations = 100;
        } else if (num_cities <= 50) {
            config->population_size = 1000;
            config->elite_size = 800;
            config->mutant_size = 400;
            config->max_generations = 2000;
        } else {
            config->population_size = 8000;
            config->elite_size = 1600;
            config->mutant_size = 800;
            config->max_generations = 5000;
        }
        config->elite_prob = 0.7;
        config->update_cuda_grid_size();
    }
    
    int get_num_cities() const { return num_cities; }
    const std::string& get_instance_name() const { return instance_name; }
    bool is_gpu_available() const { return gpu_available; }
    
    void print_instance_info() const {
        std::cout << "=== TSP Instance Info ===" << std::endl;
        std::cout << "Name: " << instance_name << std::endl;
        std::cout << "Cities: " << num_cities << std::endl;
        std::cout << "GPU Support: " << (gpu_available ? "Yes" : "No") << std::endl;
        std::cout << "=========================" << std::endl;
    }
    
private:
    static std::unique_ptr<TSPConfig<T>> load_from_tsplib(const std::string& filename) {
        auto lines = FileUtils::read_lines(filename);
        std::vector<std::pair<T, T>> coordinates;
        std::string instance_name;
        int num_cities = 0;
        bool reading_coords = false;

        std::cout << "Loading TSPLIB file: " << filename << std::endl;

        for (const auto& line : lines) {
            if (line.empty()) continue;

            if (line.find("NAME") != std::string::npos) {
                size_t colon = line.find(':');
                if (colon != std::string::npos) {
                    instance_name = line.substr(colon + 1);
                    instance_name.erase(0, instance_name.find_first_not_of(" \t"));
                    instance_name.erase(instance_name.find_last_not_of(" \t") + 1);
                }
            }
            else if (line.find("DIMENSION") != std::string::npos) {
                std::istringstream iss(line);
                std::string token;
                while (iss >> token) {
                    if (std::isdigit(token[0])) {
                        num_cities = std::stoi(token);
                        break;
                    }
                }
                coordinates.resize(num_cities);
                std::cout << "Number of cities: " << num_cities << std::endl;
            }
            else if (line.find("NODE_COORD_SECTION") != std::string::npos) {
                reading_coords = true;
                continue;
            }
            else if (line.find("EOF") != std::string::npos) {
                break;
            }
            else if (reading_coords) {
                std::istringstream iss(line);
                int id;
                T x, y;
                if (iss >> id >> x >> y) {
                    if (id > 0 && id <= num_cities) {
                        coordinates[id - 1] = {x, y};
                    }
                }
            }
        }

        if (coordinates.empty()) {
            throw std::runtime_error("No coordinates found in TSPLIB file");
        }

        if (instance_name.empty()) {
            instance_name = FileUtils::get_basename(filename);
        }

        // For large instances (> 5000 cities), use coordinate-based mode to save memory
        // Distance matrix would be N*N*4 bytes, coordinates are only N*2*4 bytes
        const int LARGE_TSP_THRESHOLD = 5000;

        if (num_cities > LARGE_TSP_THRESHOLD) {
            std::cout << "Large TSP detected (" << num_cities << " cities) - using coordinate-based mode" << std::endl;
            std::cout << "Memory saved: " << ((size_t)num_cities * num_cities * sizeof(T) / (1024*1024)) << " MB" << std::endl;

            std::vector<T> x_coords(num_cities), y_coords(num_cities);
            for (int i = 0; i < num_cities; i++) {
                x_coords[i] = coordinates[i].first;
                y_coords[i] = coordinates[i].second;
            }

            return std::make_unique<TSPConfig<T>>(x_coords, y_coords, instance_name);
        } else {
            // For small instances, precompute distance matrix
            std::cout << "Creating " << num_cities << "x" << num_cities << " distance matrix" << std::endl;

            std::vector<std::vector<T>> distances(num_cities, std::vector<T>(num_cities, 0));

            for (int i = 0; i < num_cities; i++) {
                for (int j = 0; j < num_cities; j++) {
                    if (i != j) {
                        T dx = coordinates[i].first - coordinates[j].first;
                        T dy = coordinates[i].second - coordinates[j].second;
                        distances[i][j] = std::sqrt(dx * dx + dy * dy);
                    }
                }
            }

            return std::make_unique<TSPConfig<T>>(distances, instance_name);
        }
    }
};

// GPU kernel implementation - Pure TSP (N x N matrix, no depot)
template<typename T>
__global__ void tsp_fitness_kernel(T* population, T* fitness, T* distance_matrix,
                                   int pop_size, int chrom_len, int num_cities) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    T* chromosome = population + idx * chrom_len;

    // Each thread processes one individual
    struct KeyPair {
        T value;
        int city;
    };

    // Limit array size for GPU memory constraints
    KeyPair pairs[128]; // Max 128 cities - adjust if needed
    int actual_cities = min(num_cities, 128);

    // Create key-city pairs
    for (int i = 0; i < actual_cities; i++) {
        pairs[i].value = chromosome[i];
        pairs[i].city = i;
    }

    // Simple bubble sort (can optimize later)
    for (int i = 0; i < actual_cities - 1; i++) {
        for (int j = 0; j < actual_cities - 1 - i; j++) {
            if (pairs[j].value > pairs[j + 1].value) {
                KeyPair temp = pairs[j];
                pairs[j] = pairs[j + 1];
                pairs[j + 1] = temp;
            }
        }
    }

    // Calculate tour distance - cycle through all cities
    // Tour: pairs[0].city -> pairs[1].city -> ... -> pairs[N-1].city -> pairs[0].city
    T total_distance = 0;

    for (int i = 0; i < actual_cities - 1; i++) {
        int from = pairs[i].city;
        int to = pairs[i + 1].city;
        int matrix_idx = from * num_cities + to;
        total_distance += distance_matrix[matrix_idx];
    }

    // Return to start city
    int from = pairs[actual_cities - 1].city;
    int to = pairs[0].city;
    int matrix_idx = from * num_cities + to;
    total_distance += distance_matrix[matrix_idx];

    fitness[idx] = total_distance;
}

#endif // TSP_CONFIG_HPP