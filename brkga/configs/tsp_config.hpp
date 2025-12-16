// configs/tsp_config.hpp - Pure TSP with GPU fitness evaluation (multi-GPU support)

#ifndef TSP_CONFIG_HPP
#define TSP_CONFIG_HPP

#include "../core/config.hpp"
#include "../utils/file_utils.hpp"
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iostream>
#include <map>
#include <mutex>
#include <cuda_runtime.h>

// Forward declaration of GPU kernel
template<typename T>
__global__ void tsp_fitness_kernel(T* population, T* fitness, T* distance_matrix,
                                   int pop_size, int chrom_len, int num_cities);

template<typename T>
class TSPConfig : public BRKGAConfig<T> {
private:
    std::vector<std::vector<T>> distance_matrix;
    int num_cities;
    std::string instance_name;

    // GPU-specific members (per-GPU storage for multi-GPU support)
    std::map<int, T*> d_distance_matrices;  // device_id -> matrix
    bool gpu_available;
    mutable std::mutex gpu_mutex;  // Thread safety for multi-GPU allocation

public:
    // Pure TSP: N x N distance matrix, visit all N cities
    TSPConfig(const std::vector<std::vector<T>>& distances,
              const std::string& name = "TSP")
        : BRKGAConfig<T>({static_cast<int>(distances.size())}),  // chromosome length = N
          distance_matrix(distances),
          num_cities(distances.size()),  // N cities (no depot subtraction)
          instance_name(name),
          gpu_available(false) {

        this->fitness_function = [this](const Individual<T>& individual) {
            return calculate_tsp_fitness(individual);
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

    ~TSPConfig() {
        cleanup_all_gpu_memory();
    }

    // GPU evaluation interface
    bool has_gpu_evaluation() const override { return gpu_available; }

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

        // Get device-specific pointer
        T* d_dist = nullptr;
        {
            std::lock_guard<std::mutex> lock(gpu_mutex);
            auto it = d_distance_matrices.find(device_id);
            if (it != d_distance_matrices.end()) {
                d_dist = it->second;
            }
        }

        if (!d_dist) return;

        dim3 block(this->threads_per_block);
        dim3 grid((pop_size + block.x - 1) / block.x);

        tsp_fitness_kernel<<<grid, block>>>(
            d_population, d_fitness, d_dist,
            pop_size, chrom_len, num_cities
        );

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

        // Check if already allocated on this device
        if (d_distance_matrices.find(device_id) != d_distance_matrices.end()) {
            return;
        }

        // Allocate on specified device
        cudaSetDevice(device_id);

        int matrix_size = num_cities * num_cities;
        T* d_dist = nullptr;

        cudaError_t error = cudaMalloc(&d_dist, matrix_size * sizeof(T));
        if (error != cudaSuccess) {
            std::cerr << "GPU memory allocation failed on device " << device_id
                      << ": " << cudaGetErrorString(error) << std::endl;
            return;
        }

        // Flatten and copy distance matrix to GPU
        std::vector<T> flat_matrix(matrix_size);
        for (int i = 0; i < num_cities; i++) {
            for (int j = 0; j < num_cities; j++) {
                flat_matrix[i * num_cities + j] = distance_matrix[i][j];
            }
        }

        error = cudaMemcpy(d_dist, flat_matrix.data(),
                          matrix_size * sizeof(T), cudaMemcpyHostToDevice);

        if (error != cudaSuccess) {
            std::cerr << "GPU memory copy failed on device " << device_id
                      << ": " << cudaGetErrorString(error) << std::endl;
            cudaFree(d_dist);
            return;
        }

        d_distance_matrices[device_id] = d_dist;
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
    }

    // Pure TSP fitness: cycle through all cities
    T calculate_tsp_fitness(const Individual<T>& individual) {
        auto tour = decode_tour_internal(individual);

        T total_distance = 0;

        // Tour is a cycle: tour[0] -> tour[1] -> ... -> tour[N-1] -> tour[0]
        for (int i = 0; i < num_cities - 1; i++) {
            total_distance += distance_matrix[tour[i]][tour[i + 1]];
        }
        // Return to start
        total_distance += distance_matrix[tour[num_cities - 1]][tour[0]];

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

        // Pure TSP: N x N distance matrix (no depot)
        // Matrix[i][j] = distance from city i to city j (0-indexed)
        // File cities 1..N map to indices 0..N-1
        std::vector<std::vector<T>> distances(num_cities, std::vector<T>(num_cities, 0));

        for (int i = 0; i < num_cities; i++) {
            for (int j = 0; j < num_cities; j++) {
                if (i == j) {
                    distances[i][j] = 0;
                } else {
                    T dx = coordinates[i].first - coordinates[j].first;
                    T dy = coordinates[i].second - coordinates[j].second;
                    distances[i][j] = std::sqrt(dx * dx + dy * dy);
                }
            }
        }

        if (instance_name.empty()) {
            instance_name = FileUtils::get_basename(filename);
        }

        std::cout << "Created " << num_cities << "x" << num_cities << " distance matrix" << std::endl;

        return std::make_unique<TSPConfig<T>>(distances, instance_name);
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