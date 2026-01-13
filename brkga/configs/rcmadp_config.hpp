// configs/rcmadp_config.hpp
// Resource-Constrained Multi-Agent Drop-and-Pick Problem (RCMADP)
//
// Problem Description:
// - Multiple agents with limited deployable resources
// - Each customer requires a resource dropped off, left to process for a fixed time,
//   then picked up from the same location
// - Dropoff and pickup may be performed by different agents
// - Resources are unavailable while deployed
// - Agents must respect travel times, resource capacity, and pickup-after-processing constraints
// - Objective: Minimize total travel cost while ensuring all resources are retrieved
//
// Inputs:
// - .tsp file: Customer locations (coordinates or distance matrix)
// - Job times matrix: Processing time at each customer location
//
// Chromosome Structure (3 components):
// - Component 0: Dropoff sequence (num_customers genes) - priority order for dropoffs
// - Component 1: Pickup sequence (num_customers genes) - priority order for pickups
// - Component 2: Agent assignment hints (num_customers genes) - soft hints for dropoff agent assignment

#ifndef RCMADP_CONFIG_HPP
#define RCMADP_CONFIG_HPP

#include "../core/config.hpp"
#include "../utils/file_utils.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <iomanip>
#include <queue>
#include <set>
#include <map>
#include <mutex>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Maximum customers for GPU kernel (static allocation)
#define RCMADP_MAX_CUSTOMERS 256
#define RCMADP_MAX_AGENTS 16

// GPU kernel for RCMADP fitness evaluation
template<typename T>
__global__ void rcmadp_fitness_kernel(
    T* population,           // [pop_size * chrom_len] - all chromosomes
    T* fitness,              // [pop_size] - output fitness values
    T* travel_matrix,        // [n+1 * n+1] - travel times (flattened)
    T* proc_times,           // [n+1] - processing times
    int pop_size,
    int chrom_len,
    int num_customers,
    int num_agents,
    int resources_per_agent,
    T penalty_weight
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    // Get chromosome pointer
    T* chrom = &population[idx * chrom_len];
    int n = num_customers;
    int matrix_dim = n + 1;

    // Chromosome layout: [dropoff_keys | pickup_keys | agent_hints]
    T* dropoff_keys = &chrom[0];
    T* pickup_keys = &chrom[n];
    // T* agent_hints = &chrom[2 * n];  // Not used in simplified kernel

    // Local arrays for sorting (simple insertion sort)
    int dropoff_order[RCMADP_MAX_CUSTOMERS];
    int pickup_order[RCMADP_MAX_CUSTOMERS];
    T dropoff_vals[RCMADP_MAX_CUSTOMERS];
    T pickup_vals[RCMADP_MAX_CUSTOMERS];

    // Initialize orders
    for (int i = 0; i < n && i < RCMADP_MAX_CUSTOMERS; i++) {
        dropoff_order[i] = i + 1;  // 1-indexed customers
        pickup_order[i] = i + 1;
        dropoff_vals[i] = dropoff_keys[i];
        pickup_vals[i] = pickup_keys[i];
    }

    // Sort dropoff order by keys (insertion sort)
    for (int i = 1; i < n && i < RCMADP_MAX_CUSTOMERS; i++) {
        int j = i;
        while (j > 0 && dropoff_vals[j-1] > dropoff_vals[j]) {
            // Swap
            T tv = dropoff_vals[j]; dropoff_vals[j] = dropoff_vals[j-1]; dropoff_vals[j-1] = tv;
            int to = dropoff_order[j]; dropoff_order[j] = dropoff_order[j-1]; dropoff_order[j-1] = to;
            j--;
        }
    }

    // Sort pickup order by keys
    for (int i = 1; i < n && i < RCMADP_MAX_CUSTOMERS; i++) {
        int j = i;
        while (j > 0 && pickup_vals[j-1] > pickup_vals[j]) {
            T tv = pickup_vals[j]; pickup_vals[j] = pickup_vals[j-1]; pickup_vals[j-1] = tv;
            int to = pickup_order[j]; pickup_order[j] = pickup_order[j-1]; pickup_order[j-1] = to;
            j--;
        }
    }

    // Agent state
    T agent_time[RCMADP_MAX_AGENTS];
    int agent_loc[RCMADP_MAX_AGENTS];
    int agent_res[RCMADP_MAX_AGENTS];

    for (int a = 0; a < num_agents && a < RCMADP_MAX_AGENTS; a++) {
        agent_time[a] = 0;
        agent_loc[a] = 0;  // Depot
        agent_res[a] = resources_per_agent;
    }

    // Track dropoff times and pickup status
    T dropoff_time[RCMADP_MAX_CUSTOMERS + 1];
    bool picked_up[RCMADP_MAX_CUSTOMERS + 1];
    bool dropped[RCMADP_MAX_CUSTOMERS + 1];

    for (int i = 0; i <= n && i <= RCMADP_MAX_CUSTOMERS; i++) {
        dropoff_time[i] = -1;
        picked_up[i] = false;
        dropped[i] = false;
    }

    T total_travel = 0;
    int unserviced = 0;

    // Phase 1: Perform dropoffs
    for (int i = 0; i < n && i < RCMADP_MAX_CUSTOMERS; i++) {
        int customer = dropoff_order[i];

        // Find best agent with resources
        int best_agent = -1;
        T best_arrival = 1e30;

        for (int a = 0; a < num_agents && a < RCMADP_MAX_AGENTS; a++) {
            if (agent_res[a] > 0) {
                int from = agent_loc[a];
                T travel = travel_matrix[from * matrix_dim + customer];
                T arrival = agent_time[a] + travel;
                if (arrival < best_arrival) {
                    best_arrival = arrival;
                    best_agent = a;
                }
            }
        }

        if (best_agent >= 0) {
            int from = agent_loc[best_agent];
            T travel = travel_matrix[from * matrix_dim + customer];
            total_travel += travel;
            agent_time[best_agent] += travel;
            agent_loc[best_agent] = customer;
            agent_res[best_agent]--;
            dropoff_time[customer] = agent_time[best_agent];
            dropped[customer] = true;
        }
    }

    // Phase 2: Perform pickups
    // Simple greedy: process in pickup_order, pick up earliest ready
    for (int iter = 0; iter < n * 2 && iter < RCMADP_MAX_CUSTOMERS * 2; iter++) {
        int best_customer = -1;
        int best_agent = -1;
        T best_time = 1e30;

        // Find ready customer with earliest pickup time
        for (int i = 0; i < n && i < RCMADP_MAX_CUSTOMERS; i++) {
            int customer = pickup_order[i];
            if (dropped[customer] && !picked_up[customer]) {
                T ready_time = dropoff_time[customer] + proc_times[customer];

                for (int a = 0; a < num_agents && a < RCMADP_MAX_AGENTS; a++) {
                    if (agent_res[a] < resources_per_agent) {
                        int from = agent_loc[a];
                        T travel = travel_matrix[from * matrix_dim + customer];
                        T arrival = agent_time[a] + travel;
                        if (arrival < ready_time) arrival = ready_time;

                        if (arrival < best_time) {
                            best_time = arrival;
                            best_agent = a;
                            best_customer = customer;
                        }
                    }
                }
            }
        }

        if (best_customer < 0) break;  // No more pickups available

        // Perform pickup
        int from = agent_loc[best_agent];
        T travel = travel_matrix[from * matrix_dim + best_customer];
        T ready_time = dropoff_time[best_customer] + proc_times[best_customer];
        T arrival = agent_time[best_agent] + travel;
        if (arrival < ready_time) arrival = ready_time;

        total_travel += travel;
        agent_time[best_agent] = arrival;
        agent_loc[best_agent] = best_customer;
        agent_res[best_agent]++;
        picked_up[best_customer] = true;
    }

    // Phase 3: Return to depot
    for (int a = 0; a < num_agents && a < RCMADP_MAX_AGENTS; a++) {
        if (agent_loc[a] != 0) {
            T return_travel = travel_matrix[agent_loc[a] * matrix_dim + 0];
            total_travel += return_travel;
        }
    }

    // Count unserviced
    for (int c = 1; c <= n && c <= RCMADP_MAX_CUSTOMERS; c++) {
        if (!picked_up[c]) {
            unserviced++;
        }
    }

    // Fitness = total travel + penalty for unserviced
    fitness[idx] = total_travel + unserviced * penalty_weight;
}

template<typename T>
class RCMADPConfig : public BRKGAConfig<T> {
public:
    // Problem parameters
    int num_agents;              // Number of agents/vehicles
    int resources_per_agent;     // Max resources each agent can carry
    int total_resources;         // Total available resources (may be < customers)
    T depot_return_penalty;      // Penalty for not returning to depot

private:
    // Customer/location data
    struct Location {
        int id;
        T x, y;
        T processing_time;       // Time resource must stay at location
    };

    std::vector<Location> locations;  // Index 0 = depot, 1..n = customers
    int num_customers;
    std::string instance_name;

    // Travel time matrix (or distance matrix)
    std::vector<std::vector<T>> travel_times;

    // Processing times per customer
    std::vector<T> processing_times;

    // GPU-specific members (per-GPU storage)
    std::map<int, T*> d_travel_matrices;   // device_id -> travel matrix
    std::map<int, T*> d_proc_times;        // device_id -> processing times
    bool gpu_available;
    mutable std::mutex gpu_mutex;

    // Solution representation
    struct AgentSchedule {
        int agent_id;
        std::vector<std::pair<int, T>> dropoffs;   // (customer_id, dropoff_time)
        std::vector<std::pair<int, T>> pickups;    // (customer_id, pickup_time)
        T total_travel_time;
        T finish_time;
        int resources_carried;                      // Current resources on agent
    };

    struct Solution {
        std::vector<AgentSchedule> agent_schedules;
        T total_travel_cost;
        T makespan;                                 // Latest finish time
        int unserviced_customers;
        int constraint_violations;
        std::vector<int> unserviced_list;
    };

public:
    RCMADPConfig(int agents = 3, int resources = 2)
        : BRKGAConfig<T>({1, 1, 1}),  // Will be updated after loading
          num_agents(agents),
          resources_per_agent(resources),
          total_resources(agents * resources),
          depot_return_penalty(1000.0),
          gpu_available(false) {
        check_gpu_availability();
    }

    ~RCMADPConfig() {
        cleanup_all_gpu_memory();
    }

    // GPU availability check
    void check_gpu_availability() {
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        gpu_available = (error == cudaSuccess && device_count > 0);
    }

    // GPU memory cleanup
    void cleanup_all_gpu_memory() {
        std::lock_guard<std::mutex> lock(gpu_mutex);
        for (auto& [device_id, ptr] : d_travel_matrices) {
            if (ptr) {
                cudaSetDevice(device_id);
                cudaFree(ptr);
            }
        }
        for (auto& [device_id, ptr] : d_proc_times) {
            if (ptr) {
                cudaSetDevice(device_id);
                cudaFree(ptr);
            }
        }
        d_travel_matrices.clear();
        d_proc_times.clear();
    }

    // GPU memory allocation for a specific device
    void ensure_gpu_memory(int device_id) {
        std::lock_guard<std::mutex> lock(gpu_mutex);

        if (d_travel_matrices.count(device_id) > 0) return;

        cudaSetDevice(device_id);

        int matrix_size = (num_customers + 1) * (num_customers + 1);
        int proc_size = num_customers + 1;

        // Allocate travel matrix
        T* d_travel = nullptr;
        cudaError_t err1 = cudaMalloc(&d_travel, matrix_size * sizeof(T));

        // Allocate processing times
        T* d_proc = nullptr;
        cudaError_t err2 = cudaMalloc(&d_proc, proc_size * sizeof(T));

        if (err1 != cudaSuccess || err2 != cudaSuccess) {
            if (d_travel) cudaFree(d_travel);
            if (d_proc) cudaFree(d_proc);
            return;
        }

        // Flatten and copy travel matrix
        std::vector<T> flat_travel(matrix_size);
        for (int i = 0; i <= num_customers; i++) {
            for (int j = 0; j <= num_customers; j++) {
                flat_travel[i * (num_customers + 1) + j] = travel_times[i][j];
            }
        }
        cudaMemcpy(d_travel, flat_travel.data(), matrix_size * sizeof(T), cudaMemcpyHostToDevice);

        // Copy processing times
        cudaMemcpy(d_proc, processing_times.data(), proc_size * sizeof(T), cudaMemcpyHostToDevice);

        d_travel_matrices[device_id] = d_travel;
        d_proc_times[device_id] = d_proc;
    }

    // GPU evaluation interface
    bool has_gpu_evaluation() const override { return gpu_available; }

    // GPU population evaluation
    void evaluate_population_gpu(T* d_population, T* d_fitness,
                                 int pop_size, int chrom_len) override {
        if (!gpu_available) return;

        int device_id;
        cudaGetDevice(&device_id);

        ensure_gpu_memory(device_id);
        cudaSetDevice(device_id);

        T* d_travel = nullptr;
        T* d_proc = nullptr;
        {
            std::lock_guard<std::mutex> lock(gpu_mutex);
            d_travel = d_travel_matrices[device_id];
            d_proc = d_proc_times[device_id];
        }

        if (!d_travel || !d_proc) {
            std::cerr << "RCMADP GPU: travel or proc times not allocated on device " << device_id << std::endl;
            return;
        }

        dim3 block(this->threads_per_block);
        dim3 grid((pop_size + block.x - 1) / block.x);

        rcmadp_fitness_kernel<<<grid, block>>>(
            d_population, d_fitness, d_travel, d_proc,
            pop_size, chrom_len, num_customers, num_agents,
            resources_per_agent, depot_return_penalty
        );

        cudaError_t error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            std::cerr << "RCMADP GPU kernel failed on device " << device_id
                      << ": " << cudaGetErrorString(error) << std::endl;
        }
    }

    // Getters
    int get_num_customers() const { return num_customers; }
    int get_num_agents() const { return num_agents; }
    int get_resources_per_agent() const { return resources_per_agent; }
    const std::string& get_instance_name() const { return instance_name; }
    const std::vector<std::vector<T>>& get_travel_times() const { return travel_times; }
    const std::vector<T>& get_processing_times() const { return processing_times; }

    // Load from TSP file + processing times file
    static std::unique_ptr<RCMADPConfig<T>> load_from_files(
            const std::string& tsp_file,
            const std::string& processing_times_file,
            int num_agents = 3,
            int resources_per_agent = 2) {

        auto config = std::make_unique<RCMADPConfig<T>>(num_agents, resources_per_agent);

        // Load TSP file (locations)
        config->load_tsp_file(tsp_file);

        // Load processing times
        config->load_processing_times(processing_times_file);

        // Setup BRKGA parameters
        config->setup_brkga_params();

        return config;
    }

    // Load from TSPJ format (_TT.csv and _JT.csv paired files)
    // This matches the existing berlin52_TSPJ format
    static std::unique_ptr<RCMADPConfig<T>> load_from_tspj(
            const std::string& filename,
            int num_agents = 3,
            int resources_per_agent = 2) {

        auto config = std::make_unique<RCMADPConfig<T>>(num_agents, resources_per_agent);

        std::string basename = FileUtils::get_basename(filename);
        std::string directory = FileUtils::get_directory(filename);

        std::string tt_file, jt_file;

        // Detect file naming convention
        if (basename.find("_TT") != std::string::npos) {
            tt_file = filename;
            jt_file = directory + "/" + basename.substr(0, basename.find("_TT")) + "_JT.csv";
            config->instance_name = basename.substr(0, basename.find("_TT"));
        } else if (basename.find("_JT") != std::string::npos) {
            jt_file = filename;
            tt_file = directory + "/" + basename.substr(0, basename.find("_JT")) + "_TT.csv";
            config->instance_name = basename.substr(0, basename.find("_JT"));
        } else {
            // Assume it's a base name, try both conventions
            tt_file = filename + "_TT.csv";
            jt_file = filename + "_JT.csv";
            if (!FileUtils::file_exists(tt_file)) {
                tt_file = directory + "/" + basename + "_TT.csv";
                jt_file = directory + "/" + basename + "_JT.csv";
            }
            config->instance_name = basename;
        }

        if (!FileUtils::file_exists(tt_file)) {
            throw std::runtime_error("Travel times file not found: " + tt_file);
        }
        if (!FileUtils::file_exists(jt_file)) {
            throw std::runtime_error("Job times file not found: " + jt_file);
        }

        // Load travel times matrix
        config->travel_times = load_csv_matrix(tt_file);
        config->num_customers = config->travel_times.size() - 1;  // Exclude depot

        // Load job times matrix and extract processing times
        auto job_matrix = load_csv_matrix(jt_file);
        config->extract_processing_times_from_job_matrix(job_matrix);

        // Create dummy locations
        config->locations.resize(config->travel_times.size());
        for (size_t i = 0; i < config->locations.size(); i++) {
            config->locations[i] = {static_cast<int>(i), T(0), T(0), config->processing_times[i]};
        }

        // Setup BRKGA parameters
        config->setup_brkga_params();

        return config;
    }

private:
    // Load CSV matrix (shared helper)
    static std::vector<std::vector<T>> load_csv_matrix(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open CSV file: " + filename);
        }

        std::vector<std::vector<T>> matrix;
        std::string line;

        while (std::getline(file, line)) {
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
                    } catch (const std::exception&) {
                        row.push_back(T(0));
                    }
                }
            }

            if (!row.empty()) {
                matrix.push_back(row);
            }
        }

        file.close();
        return matrix;
    }

    // Extract processing times from job matrix
    // Uses diagonal values or average of each row as processing time
    void extract_processing_times_from_job_matrix(const std::vector<std::vector<T>>& job_matrix) {
        processing_times.clear();
        processing_times.push_back(T(0));  // Depot has 0 processing time

        for (size_t i = 1; i < job_matrix.size(); i++) {
            // Option 1: Use diagonal value
            T proc_time = (i < job_matrix[i].size()) ? job_matrix[i][i] : T(0);

            // If diagonal is 0, use average of non-zero values in row
            if (proc_time <= 0) {
                T sum = 0;
                int count = 0;
                for (size_t j = 1; j < job_matrix[i].size(); j++) {
                    if (job_matrix[i][j] > 0) {
                        sum += job_matrix[i][j];
                        count++;
                    }
                }
                proc_time = (count > 0) ? (sum / count) : T(30);  // Default 30 if all zeros
            }

            // Scale to reasonable processing time (divide by 100 if values seem like costs)
            if (proc_time > 1000) {
                proc_time = proc_time / 100;
            }

            processing_times.push_back(proc_time);
        }
    }

public:

    // Alternative: Load from travel time matrix + processing times
    static std::unique_ptr<RCMADPConfig<T>> load_from_matrices(
            const std::string& travel_matrix_file,
            const std::string& processing_times_file,
            int num_agents = 3,
            int resources_per_agent = 2) {

        auto config = std::make_unique<RCMADPConfig<T>>(num_agents, resources_per_agent);

        // Load travel time matrix directly
        config->load_travel_matrix(travel_matrix_file);

        // Load processing times
        config->load_processing_times(processing_times_file);

        // Setup BRKGA parameters
        config->setup_brkga_params();

        return config;
    }

    void set_parameters(int agents, int resources) {
        num_agents = agents;
        resources_per_agent = resources;
        total_resources = agents * resources;
    }

private:
    void load_tsp_file(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open TSP file: " + filename);
        }

        std::string line;
        bool in_coord_section = false;
        bool in_edge_weight_section = false;
        int dimension = 0;
        std::string edge_weight_type = "EUC_2D";
        std::string edge_weight_format = "";

        // Parse header
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;

            if (line.find("NAME") != std::string::npos) {
                size_t pos = line.find(":");
                if (pos != std::string::npos) {
                    instance_name = line.substr(pos + 1);
                    instance_name.erase(0, instance_name.find_first_not_of(" \t"));
                    instance_name.erase(instance_name.find_last_not_of(" \t\r\n") + 1);
                }
            }
            else if (line.find("DIMENSION") != std::string::npos) {
                std::istringstream iss(line);
                std::string dummy;
                char colon;
                iss >> dummy >> colon >> dimension;
                num_customers = dimension - 1;  // Subtract depot
            }
            else if (line.find("EDGE_WEIGHT_TYPE") != std::string::npos) {
                size_t pos = line.find(":");
                if (pos != std::string::npos) {
                    edge_weight_type = line.substr(pos + 1);
                    edge_weight_type.erase(0, edge_weight_type.find_first_not_of(" \t"));
                    edge_weight_type.erase(edge_weight_type.find_last_not_of(" \t\r\n") + 1);
                }
            }
            else if (line.find("EDGE_WEIGHT_FORMAT") != std::string::npos) {
                size_t pos = line.find(":");
                if (pos != std::string::npos) {
                    edge_weight_format = line.substr(pos + 1);
                    edge_weight_format.erase(0, edge_weight_format.find_first_not_of(" \t"));
                }
            }
            else if (line.find("NODE_COORD_SECTION") != std::string::npos) {
                in_coord_section = true;
                locations.resize(dimension);
                break;
            }
            else if (line.find("EDGE_WEIGHT_SECTION") != std::string::npos) {
                in_edge_weight_section = true;
                break;
            }
        }

        if (in_coord_section) {
            // Read coordinates
            for (int i = 0; i < dimension; i++) {
                int id;
                T x, y;
                file >> id >> x >> y;
                locations[id - 1] = {id - 1, x, y, T(0)};
            }

            // Build distance matrix from coordinates
            build_distance_matrix(edge_weight_type);
        }
        else if (in_edge_weight_section) {
            // Read explicit distance matrix
            travel_times.resize(dimension, std::vector<T>(dimension));

            if (edge_weight_format.find("FULL_MATRIX") != std::string::npos) {
                for (int i = 0; i < dimension; i++) {
                    for (int j = 0; j < dimension; j++) {
                        file >> travel_times[i][j];
                    }
                }
            }
            else if (edge_weight_format.find("UPPER_ROW") != std::string::npos ||
                     edge_weight_format.find("LOWER_ROW") != std::string::npos) {
                // Triangular matrix
                for (int i = 0; i < dimension; i++) {
                    travel_times[i][i] = 0;
                    for (int j = i + 1; j < dimension; j++) {
                        file >> travel_times[i][j];
                        travel_times[j][i] = travel_times[i][j];
                    }
                }
            }

            // Create dummy locations
            locations.resize(dimension);
            for (int i = 0; i < dimension; i++) {
                locations[i] = {i, T(0), T(0), T(0)};
            }
        }

        file.close();
    }

    void build_distance_matrix(const std::string& edge_weight_type) {
        int n = locations.size();
        travel_times.resize(n, std::vector<T>(n));

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == j) {
                    travel_times[i][j] = T(0);
                } else {
                    T dx = locations[i].x - locations[j].x;
                    T dy = locations[i].y - locations[j].y;
                    T dist = std::sqrt(dx * dx + dy * dy);

                    if (edge_weight_type == "EUC_2D") {
                        travel_times[i][j] = std::round(dist);
                    } else if (edge_weight_type == "CEIL_2D") {
                        travel_times[i][j] = std::ceil(dist);
                    } else {
                        travel_times[i][j] = dist;
                    }
                }
            }
        }
    }

    void load_travel_matrix(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open travel matrix file: " + filename);
        }

        // Try to detect format
        std::string line;
        std::getline(file, line);
        file.seekg(0);

        // Count elements in first line to determine if it's dimension or data
        std::istringstream iss(line);
        std::vector<T> first_row;
        T val;
        while (iss >> val) {
            first_row.push_back(val);
        }

        int dimension;
        if (first_row.size() == 1) {
            // First line is dimension
            dimension = static_cast<int>(first_row[0]);
            file >> dimension;  // Re-read
        } else {
            // First line is data row
            dimension = first_row.size();
            file.seekg(0);
        }

        num_customers = dimension - 1;
        travel_times.resize(dimension, std::vector<T>(dimension));

        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                file >> travel_times[i][j];
            }
        }

        // Create dummy locations
        locations.resize(dimension);
        for (int i = 0; i < dimension; i++) {
            locations[i] = {i, T(0), T(0), T(0)};
        }

        file.close();

        // Extract instance name from filename
        instance_name = FileUtils::get_basename(filename);
        size_t dot_pos = instance_name.find_last_of('.');
        if (dot_pos != std::string::npos) {
            instance_name = instance_name.substr(0, dot_pos);
        }
    }

    void load_processing_times(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open processing times file: " + filename);
        }

        processing_times.clear();
        processing_times.push_back(T(0));  // Depot has 0 processing time

        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;

            std::istringstream iss(line);
            T time;
            while (iss >> time) {
                processing_times.push_back(time);
            }
        }

        file.close();

        // If processing times is a matrix (e.g., job durations per customer),
        // take diagonal or first column as the processing time
        if (processing_times.size() == 1) {
            // Try CSV format
            file.open(filename);
            processing_times.clear();
            processing_times.push_back(T(0));  // Depot

            while (std::getline(file, line)) {
                if (line.empty() || line[0] == '#') continue;

                std::istringstream iss(line);
                std::string cell;
                bool first = true;
                while (std::getline(iss, cell, ',')) {
                    if (first) {
                        cell.erase(0, cell.find_first_not_of(" \t"));
                        cell.erase(cell.find_last_not_of(" \t\r\n") + 1);
                        if (!cell.empty()) {
                            processing_times.push_back(static_cast<T>(std::stod(cell)));
                        }
                        first = false;
                        break;  // Just take first column
                    }
                }
            }
            file.close();
        }

        // Ensure we have processing times for all customers
        while (processing_times.size() < static_cast<size_t>(num_customers + 1)) {
            // Default processing time if not enough data
            processing_times.push_back(T(10));
        }

        // Update location processing times
        for (size_t i = 0; i < locations.size() && i < processing_times.size(); i++) {
            locations[i].processing_time = processing_times[i];
        }
    }

    void setup_brkga_params() {
        // Chromosome: [dropoff_sequence | pickup_sequence | agent_hints]
        // Each component has num_customers genes
        this->component_lengths = {num_customers, num_customers, num_customers};
        this->num_components = 3;

        // Scale population based on problem complexity
        int complexity = num_customers * num_agents;
        if (complexity <= 50) {
            this->population_size = 500;
            this->max_generations = 1000;
        } else if (complexity <= 150) {
            this->population_size = 800;
            this->max_generations = 1500;
        } else if (complexity <= 300) {
            this->population_size = 1200;
            this->max_generations = 2000;
        } else {
            this->population_size = 2000;
            this->max_generations = 3000;
        }

        this->elite_size = this->population_size / 5;
        this->mutant_size = this->population_size / 10;
        this->elite_prob = 0.7;

        this->update_cuda_grid_size();

        // Setup fitness function
        this->fitness_function = [this](const Individual<T>& ind) {
            return calculate_fitness(ind);
        };

        // Setup decoder
        this->decoder = [this](const Individual<T>& ind) {
            return decode_to_solution_vector(ind);
        };

        // Minimization
        this->comparator = [](T a, T b) { return a < b; };
    }

public:
    // Main fitness calculation
    T calculate_fitness(const Individual<T>& individual) const {
        Solution sol = decode_solution(individual);

        // Primary objective: total travel cost
        T fitness = sol.total_travel_cost;

        // Penalties
        T unserviced_penalty = sol.unserviced_customers * T(10000);
        T violation_penalty = sol.constraint_violations * T(5000);

        // Soft penalty for imbalanced workloads
        T max_travel = 0, min_travel = std::numeric_limits<T>::max();
        for (const auto& schedule : sol.agent_schedules) {
            if (schedule.total_travel_time > 0) {
                max_travel = std::max(max_travel, schedule.total_travel_time);
                min_travel = std::min(min_travel, schedule.total_travel_time);
            }
        }
        T balance_penalty = (max_travel - min_travel) * T(0.1);

        return fitness + unserviced_penalty + violation_penalty + balance_penalty;
    }

    // Decode chromosome into full solution
    Solution decode_solution(const Individual<T>& individual) const {
        const auto& dropoff_chr = individual.get_component(0);
        const auto& pickup_chr = individual.get_component(1);
        const auto& agent_chr = individual.get_component(2);

        Solution sol;
        sol.total_travel_cost = 0;
        sol.makespan = 0;
        sol.unserviced_customers = 0;
        sol.constraint_violations = 0;

        // Initialize agent schedules
        sol.agent_schedules.resize(num_agents);
        for (int a = 0; a < num_agents; a++) {
            sol.agent_schedules[a].agent_id = a;
            sol.agent_schedules[a].total_travel_time = 0;
            sol.agent_schedules[a].finish_time = 0;
            sol.agent_schedules[a].resources_carried = resources_per_agent;
        }

        // Decode dropoff sequence (sorted by chromosome values)
        std::vector<std::pair<T, int>> dropoff_order;
        for (int i = 0; i < num_customers; i++) {
            dropoff_order.push_back({dropoff_chr[i], i + 1});  // Customer IDs are 1-indexed
        }
        std::sort(dropoff_order.begin(), dropoff_order.end());

        // Decode pickup sequence
        std::vector<std::pair<T, int>> pickup_order;
        for (int i = 0; i < num_customers; i++) {
            pickup_order.push_back({pickup_chr[i], i + 1});
        }
        std::sort(pickup_order.begin(), pickup_order.end());

        // Track dropoff times for pickup constraints
        std::vector<T> dropoff_times(num_customers + 1, T(-1));
        std::vector<int> dropoff_agent(num_customers + 1, -1);
        std::vector<bool> picked_up(num_customers + 1, false);

        // Track agent states
        std::vector<T> agent_time(num_agents, T(0));
        std::vector<int> agent_location(num_agents, 0);  // All start at depot
        std::vector<int> agent_resources(num_agents, resources_per_agent);

        // Phase 1: Perform dropoffs
        for (const auto& [key, customer] : dropoff_order) {
            // Find best agent for this dropoff
            int best_agent = -1;
            T best_arrival = std::numeric_limits<T>::max();

            // Use agent hint to bias selection
            int hinted_agent = static_cast<int>(agent_chr[customer - 1] * num_agents);
            hinted_agent = std::min(hinted_agent, num_agents - 1);

            // Check hinted agent first
            std::vector<int> agent_order;
            agent_order.push_back(hinted_agent);
            for (int a = 0; a < num_agents; a++) {
                if (a != hinted_agent) agent_order.push_back(a);
            }

            for (int a : agent_order) {
                if (agent_resources[a] > 0) {
                    T arrival = agent_time[a] + travel_times[agent_location[a]][customer];
                    if (arrival < best_arrival) {
                        best_arrival = arrival;
                        best_agent = a;
                    }
                }
            }

            if (best_agent >= 0) {
                // Perform dropoff
                T travel = travel_times[agent_location[best_agent]][customer];
                agent_time[best_agent] += travel;
                agent_location[best_agent] = customer;
                agent_resources[best_agent]--;

                dropoff_times[customer] = agent_time[best_agent];
                dropoff_agent[customer] = best_agent;

                sol.agent_schedules[best_agent].dropoffs.push_back({customer, agent_time[best_agent]});
                sol.agent_schedules[best_agent].total_travel_time += travel;
            } else {
                // No agent available - need to do pickups first
                // This is handled in phase 2
                sol.unserviced_list.push_back(customer);
            }
        }

        // Phase 2: Perform pickups (respecting processing time constraints)
        // Use a priority queue based on when pickups become available
        using PickupEvent = std::pair<T, int>;  // (ready_time, customer)
        std::priority_queue<PickupEvent, std::vector<PickupEvent>, std::greater<>> pickup_queue;

        // Add all customers with completed dropoffs to queue
        for (int c = 1; c <= num_customers; c++) {
            if (dropoff_times[c] >= 0) {
                T ready_time = dropoff_times[c] + processing_times[c];
                pickup_queue.push({ready_time, c});
            }
        }

        // Process pickups in order of pickup chromosome (but respecting ready times)
        std::set<int> pending_pickups;
        for (const auto& [key, customer] : pickup_order) {
            if (dropoff_times[customer] >= 0) {
                pending_pickups.insert(customer);
            }
        }

        while (!pending_pickups.empty()) {
            // Find best customer to pick up
            int best_customer = -1;
            int best_agent = -1;
            T best_time = std::numeric_limits<T>::max();

            for (int customer : pending_pickups) {
                T ready_time = dropoff_times[customer] + processing_times[customer];

                for (int a = 0; a < num_agents; a++) {
                    if (agent_resources[a] < resources_per_agent) {
                        T arrival = std::max(agent_time[a] + travel_times[agent_location[a]][customer], ready_time);
                        if (arrival < best_time) {
                            best_time = arrival;
                            best_agent = a;
                            best_customer = customer;
                        }
                    }
                }
            }

            if (best_customer >= 0 && best_agent >= 0) {
                // Perform pickup
                T travel = travel_times[agent_location[best_agent]][best_customer];
                T ready_time = dropoff_times[best_customer] + processing_times[best_customer];
                T arrival = agent_time[best_agent] + travel;

                // Wait if arriving before ready
                T wait_time = std::max(T(0), ready_time - arrival);
                agent_time[best_agent] = arrival + wait_time;
                agent_location[best_agent] = best_customer;
                agent_resources[best_agent]++;

                picked_up[best_customer] = true;

                sol.agent_schedules[best_agent].pickups.push_back({best_customer, agent_time[best_agent]});
                sol.agent_schedules[best_agent].total_travel_time += travel;

                pending_pickups.erase(best_customer);

                // Check if this frees up resources for pending dropoffs
                if (!sol.unserviced_list.empty() && agent_resources[best_agent] > 0) {
                    int unserviced_customer = sol.unserviced_list.back();
                    sol.unserviced_list.pop_back();

                    // Perform the pending dropoff
                    T d_travel = travel_times[agent_location[best_agent]][unserviced_customer];
                    agent_time[best_agent] += d_travel;
                    agent_location[best_agent] = unserviced_customer;
                    agent_resources[best_agent]--;

                    dropoff_times[unserviced_customer] = agent_time[best_agent];
                    dropoff_agent[unserviced_customer] = best_agent;

                    sol.agent_schedules[best_agent].dropoffs.push_back({unserviced_customer, agent_time[best_agent]});
                    sol.agent_schedules[best_agent].total_travel_time += d_travel;

                    // Add to pickup queue
                    T new_ready = dropoff_times[unserviced_customer] + processing_times[unserviced_customer];
                    pending_pickups.insert(unserviced_customer);
                }
            } else {
                // Deadlock or infeasible - mark remaining as unserviced
                for (int c : pending_pickups) {
                    if (!picked_up[c]) {
                        sol.constraint_violations++;
                    }
                }
                break;
            }
        }

        // Phase 3: Return all agents to depot
        for (int a = 0; a < num_agents; a++) {
            if (agent_location[a] != 0) {
                T return_travel = travel_times[agent_location[a]][0];
                agent_time[a] += return_travel;
                sol.agent_schedules[a].total_travel_time += return_travel;
                sol.agent_schedules[a].finish_time = agent_time[a];
            } else {
                sol.agent_schedules[a].finish_time = agent_time[a];
            }

            sol.total_travel_cost += sol.agent_schedules[a].total_travel_time;
            sol.makespan = std::max(sol.makespan, sol.agent_schedules[a].finish_time);
        }

        // Count unserviced customers
        sol.unserviced_customers = sol.unserviced_list.size();
        for (int c = 1; c <= num_customers; c++) {
            if (!picked_up[c] && dropoff_times[c] < 0) {
                sol.unserviced_customers++;
                sol.unserviced_list.push_back(c);
            }
        }

        return sol;
    }

    // Decode to vector format for external use
    std::vector<std::vector<T>> decode_to_solution_vector(const Individual<T>& individual) const {
        Solution sol = decode_solution(individual);

        std::vector<std::vector<T>> result;

        // Format: For each agent, output [dropoff_sequence..., -1, pickup_sequence...]
        for (const auto& schedule : sol.agent_schedules) {
            std::vector<T> agent_route;

            for (const auto& [customer, time] : schedule.dropoffs) {
                agent_route.push_back(static_cast<T>(customer));
            }
            agent_route.push_back(T(-1));  // Separator
            for (const auto& [customer, time] : schedule.pickups) {
                agent_route.push_back(static_cast<T>(customer));
            }

            result.push_back(agent_route);
        }

        return result;
    }

    // Validation
    bool validate_solution(const Individual<T>& individual) override {
        Solution sol = decode_solution(individual);

        // Check all customers are serviced
        if (sol.unserviced_customers > 0) return false;

        // Check constraint violations
        if (sol.constraint_violations > 0) return false;

        // Check resource constraints
        for (const auto& schedule : sol.agent_schedules) {
            int resources = resources_per_agent;
            size_t drop_idx = 0, pick_idx = 0;

            // Interleave dropoffs and pickups by time
            while (drop_idx < schedule.dropoffs.size() || pick_idx < schedule.pickups.size()) {
                T next_drop_time = (drop_idx < schedule.dropoffs.size()) ?
                    schedule.dropoffs[drop_idx].second : std::numeric_limits<T>::max();
                T next_pick_time = (pick_idx < schedule.pickups.size()) ?
                    schedule.pickups[pick_idx].second : std::numeric_limits<T>::max();

                if (next_drop_time <= next_pick_time && drop_idx < schedule.dropoffs.size()) {
                    resources--;
                    if (resources < 0) return false;
                    drop_idx++;
                } else if (pick_idx < schedule.pickups.size()) {
                    resources++;
                    if (resources > resources_per_agent) return false;
                    pick_idx++;
                }
            }
        }

        return true;
    }

    // Print solution
    void print_solution(const Individual<T>& individual) override {
        Solution sol = decode_solution(individual);

        std::cout << "\n=== RCMADP Solution ===" << std::endl;
        std::cout << "Total Travel Cost: " << std::fixed << std::setprecision(2)
                  << sol.total_travel_cost << std::endl;
        std::cout << "Makespan: " << sol.makespan << std::endl;
        std::cout << "Unserviced Customers: " << sol.unserviced_customers << std::endl;

        for (const auto& schedule : sol.agent_schedules) {
            std::cout << "\nAgent " << schedule.agent_id << ":" << std::endl;
            std::cout << "  Travel Time: " << schedule.total_travel_time << std::endl;

            std::cout << "  Dropoffs: ";
            for (const auto& [customer, time] : schedule.dropoffs) {
                std::cout << customer << "@" << std::setprecision(1) << time << " ";
            }
            std::cout << std::endl;

            std::cout << "  Pickups: ";
            for (const auto& [customer, time] : schedule.pickups) {
                std::cout << customer << "@" << std::setprecision(1) << time << " ";
            }
            std::cout << std::endl;
        }

        if (!sol.unserviced_list.empty()) {
            std::cout << "\nUnserviced: ";
            for (int c : sol.unserviced_list) {
                std::cout << c << " ";
            }
            std::cout << std::endl;
        }
    }

    // Export solution to file
    void export_solution(const Individual<T>& individual, const std::string& filename) override {
        Solution sol = decode_solution(individual);

        FileUtils::ensure_directory(FileUtils::get_directory(filename));

        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot create solution file: " + filename);
        }

        file << "RCMADP Solution\n";
        file << "Instance: " << instance_name << "\n";
        file << "Agents: " << num_agents << "\n";
        file << "Resources per Agent: " << resources_per_agent << "\n";
        file << "Customers: " << num_customers << "\n";
        file << "\n";
        file << "Total Travel Cost: " << std::fixed << std::setprecision(2)
             << sol.total_travel_cost << "\n";
        file << "Makespan: " << sol.makespan << "\n";
        file << "Fitness: " << individual.fitness << "\n";
        file << "\n";

        for (const auto& schedule : sol.agent_schedules) {
            file << "Agent " << schedule.agent_id << ":\n";
            file << "  Route Travel Time: " << schedule.total_travel_time << "\n";
            file << "  Finish Time: " << schedule.finish_time << "\n";

            file << "  Dropoffs:\n";
            for (const auto& [customer, time] : schedule.dropoffs) {
                file << "    Customer " << customer
                     << " at time " << std::setprecision(2) << time
                     << " (processing: " << processing_times[customer] << ")\n";
            }

            file << "  Pickups:\n";
            for (const auto& [customer, time] : schedule.pickups) {
                file << "    Customer " << customer
                     << " at time " << std::setprecision(2) << time << "\n";
            }
            file << "\n";
        }

        file.close();
    }

    // Print instance info
    void print_instance_info() const {
        std::cout << "\n=== RCMADP Instance ===" << std::endl;
        std::cout << "Instance: " << instance_name << std::endl;
        std::cout << "Customers: " << num_customers << std::endl;
        std::cout << "Agents: " << num_agents << std::endl;
        std::cout << "Resources per Agent: " << resources_per_agent << std::endl;
        std::cout << "Total Resources: " << total_resources << std::endl;
        std::cout << "Chromosome Length: " << (3 * num_customers) << std::endl;
        std::cout << "  - Dropoff sequence: " << num_customers << " genes" << std::endl;
        std::cout << "  - Pickup sequence: " << num_customers << " genes" << std::endl;
        std::cout << "  - Agent hints: " << num_customers << " genes" << std::endl;

        // Print processing time statistics
        if (!processing_times.empty()) {
            T min_proc = *std::min_element(processing_times.begin() + 1, processing_times.end());
            T max_proc = *std::max_element(processing_times.begin() + 1, processing_times.end());
            T avg_proc = std::accumulate(processing_times.begin() + 1, processing_times.end(), T(0)) / num_customers;
            std::cout << "Processing Times: min=" << min_proc << ", max=" << max_proc
                      << ", avg=" << std::setprecision(1) << avg_proc << std::endl;
        }
        std::cout << "=========================" << std::endl;
    }

    // Configuration helpers
    static void configure_for_size(RCMADPConfig<T>* config, int num_customers, int num_agents) {
        int complexity = num_customers * num_agents;

        if (complexity <= 30) {
            config->population_size = 400;
            config->max_generations = 800;
        } else if (complexity <= 100) {
            config->population_size = 800;
            config->max_generations = 1500;
        } else if (complexity <= 300) {
            config->population_size = 1500;
            config->max_generations = 2500;
        } else {
            config->population_size = 2500;
            config->max_generations = 4000;
        }

        config->elite_size = config->population_size / 5;
        config->mutant_size = config->population_size / 8;
        config->elite_prob = 0.7;
        config->update_cuda_grid_size();
    }
};

#endif // RCMADP_CONFIG_HPP
