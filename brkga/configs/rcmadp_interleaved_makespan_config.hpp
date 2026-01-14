// configs/rcmadp_interleaved_makespan_config.hpp
// Resource-Constrained Multi-Agent Drop-and-Pick Problem (RCMADP) - Interleaved Version
// MAKESPAN OPTIMIZATION VARIANT
//
// This version minimizes MAKESPAN (time when last agent finishes) instead of total travel cost.
//
// Problem Description:
// - Multiple agents with limited deployable resources
// - Each customer requires a resource dropped off, left to process for a fixed time,
//   then picked up from the same location
// - Dropoff and pickup may be performed by different agents
// - Resources are unavailable while deployed
// - Agents must respect travel times, resource capacity, and pickup-after-processing constraints
// - Objective: Minimize MAKESPAN (completion time of last agent)
//
// Key Difference from Standard RCMADP Interleaved:
// - Standard version optimizes total_travel_cost (sum of all agents' travel)
// - This version optimizes makespan (time when last agent finishes)
//
// Chromosome Structure (2 components):
// - Component 0: Operation priorities (2 * num_customers genes)
//   - Genes 0 to n-1: Priority keys for dropoffs of customers 1..n
//   - Genes n to 2n-1: Priority keys for pickups of customers 1..n
// - Component 1: Agent assignment hints (2 * num_customers genes)
//   - Same layout as component 0, hints which agent should perform each operation

#ifndef RCMADP_INTERLEAVED_MAKESPAN_CONFIG_HPP
#define RCMADP_INTERLEAVED_MAKESPAN_CONFIG_HPP

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
#define RCMADP_INTERLEAVED_MAKESPAN_MAX_CUSTOMERS 256
#define RCMADP_INTERLEAVED_MAKESPAN_MAX_AGENTS 16
#define RCMADP_INTERLEAVED_MAKESPAN_MAX_OPS (2 * RCMADP_INTERLEAVED_MAKESPAN_MAX_CUSTOMERS)

// Operation type enum (use unique namespace to avoid conflicts)
namespace makespan_opt {
    enum class OpType { DROPOFF = 0, PICKUP = 1 };
}

// GPU kernel for interleaved RCMADP fitness evaluation (MAKESPAN version)
template<typename T>
__global__ void rcmadp_interleaved_makespan_fitness_kernel(
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
    int num_ops = 2 * n;
    int matrix_dim = n + 1;

    // Chromosome layout: [op_priorities (2n) | agent_hints (2n)]
    T* op_priorities = &chrom[0];
    T* agent_hints = &chrom[num_ops];

    // Create operation list: (priority, op_index)
    // op_index 0..n-1 = dropoff customer (op_index+1)
    // op_index n..2n-1 = pickup customer (op_index-n+1)
    int op_order[RCMADP_INTERLEAVED_MAKESPAN_MAX_OPS];
    T op_vals[RCMADP_INTERLEAVED_MAKESPAN_MAX_OPS];

    for (int i = 0; i < num_ops && i < RCMADP_INTERLEAVED_MAKESPAN_MAX_OPS; i++) {
        op_order[i] = i;
        op_vals[i] = op_priorities[i];
    }

    // Sort operations by priority (insertion sort)
    for (int i = 1; i < num_ops && i < RCMADP_INTERLEAVED_MAKESPAN_MAX_OPS; i++) {
        int j = i;
        while (j > 0 && op_vals[j-1] > op_vals[j]) {
            T tv = op_vals[j]; op_vals[j] = op_vals[j-1]; op_vals[j-1] = tv;
            int to = op_order[j]; op_order[j] = op_order[j-1]; op_order[j-1] = to;
            j--;
        }
    }

    // Agent state
    T agent_time[RCMADP_INTERLEAVED_MAKESPAN_MAX_AGENTS];
    int agent_loc[RCMADP_INTERLEAVED_MAKESPAN_MAX_AGENTS];
    int agent_res[RCMADP_INTERLEAVED_MAKESPAN_MAX_AGENTS];

    for (int a = 0; a < num_agents && a < RCMADP_INTERLEAVED_MAKESPAN_MAX_AGENTS; a++) {
        agent_time[a] = 0;
        agent_loc[a] = 0;  // Depot
        agent_res[a] = resources_per_agent;
    }

    // Track customer states
    T dropoff_time[RCMADP_INTERLEAVED_MAKESPAN_MAX_CUSTOMERS + 1];
    bool dropped[RCMADP_INTERLEAVED_MAKESPAN_MAX_CUSTOMERS + 1];
    bool picked_up[RCMADP_INTERLEAVED_MAKESPAN_MAX_CUSTOMERS + 1];

    for (int i = 0; i <= n && i <= RCMADP_INTERLEAVED_MAKESPAN_MAX_CUSTOMERS; i++) {
        dropoff_time[i] = -1;
        dropped[i] = false;
        picked_up[i] = false;
    }

    T total_travel = 0;
    int completed_ops = 0;
    int constraint_violations = 0;

    // Track which operations are still pending
    bool op_done[RCMADP_INTERLEAVED_MAKESPAN_MAX_OPS];
    for (int i = 0; i < num_ops && i < RCMADP_INTERLEAVED_MAKESPAN_MAX_OPS; i++) {
        op_done[i] = false;
    }

    // Multiple passes to handle dependencies (pickup must wait for dropoff + processing)
    for (int pass = 0; pass < num_ops && completed_ops < num_ops; pass++) {
        bool made_progress = false;

        for (int i = 0; i < num_ops && i < RCMADP_INTERLEAVED_MAKESPAN_MAX_OPS; i++) {
            int op_idx = op_order[i];
            if (op_done[op_idx]) continue;

            bool is_dropoff = (op_idx < n);
            int customer = is_dropoff ? (op_idx + 1) : (op_idx - n + 1);

            // Check feasibility
            bool feasible = false;
            int best_agent = -1;
            T best_time = 1e30;

            if (is_dropoff) {
                // Dropoff: need agent with available resources
                if (!dropped[customer]) {
                    int hinted_agent = (int)(agent_hints[op_idx] * num_agents);
                    if (hinted_agent >= num_agents) hinted_agent = num_agents - 1;

                    // Try hinted agent first, then others
                    for (int a = 0; a < num_agents && a < RCMADP_INTERLEAVED_MAKESPAN_MAX_AGENTS; a++) {
                        int agent = (a == 0) ? hinted_agent : ((a <= hinted_agent) ? a - 1 : a);
                        if (agent_res[agent] > 0) {
                            T travel = travel_matrix[agent_loc[agent] * matrix_dim + customer];
                            T arrival = agent_time[agent] + travel;
                            if (arrival < best_time) {
                                best_time = arrival;
                                best_agent = agent;
                                feasible = true;
                            }
                        }
                    }
                }
            } else {
                // Pickup: need dropoff done AND processing complete AND agent with capacity
                if (dropped[customer] && !picked_up[customer]) {
                    T ready_time = dropoff_time[customer] + proc_times[customer];

                    int hinted_agent = (int)(agent_hints[op_idx] * num_agents);
                    if (hinted_agent >= num_agents) hinted_agent = num_agents - 1;

                    for (int a = 0; a < num_agents && a < RCMADP_INTERLEAVED_MAKESPAN_MAX_AGENTS; a++) {
                        int agent = (a == 0) ? hinted_agent : ((a <= hinted_agent) ? a - 1 : a);
                        if (agent_res[agent] < resources_per_agent) {
                            T travel = travel_matrix[agent_loc[agent] * matrix_dim + customer];
                            T arrival = agent_time[agent] + travel;
                            if (arrival < ready_time) arrival = ready_time;
                            if (arrival < best_time) {
                                best_time = arrival;
                                best_agent = agent;
                                feasible = true;
                            }
                        }
                    }
                }
            }

            if (feasible && best_agent >= 0) {
                // Execute operation
                T travel = travel_matrix[agent_loc[best_agent] * matrix_dim + customer];
                total_travel += travel;

                if (is_dropoff) {
                    agent_time[best_agent] += travel;
                    agent_loc[best_agent] = customer;
                    agent_res[best_agent]--;
                    dropoff_time[customer] = agent_time[best_agent];
                    dropped[customer] = true;
                } else {
                    T ready_time = dropoff_time[customer] + proc_times[customer];
                    T arrival = agent_time[best_agent] + travel;
                    if (arrival < ready_time) arrival = ready_time;
                    agent_time[best_agent] = arrival;
                    agent_loc[best_agent] = customer;
                    agent_res[best_agent]++;
                    picked_up[customer] = true;
                }

                op_done[op_idx] = true;
                completed_ops++;
                made_progress = true;
            }
        }

        if (!made_progress) break;  // No progress possible, deadlock or complete
    }

    // Return all agents to depot and compute makespan
    T makespan = 0;
    for (int a = 0; a < num_agents && a < RCMADP_INTERLEAVED_MAKESPAN_MAX_AGENTS; a++) {
        if (agent_loc[a] != 0) {
            T return_travel = travel_matrix[agent_loc[a] * matrix_dim + 0];
            total_travel += return_travel;
            agent_time[a] += return_travel;
        }
        if (agent_time[a] > makespan) {
            makespan = agent_time[a];
        }
    }

    // Count unserviced customers
    int unserviced = 0;
    for (int c = 1; c <= n && c <= RCMADP_INTERLEAVED_MAKESPAN_MAX_CUSTOMERS; c++) {
        if (!dropped[c]) unserviced++;
        if (!picked_up[c]) unserviced++;
    }

    // Fitness = MAKESPAN + penalty for unserviced operations
    fitness[idx] = makespan + unserviced * penalty_weight;
}


template<typename T>
class RCMADPInterleavedMakespanConfig : public BRKGAConfig<T> {
public:
    // Problem parameters
    int num_agents;              // Number of agents/vehicles
    int resources_per_agent;     // Max resources each agent can carry
    int total_resources;         // Total available resources
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

    // Travel time matrix
    std::vector<std::vector<T>> travel_times;

    // Processing times per customer
    std::vector<T> processing_times;

    // GPU-specific members
    std::map<int, T*> d_travel_matrices;
    std::map<int, T*> d_proc_times;
    bool gpu_available;
    mutable std::mutex gpu_mutex;

    // Operation representation
    struct Operation {
        int op_id;           // Unique operation ID
        int customer;        // Customer ID (1-indexed)
        makespan_opt::OpType type;         // DROPOFF or PICKUP
        T priority;          // Chromosome-derived priority
        int agent_hint;      // Hinted agent
    };

    // Solution representation
    struct AgentSchedule {
        int agent_id;
        std::vector<std::tuple<int, makespan_opt::OpType, T>> operations;  // (customer_id, type, time)
        T total_travel_time;
        T finish_time;
        int resources_carried;
    };

    struct Solution {
        std::vector<AgentSchedule> agent_schedules;
        T total_travel_cost;
        T makespan;
        int unserviced_dropoffs;
        int unserviced_pickups;
        int constraint_violations;
        std::vector<std::pair<int, makespan_opt::OpType>> unserviced_list;
    };

public:
    RCMADPInterleavedMakespanConfig(int agents = 3, int resources = 2)
        : BRKGAConfig<T>({1, 1}),  // Will be updated after loading
          num_agents(agents),
          resources_per_agent(resources),
          total_resources(agents * resources),
          depot_return_penalty(10000.0),
          gpu_available(false) {
        check_gpu_availability();
    }

    ~RCMADPInterleavedMakespanConfig() {
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

        T* d_travel = nullptr;
        cudaError_t err1 = cudaMalloc(&d_travel, matrix_size * sizeof(T));

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
            std::cerr << "RCMADP Interleaved Makespan GPU: memory not allocated on device " << device_id << std::endl;
            return;
        }

        dim3 block(this->threads_per_block);
        dim3 grid((pop_size + block.x - 1) / block.x);

        rcmadp_interleaved_makespan_fitness_kernel<<<grid, block>>>(
            d_population, d_fitness, d_travel, d_proc,
            pop_size, chrom_len, num_customers, num_agents,
            resources_per_agent, depot_return_penalty
        );

        cudaError_t error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            std::cerr << "RCMADP Interleaved Makespan GPU kernel failed on device " << device_id
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

    // Load from TSPJ format (_TT.csv and _JT.csv paired files)
    static std::unique_ptr<RCMADPInterleavedMakespanConfig<T>> load_from_tspj(
            const std::string& filename,
            int num_agents = 3,
            int resources_per_agent = 2) {

        auto config = std::make_unique<RCMADPInterleavedMakespanConfig<T>>(num_agents, resources_per_agent);

        std::string basename = FileUtils::get_basename(filename);
        std::string directory = FileUtils::get_directory(filename);

        std::string tt_file, jt_file;

        if (basename.find("_TT") != std::string::npos) {
            tt_file = filename;
            jt_file = directory + "/" + basename.substr(0, basename.find("_TT")) + "_JT.csv";
            config->instance_name = basename.substr(0, basename.find("_TT"));
        } else if (basename.find("_JT") != std::string::npos) {
            jt_file = filename;
            tt_file = directory + "/" + basename.substr(0, basename.find("_JT")) + "_TT.csv";
            config->instance_name = basename.substr(0, basename.find("_JT"));
        } else {
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

        config->travel_times = load_csv_matrix(tt_file);
        config->num_customers = config->travel_times.size() - 1;

        auto job_matrix = load_csv_matrix(jt_file);
        config->extract_processing_times_from_job_matrix(job_matrix);

        config->locations.resize(config->travel_times.size());
        for (size_t i = 0; i < config->locations.size(); i++) {
            config->locations[i] = {static_cast<int>(i), T(0), T(0), config->processing_times[i]};
        }

        config->setup_brkga_params();

        return config;
    }

    // Load from TSP + processing times files
    static std::unique_ptr<RCMADPInterleavedMakespanConfig<T>> load_from_files(
            const std::string& tsp_file,
            const std::string& processing_times_file,
            int num_agents = 3,
            int resources_per_agent = 2) {

        auto config = std::make_unique<RCMADPInterleavedMakespanConfig<T>>(num_agents, resources_per_agent);
        config->load_tsp_file(tsp_file);
        config->load_processing_times(processing_times_file);
        config->setup_brkga_params();

        return config;
    }

    void set_parameters(int agents, int resources) {
        num_agents = agents;
        resources_per_agent = resources;
        total_resources = agents * resources;
    }

private:
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

    void extract_processing_times_from_job_matrix(const std::vector<std::vector<T>>& job_matrix) {
        processing_times.clear();
        processing_times.push_back(T(0));  // Depot

        for (size_t i = 1; i < job_matrix.size(); i++) {
            T proc_time = (i < job_matrix[i].size()) ? job_matrix[i][i] : T(0);

            if (proc_time <= 0) {
                T sum = 0;
                int count = 0;
                for (size_t j = 1; j < job_matrix[i].size(); j++) {
                    if (job_matrix[i][j] > 0) {
                        sum += job_matrix[i][j];
                        count++;
                    }
                }
                proc_time = (count > 0) ? (sum / count) : T(30);
            }

            if (proc_time > 1000) {
                proc_time = proc_time / 100;
            }

            processing_times.push_back(proc_time);
        }
    }

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
                size_t pos = line.find(":");
                if (pos != std::string::npos) {
                    dimension = std::stoi(line.substr(pos + 1));
                    num_customers = dimension - 1;
                }
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
            for (int i = 0; i < dimension; i++) {
                int id;
                T x, y;
                file >> id >> x >> y;
                locations[id - 1] = {id - 1, x, y, T(0)};
            }
            build_distance_matrix(edge_weight_type);
        }
        else if (in_edge_weight_section) {
            travel_times.resize(dimension, std::vector<T>(dimension));

            if (edge_weight_format.find("FULL_MATRIX") != std::string::npos) {
                for (int i = 0; i < dimension; i++) {
                    for (int j = 0; j < dimension; j++) {
                        file >> travel_times[i][j];
                    }
                }
            }
            else {
                for (int i = 0; i < dimension; i++) {
                    travel_times[i][i] = 0;
                    for (int j = i + 1; j < dimension; j++) {
                        file >> travel_times[i][j];
                        travel_times[j][i] = travel_times[i][j];
                    }
                }
            }

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

    void load_processing_times(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open processing times file: " + filename);
        }

        processing_times.clear();
        processing_times.push_back(T(0));  // Depot

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

        while (processing_times.size() < static_cast<size_t>(num_customers + 1)) {
            processing_times.push_back(T(10));
        }

        for (size_t i = 0; i < locations.size() && i < processing_times.size(); i++) {
            locations[i].processing_time = processing_times[i];
        }
    }

    void setup_brkga_params() {
        // Chromosome: [operation_priorities (2n) | agent_hints (2n)]
        int num_ops = 2 * num_customers;
        this->component_lengths = {num_ops, num_ops};
        this->num_components = 2;

        // Scale population based on problem complexity
        int complexity = num_customers * num_agents;
        if (complexity <= 50) {
            this->population_size = 600;
            this->max_generations = 1200;
        } else if (complexity <= 150) {
            this->population_size = 1000;
            this->max_generations = 2000;
        } else if (complexity <= 300) {
            this->population_size = 1500;
            this->max_generations = 2500;
        } else {
            this->population_size = 2500;
            this->max_generations = 4000;
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
    // Main fitness calculation - MINIMIZES MAKESPAN
    T calculate_fitness(const Individual<T>& individual) const {
        Solution sol = decode_solution(individual);

        // Primary objective: MAKESPAN (time when last agent finishes)
        T fitness = sol.makespan;

        // Penalties
        T unserviced_penalty = (sol.unserviced_dropoffs + sol.unserviced_pickups) * depot_return_penalty;
        T violation_penalty = sol.constraint_violations * T(5000);

        return fitness + unserviced_penalty + violation_penalty;
    }

    // Decode chromosome into full solution with interleaved operations
    Solution decode_solution(const Individual<T>& individual) const {
        const auto& op_priorities = individual.get_component(0);
        const auto& agent_hints = individual.get_component(1);

        int num_ops = 2 * num_customers;

        Solution sol;
        sol.total_travel_cost = 0;
        sol.makespan = 0;
        sol.unserviced_dropoffs = 0;
        sol.unserviced_pickups = 0;
        sol.constraint_violations = 0;

        // Initialize agent schedules
        sol.agent_schedules.resize(num_agents);
        for (int a = 0; a < num_agents; a++) {
            sol.agent_schedules[a].agent_id = a;
            sol.agent_schedules[a].total_travel_time = 0;
            sol.agent_schedules[a].finish_time = 0;
            sol.agent_schedules[a].resources_carried = resources_per_agent;
        }

        // Create unified operation list
        // op_index 0..n-1 = dropoff for customer (op_index+1)
        // op_index n..2n-1 = pickup for customer (op_index-n+1)
        std::vector<std::pair<T, int>> op_order;
        for (int i = 0; i < num_ops; i++) {
            op_order.push_back({op_priorities[i], i});
        }
        std::sort(op_order.begin(), op_order.end());

        // Track agent states
        std::vector<T> agent_time(num_agents, T(0));
        std::vector<int> agent_location(num_agents, 0);  // All start at depot
        std::vector<int> agent_resources(num_agents, resources_per_agent);

        // Track customer states
        std::vector<T> dropoff_times(num_customers + 1, T(-1));
        std::vector<bool> dropped(num_customers + 1, false);
        std::vector<bool> picked_up(num_customers + 1, false);

        // Track operation completion
        std::vector<bool> op_done(num_ops, false);
        int completed_ops = 0;

        // Multiple passes to handle dependencies
        for (int pass = 0; pass < num_ops && completed_ops < num_ops; pass++) {
            bool made_progress = false;

            for (const auto& [priority, op_idx] : op_order) {
                if (op_done[op_idx]) continue;

                bool is_dropoff = (op_idx < num_customers);
                int customer = is_dropoff ? (op_idx + 1) : (op_idx - num_customers + 1);

                // Get agent hint
                int hinted_agent = static_cast<int>(agent_hints[op_idx] * num_agents);
                hinted_agent = std::min(hinted_agent, num_agents - 1);

                // Build agent order (hinted first)
                std::vector<int> agent_order;
                agent_order.push_back(hinted_agent);
                for (int a = 0; a < num_agents; a++) {
                    if (a != hinted_agent) agent_order.push_back(a);
                }

                bool feasible = false;
                int best_agent = -1;
                T best_time = std::numeric_limits<T>::max();

                if (is_dropoff) {
                    // Dropoff: need agent with available resources
                    if (!dropped[customer]) {
                        for (int a : agent_order) {
                            if (agent_resources[a] > 0) {
                                T travel = travel_times[agent_location[a]][customer];
                                T arrival = agent_time[a] + travel;
                                if (arrival < best_time) {
                                    best_time = arrival;
                                    best_agent = a;
                                    feasible = true;
                                }
                            }
                        }
                    }
                } else {
                    // Pickup: need dropoff done AND processing complete
                    if (dropped[customer] && !picked_up[customer]) {
                        T ready_time = dropoff_times[customer] + processing_times[customer];

                        for (int a : agent_order) {
                            if (agent_resources[a] < resources_per_agent) {
                                T travel = travel_times[agent_location[a]][customer];
                                T arrival = std::max(agent_time[a] + travel, ready_time);
                                if (arrival < best_time) {
                                    best_time = arrival;
                                    best_agent = a;
                                    feasible = true;
                                }
                            }
                        }
                    }
                }

                if (feasible && best_agent >= 0) {
                    // Execute operation
                    T travel = travel_times[agent_location[best_agent]][customer];

                    if (is_dropoff) {
                        agent_time[best_agent] += travel;
                        agent_location[best_agent] = customer;
                        agent_resources[best_agent]--;
                        dropoff_times[customer] = agent_time[best_agent];
                        dropped[customer] = true;

                        sol.agent_schedules[best_agent].operations.push_back(
                            {customer, makespan_opt::OpType::DROPOFF, agent_time[best_agent]});
                    } else {
                        T ready_time = dropoff_times[customer] + processing_times[customer];
                        T arrival = std::max(agent_time[best_agent] + travel, ready_time);
                        agent_time[best_agent] = arrival;
                        agent_location[best_agent] = customer;
                        agent_resources[best_agent]++;
                        picked_up[customer] = true;

                        sol.agent_schedules[best_agent].operations.push_back(
                            {customer, makespan_opt::OpType::PICKUP, agent_time[best_agent]});
                    }

                    sol.agent_schedules[best_agent].total_travel_time += travel;
                    op_done[op_idx] = true;
                    completed_ops++;
                    made_progress = true;
                }
            }

            if (!made_progress) break;
        }

        // Return all agents to depot
        for (int a = 0; a < num_agents; a++) {
            if (agent_location[a] != 0) {
                T return_travel = travel_times[agent_location[a]][0];
                agent_time[a] += return_travel;
                sol.agent_schedules[a].total_travel_time += return_travel;
            }
            sol.agent_schedules[a].finish_time = agent_time[a];
            sol.total_travel_cost += sol.agent_schedules[a].total_travel_time;
            sol.makespan = std::max(sol.makespan, sol.agent_schedules[a].finish_time);
        }

        // Count unserviced
        for (int c = 1; c <= num_customers; c++) {
            if (!dropped[c]) {
                sol.unserviced_dropoffs++;
                sol.unserviced_list.push_back({c, makespan_opt::OpType::DROPOFF});
            }
            if (!picked_up[c]) {
                sol.unserviced_pickups++;
                sol.unserviced_list.push_back({c, makespan_opt::OpType::PICKUP});
            }
        }

        return sol;
    }

    // Decode to vector format for external use
    std::vector<std::vector<T>> decode_to_solution_vector(const Individual<T>& individual) const {
        Solution sol = decode_solution(individual);

        std::vector<std::vector<T>> result;

        // Format: For each agent, output operations as [op_type, customer, time, ...]
        // op_type: 0 = dropoff, 1 = pickup
        for (const auto& schedule : sol.agent_schedules) {
            std::vector<T> agent_ops;

            for (const auto& [customer, type, time] : schedule.operations) {
                agent_ops.push_back(type == makespan_opt::OpType::DROPOFF ? T(0) : T(1));
                agent_ops.push_back(static_cast<T>(customer));
                agent_ops.push_back(time);
            }

            result.push_back(agent_ops);
        }

        return result;
    }

    // Validation
    bool validate_solution(const Individual<T>& individual) override {
        Solution sol = decode_solution(individual);

        // Check all operations serviced
        if (sol.unserviced_dropoffs > 0 || sol.unserviced_pickups > 0) return false;
        if (sol.constraint_violations > 0) return false;

        // Verify resource constraints for each agent
        for (const auto& schedule : sol.agent_schedules) {
            int resources = resources_per_agent;

            for (const auto& [customer, type, time] : schedule.operations) {
                if (type == makespan_opt::OpType::DROPOFF) {
                    resources--;
                    if (resources < 0) return false;
                } else {
                    resources++;
                    if (resources > resources_per_agent) return false;
                }
            }
        }

        // Verify pickup-after-processing constraint
        std::vector<T> dropoff_times(num_customers + 1, T(-1));
        for (const auto& schedule : sol.agent_schedules) {
            for (const auto& [customer, type, time] : schedule.operations) {
                if (type == makespan_opt::OpType::DROPOFF) {
                    dropoff_times[customer] = time;
                } else {
                    T ready_time = dropoff_times[customer] + processing_times[customer];
                    if (time < ready_time - T(0.001)) return false;  // Small epsilon for floating point
                }
            }
        }

        return true;
    }

    // Print solution
    void print_solution(const Individual<T>& individual) override {
        Solution sol = decode_solution(individual);

        std::cout << "\n=== RCMADP Interleaved Solution (Makespan Optimized) ===" << std::endl;
        std::cout << "Makespan: " << std::fixed << std::setprecision(2)
                  << sol.makespan << std::endl;
        std::cout << "Total Travel Cost: " << sol.total_travel_cost << std::endl;
        std::cout << "Unserviced: " << (sol.unserviced_dropoffs + sol.unserviced_pickups)
                  << " (" << sol.unserviced_dropoffs << " dropoffs, "
                  << sol.unserviced_pickups << " pickups)" << std::endl;

        for (const auto& schedule : sol.agent_schedules) {
            std::cout << "\nAgent " << schedule.agent_id << ":" << std::endl;
            std::cout << "  Finish Time: " << schedule.finish_time << std::endl;
            std::cout << "  Travel Time: " << schedule.total_travel_time << std::endl;
            std::cout << "  Operations: ";

            for (const auto& [customer, type, time] : schedule.operations) {
                std::cout << (type == makespan_opt::OpType::DROPOFF ? "D" : "P")
                          << customer << "@" << std::setprecision(1) << time << " ";
            }
            std::cout << std::endl;
        }

        if (!sol.unserviced_list.empty()) {
            std::cout << "\nUnserviced: ";
            for (const auto& [customer, type] : sol.unserviced_list) {
                std::cout << (type == makespan_opt::OpType::DROPOFF ? "D" : "P") << customer << " ";
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

        file << "RCMADP Interleaved Solution (Makespan Optimized)\n";
        file << "Instance: " << instance_name << "\n";
        file << "Agents: " << num_agents << "\n";
        file << "Resources per Agent: " << resources_per_agent << "\n";
        file << "Customers: " << num_customers << "\n";
        file << "\n";
        file << "Makespan: " << std::fixed << std::setprecision(2)
             << sol.makespan << "\n";
        file << "Total Travel Cost: " << sol.total_travel_cost << "\n";
        file << "Fitness: " << individual.fitness << "\n";
        file << "\n";

        for (const auto& schedule : sol.agent_schedules) {
            file << "Agent " << schedule.agent_id << ":\n";
            file << "  Finish Time: " << schedule.finish_time << "\n";
            file << "  Route Travel Time: " << schedule.total_travel_time << "\n";
            file << "  Operations:\n";

            for (const auto& [customer, type, time] : schedule.operations) {
                file << "    " << (type == makespan_opt::OpType::DROPOFF ? "DROPOFF" : "PICKUP")
                     << " Customer " << customer
                     << " at time " << std::setprecision(2) << time;
                if (type == makespan_opt::OpType::DROPOFF) {
                    file << " (processing: " << processing_times[customer] << ")";
                }
                file << "\n";
            }
            file << "\n";
        }

        file.close();
    }

    // Export solution to JSON format (compatible with generate_html_report.py)
    void export_solution_json(const Individual<T>& individual, const std::string& filename,
                              double solve_time_seconds = 0.0,
                              const std::vector<std::pair<int, T>>& convergence = {}) {
        Solution sol = decode_solution(individual);

        FileUtils::ensure_directory(FileUtils::get_directory(filename));

        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot create JSON solution file: " + filename);
        }

        file << std::fixed << std::setprecision(2);
        file << "{\n";
        file << "  \"instance\": \"" << instance_name << "\",\n";
        file << "  \"objective\": \"makespan\",\n";
        file << "  \"makespan\": " << sol.makespan << ",\n";
        file << "  \"total_travel_cost\": " << sol.total_travel_cost << ",\n";
        file << "  \"fitness\": " << individual.fitness << ",\n";
        file << "  \"solve_time_seconds\": " << solve_time_seconds << ",\n";
        file << "  \"problem\": {\n";
        file << "    \"n_customers\": " << num_customers << ",\n";
        file << "    \"n_agents\": " << num_agents << ",\n";
        file << "    \"resources_per_agent\": " << resources_per_agent << "\n";
        file << "  },\n";

        // Routes array
        file << "  \"routes\": [\n";
        for (size_t a = 0; a < sol.agent_schedules.size(); a++) {
            const auto& schedule = sol.agent_schedules[a];
            file << "    {\n";
            file << "      \"agent\": " << schedule.agent_id << ",\n";
            file << "      \"travel_time\": " << schedule.total_travel_time << ",\n";
            file << "      \"finish_time\": " << schedule.finish_time << ",\n";
            file << "      \"stops\": [\n";

            // Add depot start
            file << "        {\"time\": 0, \"op\": \"D\", \"node\": 0}";

            for (const auto& [customer, type, time] : schedule.operations) {
                file << ",\n        {\"time\": " << time
                     << ", \"op\": \"" << (type == makespan_opt::OpType::DROPOFF ? "D" : "P")
                     << "\", \"node\": " << customer << "}";
            }

            // Add depot end
            file << ",\n        {\"time\": " << schedule.finish_time << ", \"op\": \"P\", \"node\": 0}";

            file << "\n      ]\n";
            file << "    }";
            if (a < sol.agent_schedules.size() - 1) file << ",";
            file << "\n";
        }
        file << "  ],\n";

        // Convergence array
        file << "  \"convergence\": [";
        for (size_t i = 0; i < convergence.size(); i++) {
            if (i > 0) file << ", ";
            file << "[" << convergence[i].first << ", " << convergence[i].second << "]";
        }
        file << "],\n";

        // Unserviced info
        file << "  \"unserviced_dropoffs\": " << sol.unserviced_dropoffs << ",\n";
        file << "  \"unserviced_pickups\": " << sol.unserviced_pickups << "\n";
        file << "}\n";

        file.close();
    }

    // Print instance info
    void print_instance_info() const {
        std::cout << "\n=== RCMADP Interleaved Instance (Makespan Optimization) ===" << std::endl;
        std::cout << "Instance: " << instance_name << std::endl;
        std::cout << "Customers: " << num_customers << std::endl;
        std::cout << "Agents: " << num_agents << std::endl;
        std::cout << "Resources per Agent: " << resources_per_agent << std::endl;
        std::cout << "Total Resources: " << total_resources << std::endl;
        std::cout << "Chromosome Length: " << (4 * num_customers) << " genes" << std::endl;
        std::cout << "Objective: MINIMIZE MAKESPAN" << std::endl;

        if (!processing_times.empty()) {
            T min_proc = *std::min_element(processing_times.begin() + 1, processing_times.end());
            T max_proc = *std::max_element(processing_times.begin() + 1, processing_times.end());
            T avg_proc = std::accumulate(processing_times.begin() + 1, processing_times.end(), T(0)) / num_customers;
            std::cout << "Processing Times: min=" << min_proc << ", max=" << max_proc
                      << ", avg=" << std::setprecision(1) << avg_proc << std::endl;
        }
        std::cout << "============================================================" << std::endl;
    }

    // Configuration helpers
    static void configure_for_size(RCMADPInterleavedMakespanConfig<T>* config, int num_customers, int num_agents) {
        int complexity = num_customers * num_agents;

        if (complexity <= 30) {
            config->population_size = 500;
            config->max_generations = 1000;
        } else if (complexity <= 100) {
            config->population_size = 1000;
            config->max_generations = 1800;
        } else if (complexity <= 300) {
            config->population_size = 1800;
            config->max_generations = 3000;
        } else {
            config->population_size = 3000;
            config->max_generations = 5000;
        }

        config->elite_size = config->population_size / 5;
        config->mutant_size = config->population_size / 8;
        config->elite_prob = 0.7;
        config->update_cuda_grid_size();
    }
};

#endif // RCMADP_INTERLEAVED_MAKESPAN_CONFIG_HPP
