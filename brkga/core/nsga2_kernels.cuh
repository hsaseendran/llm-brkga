// core/nsga2_kernels.cuh - CUDA kernels for multi-objective optimization
#ifndef NSGA2_KERNELS_CUH
#define NSGA2_KERNELS_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>

// Kernel for parallel objective evaluation
template<typename T>
__global__ void evaluate_objectives_kernel(
    T* population,
    T* objectives,
    int pop_size,
    int chrom_len,
    int num_objectives,
    T* problem_data,
    int problem_data_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;
    
    T* chromosome = population + idx * chrom_len;
    T* obj_values = objectives + idx * num_objectives;
    
    // Example: problem-specific objective calculation
    // This would be specialized per problem type
    for (int obj = 0; obj < num_objectives; obj++) {
        obj_values[obj] = 0;
        for (int i = 0; i < chrom_len; i++) {
            obj_values[obj] += chromosome[i] * problem_data[i % problem_data_size];
        }
    }
}

// Kernel for parallel dominance checking
template<typename T>
__global__ void check_dominance_kernel(
    T* objectives,
    int* domination_matrix,
    int pop_size,
    int num_objectives,
    bool* minimize_flags
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= pop_size || j >= pop_size || i == j) return;
    
    T* obj_i = objectives + i * num_objectives;
    T* obj_j = objectives + j * num_objectives;
    
    bool i_dominates_j = true;
    bool j_dominates_i = true;
    bool at_least_one_better_i = false;
    bool at_least_one_better_j = false;
    
    for (int obj = 0; obj < num_objectives; obj++) {
        bool minimize = minimize_flags[obj];
        
        if (minimize) {
            if (obj_i[obj] > obj_j[obj]) i_dominates_j = false;
            if (obj_j[obj] > obj_i[obj]) j_dominates_i = false;
            if (obj_i[obj] < obj_j[obj]) at_least_one_better_i = true;
            if (obj_j[obj] < obj_i[obj]) at_least_one_better_j = true;
        } else {
            if (obj_i[obj] < obj_j[obj]) i_dominates_j = false;
            if (obj_j[obj] < obj_i[obj]) j_dominates_i = false;
            if (obj_i[obj] > obj_j[obj]) at_least_one_better_i = true;
            if (obj_j[obj] > obj_i[obj]) at_least_one_better_j = true;
        }
    }
    
    i_dominates_j = i_dominates_j && at_least_one_better_i;
    j_dominates_i = j_dominates_i && at_least_one_better_j;
    
    int matrix_idx = i * pop_size + j;
    if (i_dominates_j) {
        domination_matrix[matrix_idx] = 1;  // i dominates j
    } else if (j_dominates_i) {
        domination_matrix[matrix_idx] = -1; // j dominates i
    } else {
        domination_matrix[matrix_idx] = 0;  // non-dominated
    }
}

// Kernel for counting domination
template<typename T>
__global__ void count_domination_kernel(
    int* domination_matrix,
    int* domination_count,
    int* dominated_count,
    int pop_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;
    
    int dominated_by = 0;
    int dominates = 0;
    
    for (int j = 0; j < pop_size; j++) {
        int value = domination_matrix[idx * pop_size + j];
        if (value == 1) dominates++;
        if (value == -1) dominated_by++;
    }
    
    domination_count[idx] = dominated_by;
    dominated_count[idx] = dominates;
}

// Kernel for crowding distance calculation
template<typename T>
__global__ void crowding_distance_kernel(
    T* objectives,
    int* front_indices,
    T* crowding_distances,
    int front_size,
    int num_objectives,
    int objective_idx
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= front_size) return;
    
    int individual_idx = front_indices[idx];
    
    // Boundary points get infinite distance
    if (idx == 0 || idx == front_size - 1) {
        crowding_distances[individual_idx] = INFINITY;
        return;
    }
    
    int prev_idx = front_indices[idx - 1];
    int next_idx = front_indices[idx + 1];
    
    T obj_prev = objectives[prev_idx * num_objectives + objective_idx];
    T obj_curr = objectives[individual_idx * num_objectives + objective_idx];
    T obj_next = objectives[next_idx * num_objectives + objective_idx];
    
    T obj_min = objectives[front_indices[0] * num_objectives + objective_idx];
    T obj_max = objectives[front_indices[front_size - 1] * num_objectives + objective_idx];
    
    T range = obj_max - obj_min;
    if (range > 1e-10) {
        T distance = (obj_next - obj_prev) / range;
        atomicAdd(&crowding_distances[individual_idx], distance);
    }
}

// Kernel for tournament selection
template<typename T>
__global__ void tournament_selection_kernel(
    int* ranks,
    T* crowding_distances,
    int* selected_indices,
    curandState* states,
    int pop_size,
    int num_selections,
    int tournament_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_selections) return;
    
    curandState localState = states[idx];
    
    int best_idx = curand(&localState) % pop_size;
    int best_rank = ranks[best_idx];
    T best_crowding = crowding_distances[best_idx];
    
    for (int i = 1; i < tournament_size; i++) {
        int candidate_idx = curand(&localState) % pop_size;
        int candidate_rank = ranks[candidate_idx];
        T candidate_crowding = crowding_distances[candidate_idx];
        
        // Better rank (lower is better)
        if (candidate_rank < best_rank) {
            best_idx = candidate_idx;
            best_rank = candidate_rank;
            best_crowding = candidate_crowding;
        }
        // Same rank, better crowding distance
        else if (candidate_rank == best_rank && candidate_crowding > best_crowding) {
            best_idx = candidate_idx;
            best_crowding = candidate_crowding;
        }
    }
    
    selected_indices[idx] = best_idx;
    states[idx] = localState;
}

// Kernel for uniform crossover
template<typename T>
__global__ void uniform_crossover_kernel(
    T* parent1_population,
    T* parent2_population,
    T* offspring_population,
    int* parent1_indices,
    int* parent2_indices,
    curandState* states,
    int num_offspring,
    int chrom_len,
    double crossover_prob
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_offspring) return;
    
    curandState localState = states[idx];
    
    int p1_idx = parent1_indices[idx];
    int p2_idx = parent2_indices[idx];
    
    T* parent1 = parent1_population + p1_idx * chrom_len;
    T* parent2 = parent2_population + p2_idx * chrom_len;
    T* offspring = offspring_population + idx * chrom_len;
    
    for (int i = 0; i < chrom_len; i++) {
        if (curand_uniform(&localState) < crossover_prob) {
            offspring[i] = parent1[i];
        } else {
            offspring[i] = parent2[i];
        }
    }
    
    states[idx] = localState;
}

// Kernel for polynomial mutation (NSGA-II style)
template<typename T>
__global__ void polynomial_mutation_kernel(
    T* population,
    curandState* states,
    int pop_size,
    int chrom_len,
    double mutation_rate,
    double distribution_index
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;
    
    curandState localState = states[idx];
    T* individual = population + idx * chrom_len;
    
    for (int i = 0; i < chrom_len; i++) {
        if (curand_uniform(&localState) < mutation_rate) {
            T gene = individual[i];
            T delta;
            
            double rand = curand_uniform(&localState);
            
            if (rand < 0.5) {
                delta = pow(2.0 * rand, 1.0 / (distribution_index + 1.0)) - 1.0;
            } else {
                delta = 1.0 - pow(2.0 * (1.0 - rand), 1.0 / (distribution_index + 1.0));
            }
            
            gene += delta;
            
            // Clamp to [0, 1]
            gene = max(T(0), min(T(1), gene));
            individual[i] = gene;
        }
    }
    
    states[idx] = localState;
}

// Kernel for parallel sorting of front by objective
template<typename T>
__global__ void sort_front_by_objective_kernel(
    T* objectives,
    int* front_indices,
    int* sorted_indices,
    int front_size,
    int num_objectives,
    int objective_idx
) {
    // Simple bubble sort for small fronts
    // For larger fronts, use thrust or more sophisticated sorting
    extern __shared__ T shared_values[];
    int* shared_indices = (int*)&shared_values[blockDim.x];
    
    int tid = threadIdx.x;
    
    if (tid < front_size) {
        int idx = front_indices[tid];
        shared_values[tid] = objectives[idx * num_objectives + objective_idx];
        shared_indices[tid] = idx;
    }
    __syncthreads();
    
    // Bubble sort in shared memory
    for (int i = 0; i < front_size; i++) {
        for (int j = tid; j < front_size - 1; j += blockDim.x) {
            if (shared_values[j] > shared_values[j + 1]) {
                // Swap values
                T temp_val = shared_values[j];
                shared_values[j] = shared_values[j + 1];
                shared_values[j + 1] = temp_val;
                
                // Swap indices
                int temp_idx = shared_indices[j];
                shared_indices[j] = shared_indices[j + 1];
                shared_indices[j + 1] = temp_idx;
            }
        }
        __syncthreads();
    }
    
    if (tid < front_size) {
        sorted_indices[tid] = shared_indices[tid];
    }
}

// Kernel for merging populations
template<typename T>
__global__ void merge_populations_kernel(
    T* pop1,
    T* pop2,
    T* merged,
    int pop1_size,
    int pop2_size,
    int chrom_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = pop1_size + pop2_size;
    
    if (idx >= total_size) return;
    
    if (idx < pop1_size) {
        // Copy from pop1
        for (int i = 0; i < chrom_len; i++) {
            merged[idx * chrom_len + i] = pop1[idx * chrom_len + i];
        }
    } else {
        // Copy from pop2
        int pop2_idx = idx - pop1_size;
        for (int i = 0; i < chrom_len; i++) {
            merged[idx * chrom_len + i] = pop2[pop2_idx * chrom_len + i];
        }
    }
}

// Helper function to initialize dominance flags
template<typename T>
__global__ void init_minimize_flags_kernel(
    bool* minimize_flags,
    int num_objectives,
    bool all_minimize
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_objectives) return;
    
    minimize_flags[idx] = all_minimize;
}

// Utility namespace for CUDA operations
namespace NSGA2CudaUtils {
    template<typename T>
    void check_cuda_error(cudaError_t error, const char* message) {
        if (error != cudaSuccess) {
            throw std::runtime_error(std::string(message) + ": " + 
                                   std::string(cudaGetErrorString(error)));
        }
    }
    
    template<typename T>
    void sync_and_check(const char* operation) {
        cudaError_t error = cudaDeviceSynchronize();
        check_cuda_error<T>(error, operation);
    }
    
    dim3 calculate_grid_size(int total_threads, int threads_per_block) {
        return dim3((total_threads + threads_per_block - 1) / threads_per_block);
    }
    
    dim3 calculate_2d_grid_size(int width, int height, int threads_per_dim) {
        return dim3(
            (width + threads_per_dim - 1) / threads_per_dim,
            (height + threads_per_dim - 1) / threads_per_dim
        );
    }
}

// Wrapper class for managing NSGA-II GPU operations
template<typename T>
class NSGA2GpuManager {
private:
    int device_id;
    int pop_size;
    int chrom_len;
    int num_objectives;
    
    T* d_objectives;
    int* d_domination_matrix;
    int* d_domination_count;
    int* d_dominated_count;
    int* d_ranks;
    T* d_crowding_distances;
    bool* d_minimize_flags;
    
    bool memory_allocated;
    
public:
    NSGA2GpuManager(int device, int pop, int chrom, int obj)
        : device_id(device), pop_size(pop), chrom_len(chrom), num_objectives(obj),
          memory_allocated(false) {
        allocate_memory();
    }
    
    ~NSGA2GpuManager() {
        cleanup();
    }
    
    void allocate_memory() {
        cudaSetDevice(device_id);
        
        cudaMalloc(&d_objectives, pop_size * num_objectives * sizeof(T));
        cudaMalloc(&d_domination_matrix, pop_size * pop_size * sizeof(int));
        cudaMalloc(&d_domination_count, pop_size * sizeof(int));
        cudaMalloc(&d_dominated_count, pop_size * sizeof(int));
        cudaMalloc(&d_ranks, pop_size * sizeof(int));
        cudaMalloc(&d_crowding_distances, pop_size * sizeof(T));
        cudaMalloc(&d_minimize_flags, num_objectives * sizeof(bool));
        
        // Initialize minimize flags (all minimize by default)
        dim3 block(256);
        dim3 grid = NSGA2CudaUtils::calculate_grid_size(num_objectives, 256);
        init_minimize_flags_kernel<<<grid, block>>>(d_minimize_flags, num_objectives, true);
        
        NSGA2CudaUtils::sync_and_check<T>("NSGA-II GPU memory allocation");
        memory_allocated = true;
    }
    
    void cleanup() {
        if (memory_allocated) {
            cudaSetDevice(device_id);
            cudaFree(d_objectives);
            cudaFree(d_domination_matrix);
            cudaFree(d_domination_count);
            cudaFree(d_dominated_count);
            cudaFree(d_ranks);
            cudaFree(d_crowding_distances);
            cudaFree(d_minimize_flags);
            memory_allocated = false;
        }
    }
    
    T* get_objectives_ptr() { return d_objectives; }
    int* get_ranks_ptr() { return d_ranks; }
    T* get_crowding_distances_ptr() { return d_crowding_distances; }
};

#endif // NSGA2_KERNELS_CUH