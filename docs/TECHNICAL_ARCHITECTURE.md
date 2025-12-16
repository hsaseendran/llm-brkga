# LLM-BRKGA Technical Architecture Document

## Detailed Implementation Reference

---

## Table of Contents

1. [System Components Deep Dive](#1-system-components-deep-dive)
2. [Data Flow and Pipelines](#2-data-flow-and-pipelines)
3. [CUDA Implementation Details](#3-cuda-implementation-details)
4. [LLM Prompt Engineering](#4-llm-prompt-engineering)
5. [Configuration System](#5-configuration-system)
6. [Memory Management](#6-memory-management)
7. [Performance Analysis](#7-performance-analysis)
8. [Error Handling and Recovery](#8-error-handling-and-recovery)
9. [Testing Framework](#9-testing-framework)
10. [Deployment Guide](#10-deployment-guide)

---

## 1. System Components Deep Dive

### 1.1 Core Framework Components

#### 1.1.1 Individual Class (`individual.hpp`)

The `Individual` class is the fundamental unit of the genetic algorithm:

**Key Design Decisions:**
- **Multi-component support**: Chromosomes can have multiple components (e.g., route + schedule)
- **Lazy evaluation**: Fitness computed only when needed
- **GPU compatibility**: Flatten/unflatten methods for efficient device transfers

**Memory Layout:**
```
Individual Memory Structure:
┌─────────────────────────────────────────────────────────┐
│ Components Vector                                        │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐        │
│ │ Component 0 │ │ Component 1 │ │ Component N │        │
│ │ [r0...rk]   │ │ [r0...rm]   │ │ [r0...rp]   │        │
│ └─────────────┘ └─────────────┘ └─────────────┘        │
├─────────────────────────────────────────────────────────┤
│ Metadata                                                 │
│ - fitness: T                                            │
│ - objectives: vector<T>                                  │
│ - rank: int                                             │
│ - crowding_distance: T                                  │
│ - evaluated: bool                                       │
└─────────────────────────────────────────────────────────┘
```

**Flattening for GPU:**
```cpp
std::vector<T> Individual<T>::flatten() const {
    std::vector<T> flat;
    for (const auto& component : components) {
        flat.insert(flat.end(), component.begin(), component.end());
    }
    return flat;
}

void Individual<T>::unflatten(const std::vector<T>& data) {
    size_t offset = 0;
    for (auto& component : components) {
        std::copy(data.begin() + offset,
                  data.begin() + offset + component.size(),
                  component.begin());
        offset += component.size();
    }
}
```

#### 1.1.2 Population Class (`population.hpp`)

Manages the collection of individuals and implements NSGA-II operations:

**Population Structure:**
```
Population Layout:
┌──────────────────────────────────────────────────────┐
│                    Elite Section                      │
│  [Individual 0] [Individual 1] ... [Individual E-1]  │
├──────────────────────────────────────────────────────┤
│                  Offspring Section                    │
│  [Individual E] [Individual E+1] ... [Individual O]  │
├──────────────────────────────────────────────────────┤
│                   Mutant Section                      │
│  [Individual O+1] ... [Individual N-1]               │
└──────────────────────────────────────────────────────┘

Where: E = elite_size, O = elite_size + offspring_size, N = population_size
```

**NSGA-II Implementation:**

1. **Fast Non-Dominated Sort**: O(MN²) complexity
```cpp
void Population<T>::fast_non_dominated_sort() {
    // For each individual, count dominating solutions
    std::vector<int> domination_count(n, 0);
    std::vector<std::vector<int>> dominated_by(n);

    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (dominates(individuals[i], individuals[j])) {
                dominated_by[i].push_back(j);
                domination_count[j]++;
            } else if (dominates(individuals[j], individuals[i])) {
                dominated_by[j].push_back(i);
                domination_count[i]++;
            }
        }
    }

    // Build fronts
    fronts.clear();
    std::vector<int> current_front;
    for (int i = 0; i < n; i++) {
        if (domination_count[i] == 0) {
            current_front.push_back(i);
            individuals[i].rank = 0;
        }
    }
    // Continue for subsequent fronts...
}
```

2. **Crowding Distance**: Measures solution spread
```cpp
void Population<T>::calculate_crowding_distance_for_front(std::vector<int>& front) {
    for (int obj = 0; obj < num_objectives; obj++) {
        // Sort by objective
        std::sort(front.begin(), front.end(),
            [this, obj](int a, int b) {
                return individuals[a].objectives[obj] < individuals[b].objectives[obj];
            });

        // Boundary points get infinite distance
        individuals[front.front()].crowding_distance = INFINITY;
        individuals[front.back()].crowding_distance = INFINITY;

        // Calculate distances for interior points
        double range = individuals[front.back()].objectives[obj] -
                      individuals[front.front()].objectives[obj];
        if (range > 0) {
            for (size_t i = 1; i < front.size() - 1; i++) {
                individuals[front[i]].crowding_distance +=
                    (individuals[front[i+1]].objectives[obj] -
                     individuals[front[i-1]].objectives[obj]) / range;
            }
        }
    }
}
```

#### 1.1.3 Solver Class (`solver.hpp`)

The main orchestration class that coordinates all operations:

**Execution Flow:**
```
┌─────────────────────────────────────────────────────────┐
│                    Solver::run()                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. initialize_gpu_system()                             │
│     ├── detect_gpus()                                   │
│     ├── determine_execution_strategy()                  │
│     └── setup_gpu_resources()                           │
│                                                          │
│  2. initialize()                                        │
│     ├── Initialize GPU-resident islands (if enabled)   │
│     └── Evaluate initial population                     │
│                                                          │
│  3. for gen in 0..max_generations:                      │
│     │                                                    │
│     ├── evolve_generation()                             │
│     │   ├── [BRKGA] evolve_generation_brkga()          │
│     │   │   ├── perform_crossover()                    │
│     │   │   ├── perform_mutation()                     │
│     │   │   └── finalize_generation()                  │
│     │   │                                               │
│     │   └── [NSGA-II] evolve_generation_nsga2()        │
│     │       ├── generate_offspring()                   │
│     │       ├── evaluate_population()                  │
│     │       ├── fast_non_dominated_sort()              │
│     │       └── select_next_generation()               │
│     │                                                    │
│     ├── apply_local_search() (if configured)           │
│     │                                                    │
│     └── print_statistics() (periodic)                  │
│                                                          │
│  4. apply_final_polish() (intensive local search)      │
│                                                          │
│  5. print_final_results()                               │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 1.2 LLM Solver Components

#### 1.2.1 Problem Analyzer (`problem_analyzer.py`)

**Analysis Pipeline:**
```
Natural Language Description
        │
        ▼
┌───────────────────────────┐
│  Build Analysis Prompt    │
│  - Include problem desc   │
│  - Add clarifying Q&A     │
│  - Parse data file info   │
└───────────┬───────────────┘
            │
            ▼
┌───────────────────────────┐
│  LLM API Call             │
│  - Model: claude-sonnet   │
│  - Temperature: 0.2       │
│  - Max tokens: 4000       │
└───────────┬───────────────┘
            │
            ▼
┌───────────────────────────┐
│  Parse JSON Response      │
│  - Extract problem_type   │
│  - Extract variables      │
│  - Extract objectives     │
│  - Extract constraints    │
│  - Select decoder         │
└───────────┬───────────────┘
            │
            ▼
    ProblemStructure Object
```

**Data File Parsing:**
The system supports multiple data formats:

| Format | Parser | Example Files |
|--------|--------|---------------|
| TSPLIB | `parse_tsp_file()` | berlin52.tsp, kroA100.tsp |
| CSV | `parse_csv_file()` | jobs.csv, distances.csv |
| JSON | `parse_json_file()` | problem.json |
| Custom | User-defined | Varies |

```python
class DataFileParser:
    def parse_file(self, file_path: str) -> DataMetadata:
        extension = Path(file_path).suffix.lower()

        if extension == '.tsp':
            return self._parse_tsplib(file_path)
        elif extension == '.csv':
            return self._parse_csv(file_path)
        elif extension == '.json':
            return self._parse_json(file_path)
        else:
            return self._parse_generic(file_path)

    def _parse_tsplib(self, path: str) -> DataMetadata:
        # Parse TSPLIB format
        # NAME: berlin52
        # TYPE: TSP
        # DIMENSION: 52
        # EDGE_WEIGHT_TYPE: EUC_2D
        # NODE_COORD_SECTION
        # 1 565.0 575.0
        # ...
        pass
```

#### 1.2.2 Code Generator (`code_generator.py`)

**Generation Process:**
```
ProblemStructure + Context
        │
        ▼
┌───────────────────────────┐
│  Build Generation Prompt  │
│  - Problem specification  │
│  - Hyperparameters        │
│  - Data file info         │
│  - Code requirements      │
└───────────┬───────────────┘
            │
            ▼
┌───────────────────────────┐
│  LLM API Call             │
│  - System: BRKGA context  │
│  - Model: claude-sonnet   │
│  - Temperature: 0.1       │
│  - Max tokens: 8000       │
└───────────┬───────────────┘
            │
            ▼
┌───────────────────────────┐
│  Extract C++ Code         │
│  - Find code blocks       │
│  - Clean up formatting    │
└───────────┬───────────────┘
            │
            ▼
    config_name.hpp File
```

**Context Package:**
The context provided to the LLM contains:
1. Complete BRKGA framework API documentation
2. Example configurations for common problems
3. Critical rules for code generation
4. Common pitfalls and how to avoid them

#### 1.2.3 Validator (`validator.py`)

Validates generated code before compilation:

**Validation Checks:**
```python
class Validator:
    def full_validation(self, config_path: str,
                       problem: ProblemStructure) -> Tuple[bool, List]:
        results = []

        # Syntax validation
        results.append(self._validate_syntax(config_path))

        # Structure validation
        results.append(self._validate_structure(config_path))

        # API compliance
        results.append(self._validate_api_compliance(config_path))

        # Problem-specific checks
        results.append(self._validate_problem_specific(config_path, problem))

        overall_valid = all(r.success for r in results)
        return overall_valid, results

    def _validate_syntax(self, path: str) -> ValidationResult:
        # Check for balanced braces, proper includes, etc.
        pass

    def _validate_structure(self, path: str) -> ValidationResult:
        # Check class inherits BRKGAConfig<T>
        # Check required methods exist
        pass

    def _validate_api_compliance(self, path: str) -> ValidationResult:
        # Check function signatures match expected API
        # Check return types are correct
        pass
```

#### 1.2.4 Execution Manager (`execution_manager.py`)

Handles compilation and execution:

**Compilation Pipeline:**
```
Generated Config (.hpp)
        │
        ▼
┌───────────────────────────┐
│  Create Main File         │
│  - Detect config class    │
│  - Generate main()        │
│  - Handle data files      │
└───────────┬───────────────┘
            │
            ▼
┌───────────────────────────┐
│  NVCC Compilation         │
│  - C++17 standard         │
│  - CUDA architecture      │
│  - OpenMP linking         │
│  - cuRAND linking         │
└───────────┬───────────────┘
            │
            ▼
    Executable Binary
```

**Compilation Command:**
```bash
nvcc -std=c++17 -arch=sm_75 -O3 \
    -Xcompiler -fopenmp \
    -I brkga \
    main.cu \
    -o solver \
    -lcurand -lcudart -lpthread
```

---

## 2. Data Flow and Pipelines

### 2.1 Complete Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    Complete Data Flow                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  User Input                                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ "Minimize travel distance for 52 cities in Berlin"   │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         │                                   │
│                         ▼                                   │
│  Problem Analysis (LLM)                                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ {                                                     │  │
│  │   "problem_type": "routing",                         │  │
│  │   "decoder_strategy": "sorted_index",                │  │
│  │   "chromosome_length": 52,                           │  │
│  │   "objectives": [{"type": "minimize", "name": "tour_length"}] │
│  │ }                                                     │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         │                                   │
│                         ▼                                   │
│  Code Generation (LLM)                                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ template<typename T>                                  │  │
│  │ class TSPConfig : public BRKGAConfig<T> {            │  │
│  │     ...                                              │  │
│  │ };                                                    │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         │                                   │
│                         ▼                                   │
│  Validation                                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ ✓ Syntax check passed                                │  │
│  │ ✓ Structure validation passed                        │  │
│  │ ✓ API compliance check passed                        │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         │                                   │
│                         ▼                                   │
│  NVCC Compilation                                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ nvcc -std=c++17 -arch=sm_75 -O3 ...                  │  │
│  │ ✓ Compilation successful (5.2s)                      │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         │                                   │
│                         ▼                                   │
│  GPU Execution                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Population: 200, Generations: 1000                   │  │
│  │ GPU-resident island model: 2 islands                 │  │
│  │ Best fitness: 7544.32 (optimal: 7542)                │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 GPU Data Flow

**GPU-Resident Mode:**
```
┌─────────────────────────────────────────────────────────────┐
│                  GPU-Resident Data Flow                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  GPU Memory (stays on device)                               │
│  ┌────────────────────────────────────────────────────────┐│
│  │                                                         ││
│  │  ┌─────────────┐      ┌─────────────┐                 ││
│  │  │ Chromosomes │ ───▶ │  Fitness    │                 ││
│  │  │ (random key)│      │  Values     │                 ││
│  │  └─────────────┘      └─────────────┘                 ││
│  │         │                    │                         ││
│  │         ▼                    │                         ││
│  │  ┌─────────────┐            │                         ││
│  │  │   Sort by   │◀───────────┘                         ││
│  │  │   Fitness   │                                      ││
│  │  └─────────────┘                                      ││
│  │         │                                              ││
│  │         ▼                                              ││
│  │  ┌─────────────┐                                      ││
│  │  │  Selection  │ (Elite preserved)                    ││
│  │  └─────────────┘                                      ││
│  │         │                                              ││
│  │         ▼                                              ││
│  │  ┌─────────────┐      ┌─────────────┐                 ││
│  │  │  Crossover  │ ───▶ │   Mutants   │                 ││
│  │  │  Kernel     │      │   Kernel    │                 ││
│  │  └─────────────┘      └─────────────┘                 ││
│  │         │                    │                         ││
│  │         └────────┬───────────┘                         ││
│  │                  ▼                                     ││
│  │           New Generation                               ││
│  │                                                         ││
│  └────────────────────────────────────────────────────────┘│
│                                                              │
│  Host Memory (minimal)                                       │
│  ┌────────────────────────────────────────────────────────┐│
│  │ - Best solution (periodic sync)                        ││
│  │ - Statistics                                           ││
│  │ - Local search candidates                              ││
│  └────────────────────────────────────────────────────────┘│
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Traditional Mode (with transfers):**
```
Generation Loop:
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  Host                          Device                        │
│  ┌────────────┐               ┌────────────┐               │
│  │ Elite      │ ───────────▶ │ d_elite    │               │
│  │ Population │               └──────┬─────┘               │
│  └────────────┘                      │                      │
│                                      ▼                      │
│  ┌────────────┐               ┌────────────┐               │
│  │ Non-Elite  │ ───────────▶ │ Crossover  │               │
│  │ Population │               │ Kernel     │               │
│  └────────────┘               └──────┬─────┘               │
│                                      │                      │
│  ┌────────────┐               ┌──────▼─────┐               │
│  │ Offspring  │ ◀─────────── │ d_offspring│               │
│  │ (Host)     │               └────────────┘               │
│  └────────────┘                                             │
│                                                              │
│  ┌────────────┐               ┌────────────┐               │
│  │ Mutants    │ ◀─────────── │ Mutation   │               │
│  │ (Host)     │               │ Kernel     │               │
│  └────────────┘               └────────────┘               │
│                                                              │
│  Evaluate on Host (fitness_function lambda)                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. CUDA Implementation Details

### 3.1 Thread Organization

**Grid and Block Configuration:**
```cpp
// Standard configuration
int threads_per_block = 256;
int num_blocks = (population_size + threads_per_block - 1) / threads_per_block;

dim3 block(threads_per_block);
dim3 grid(num_blocks);

// Launch kernel
kernel<<<grid, block>>>(args...);
```

**Occupancy Considerations:**
- 256 threads/block provides good occupancy on most GPUs
- Register usage should be monitored for complex kernels
- Shared memory is not heavily used (data too large for sharing)

### 3.2 Random Number Generation

**cuRAND State Management:**
```cuda
// Initialize states (once per GPU)
__global__ void init_curand_states(curandState* states, int n, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed + idx, 0, 0, &states[idx]);
    }
}

// Usage in kernels
__global__ void crossover_kernel(..., curandState* states, ...) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_offspring) {
        curandState localState = states[idx];  // Copy to local

        // Use curand_uniform() for [0,1) random numbers
        float r = curand_uniform(&localState);

        // Use curand() for integers
        int parent = curand(&localState) % elite_size;

        states[idx] = localState;  // Save state back
    }
}
```

### 3.3 GPU Population Class (`gpu_population.cuh`)

```cpp
template<typename T>
class GPUPopulation {
private:
    // Device arrays
    T* d_chromosomes;     // Population chromosomes [pop_size × chrom_len]
    T* d_fitness;         // Fitness values [pop_size]
    int* d_indices;       // Sorted indices [pop_size]
    curandState* d_states;// RNG states [pop_size]

    // Configuration
    int pop_size;
    int chrom_len;
    int elite_size;
    int offspring_size;
    int mutant_size;
    int threads_per_block;
    bool minimize;

public:
    // Core operations
    void initialize(unsigned long seed);
    void evaluate(BRKGAConfig<T>* config);
    void select();      // GPU-based selection
    void crossover(double elite_prob);
    void mutate();

    // Migration support
    T* get_d_chromosomes() { return d_chromosomes; }
    T* get_d_fitness() { return d_fitness; }
    T get_best_fitness();

    // Host synchronization
    void sync_to_host(std::vector<Individual<T>>& individuals,
                      const std::vector<int>& component_lengths);
    void sync_individuals_from_host(const std::vector<Individual<T>>& individuals,
                                    const std::vector<int>& indices);
};
```

### 3.4 Multi-GPU Communication

**Peer-to-Peer Access:**
```cpp
void enable_peer_access() {
    for (int i = 0; i < gpu_count; i++) {
        cudaSetDevice(devices[i]);
        for (int j = 0; j < gpu_count; j++) {
            if (i != j) {
                int can_access;
                cudaDeviceCanAccessPeer(&can_access, devices[i], devices[j]);
                if (can_access) {
                    cudaDeviceEnablePeerAccess(devices[j], 0);
                }
            }
        }
    }
}
```

**Migration Implementation:**
```cpp
void perform_island_migration() {
    // Ring topology migration
    for (int i = 0; i < num_islands; i++) {
        int source = (i + num_islands - 1) % num_islands;

        // Copy elite from source to worst positions of target
        cudaMemcpy(
            islands[i]->d_chromosomes + (pop_size - migration_size) * chrom_len,
            islands[source]->d_chromosomes,
            migration_size * chrom_len * sizeof(T),
            cudaMemcpyDeviceToDevice
        );

        // Copy corresponding fitness values
        cudaMemcpy(
            islands[i]->d_fitness + (pop_size - migration_size),
            islands[source]->d_fitness,
            migration_size * sizeof(T),
            cudaMemcpyDeviceToDevice
        );
    }
}
```

---

## 4. LLM Prompt Engineering

### 4.1 Problem Analysis Prompt Template

```python
analysis_prompt = f"""
You are an expert in optimization and genetic algorithms.
Analyze the following optimization problem description and extract
structured information needed to generate a BRKGA solver.

PROBLEM DESCRIPTION:
{description}

{f"CLARIFYING Q&A: {clarifying_qa}" if clarifying_qa else ""}

{f"DATA FILE INFO: {data_context}" if data_files else ""}

Please analyze this problem and provide a structured analysis in JSON format with:

{{
  "problem_name": "short descriptive name",
  "problem_type": "routing|packing|scheduling|assignment|selection|sequencing|partitioning|custom",
  "domain": "detailed problem description",
  "chromosome_length": <integer>,
  "decision_variables": [...],
  "objectives": [...],
  "constraints": [...],
  "decoder_strategy": "sorted_index|threshold|assignment|priority|custom",
  "decoder_rationale": "why this decoder was chosen",
  "local_search_recommended": <boolean>,
  "local_search_type": "none|two_opt|three_opt|swap|shift|insert|exchange",
  ...
}}

Key considerations:
1. Chromosome length = total number of decision variables
2. Choose decoder strategy based on problem type
3. For local search, recommend based on problem type:
   - routing: two_opt or three_opt
   - scheduling: shift or swap
   - selection: swap
"""
```

### 4.2 Code Generation Prompt Template

```python
generation_prompt = f"""
Generate a complete BRKGA configuration file for the following optimization problem.

PROBLEM SPECIFICATION:
=====================
Name: {problem.problem_name}
Type: {problem.problem_type.value}
Description: {problem.domain}

DECISION VARIABLES:
{[f"- {dv.name}: {dv.count} variables ({dv.semantics})" for dv in problem.decision_variables]}

Chromosome Length: {problem.chromosome_length}

OBJECTIVES:
{[f"- {obj.type.upper()} {obj.name}: {obj.description}" for obj in problem.objectives]}

CONSTRAINTS:
{constraints_section}

DECODER STRATEGY: {problem.decoder_strategy.value}
Rationale: {problem.decoder_rationale}

BRKGA HYPERPARAMETERS:
{hyperparameters_section}

GENERATION REQUIREMENTS:
========================
1. Create a complete, compilable C++ header file
2. Follow the BRKGA framework patterns exactly
3. Include: #include "../../brkga/core/config.hpp"
4. Function signatures MUST be:
   - fitness_function: [this](const Individual<T>& individual) -> T
   - decoder: [this](const Individual<T>& individual) -> std::vector<std::vector<T>>
   - comparator: [](T a, T b) -> bool

CRITICAL REQUIREMENTS:
- Access chromosome via: individual.get_chromosome()
- For minimization: comparator = [](T a, T b) {{ return a < b; }};
- For maximization: comparator = [](T a, T b) {{ return a > b; }};
- DO NOT use negative fitness values!

Generate the complete code now. Provide ONLY the C++ code, no explanations.
"""
```

### 4.3 Context Package Structure

The context file (`context_for_llm_full.txt`) contains:

1. **Framework Overview** (~500 words)
   - What BRKGA is
   - How the framework works
   - Key concepts

2. **API Reference** (~2000 words)
   - BRKGAConfig class interface
   - Individual class interface
   - Required function signatures
   - Configuration parameters

3. **Code Examples** (~3000 words)
   - Complete TSP configuration
   - Complete Knapsack configuration
   - Local search integration example

4. **Critical Rules** (~1000 words)
   - Common mistakes and how to avoid them
   - Required patterns
   - Forbidden patterns

---

## 5. Configuration System

### 5.1 Configuration Hierarchy

```
BRKGAConfig<T> (Base)
    │
    ├── Single-Objective Problems
    │   ├── TSPConfig
    │   ├── KnapsackConfig
    │   ├── SchedulingConfig
    │   └── Custom...
    │
    └── Multi-Objective Problems
        ├── BiObjectiveTSPConfig
        ├── PortfolioConfig
        └── Custom...
```

### 5.2 Configuration Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `population_size` | int | 100 | 50-5000 | Number of individuals |
| `elite_size` | int | 15 | 5-25% of pop | Number of elites |
| `mutant_size` | int | 15 | 5-25% of pop | Number of mutants |
| `elite_prob` | double | 0.7 | 0.5-0.9 | Bias toward elite parent |
| `max_generations` | int | 500 | 100-10000 | Termination criterion |
| `threads_per_block` | int | 256 | 128-1024 | CUDA threads per block |
| `num_components` | int | 1 | 1-10 | Chromosome components |
| `num_objectives` | int | 1 | 1-5 | Number of objectives |

### 5.3 Factory Pattern for Configurations

```cpp
template<typename T>
class TSPConfig : public BRKGAConfig<T> {
public:
    // Constructor from file
    TSPConfig(const std::string& tsp_file) { ... }

    // Factory method for file-based creation
    static std::unique_ptr<TSPConfig<T>> create_from_file(const std::string& path) {
        return std::make_unique<TSPConfig<T>>(path);
    }

    // Factory method for default test instance
    static std::unique_ptr<TSPConfig<T>> create_default() {
        // Return small test instance
        std::vector<std::vector<T>> distances = { ... };
        return std::make_unique<TSPConfig<T>>(distances);
    }
};
```

---

## 6. Memory Management

### 6.1 Host Memory

**Population Memory:**
```
Per Individual:
- components: num_components × component_lengths × sizeof(T)
- objectives: num_objectives × sizeof(T)
- metadata: ~32 bytes

Total Population:
≈ population_size × (chromosome_length × sizeof(T) + 64) bytes
```

**Example for TSP berlin52:**
- Chromosome length: 52
- Population size: 200
- Memory per individual: 52 × 4 + 64 = 272 bytes
- Total population: 200 × 272 = 54.4 KB

### 6.2 GPU Memory

**Per GPU Workspace:**
```cpp
struct GPUWorkspace {
    T* d_population;      // pop_size × chrom_len × sizeof(T)
    T* d_elite_pop;       // elite_size × chrom_len × sizeof(T)
    T* d_non_elite_pop;   // (pop_size - elite_size) × chrom_len × sizeof(T)
    T* d_offspring;       // offspring_size × chrom_len × sizeof(T)
    T* d_mutants;         // mutant_size × chrom_len × sizeof(T)
    curandState* d_states; // pop_size × sizeof(curandState) ≈ 48 bytes each

    // Multi-objective only
    T* d_objectives;      // pop_size × num_objectives × sizeof(T)
    int* d_ranks;         // pop_size × sizeof(int)
    T* d_crowding_dist;   // pop_size × sizeof(T)
};
```

**Memory Calculation:**
```
For single-objective TSP berlin52 (pop=200, chrom=52):

d_population:    200 × 52 × 4   =  41.6 KB
d_elite_pop:      30 × 52 × 4   =   6.2 KB
d_non_elite_pop: 170 × 52 × 4   =  35.4 KB
d_offspring:     140 × 52 × 4   =  29.1 KB
d_mutants:        30 × 52 × 4   =   6.2 KB
d_states:        200 × 48       =   9.6 KB
─────────────────────────────────────────
Total per GPU:                  ≈ 128 KB
```

### 6.3 Memory Optimization Strategies

1. **Lazy Allocation**: Only allocate when needed
2. **Buffer Reuse**: Reuse offspring buffer for mutants when possible
3. **Pinned Memory**: Use cudaMallocHost for faster transfers
4. **Unified Memory**: Optional for debugging (slower)

---

## 7. Performance Analysis

### 7.1 Time Complexity

| Operation | CPU Complexity | GPU Complexity |
|-----------|---------------|----------------|
| Initialization | O(N × L) | O(L) per thread |
| Crossover | O(N × L) | O(L) per thread |
| Mutation | O(M × L) | O(L) per thread |
| Evaluation | O(N × F) | O(F) per thread |
| Selection (sort) | O(N log N) | O(N log N) |
| Non-dominated sort | O(M × N²) | O(M × N²) |

Where: N=population, L=chromosome length, M=objectives, F=fitness complexity

### 7.2 Speedup Analysis

**Expected GPU Speedup Factors:**
- Population initialization: 10-50x
- Crossover: 20-100x
- Mutation: 20-100x
- Fitness evaluation: Depends on complexity (1-1000x)

**Overhead Sources:**
- Host-device transfers: ~1-10 ms per transfer
- Kernel launch: ~5-50 μs per launch
- Synchronization: ~1-100 μs

### 7.3 Bottleneck Identification

From the system reminder showing:
```
Time breakdown:
  Evaluation: 515.99s (100.00%)
  Selection (GPU sort): 0.00s (0.00%)
  Crossover: 0.00s (0.00%)
  Mutation: 0.00s (0.00%)
```

**Analysis:**
- Fitness evaluation is the bottleneck (running on host)
- Genetic operations are fast (GPU-accelerated)
- **Solution**: Move fitness function to GPU or optimize host computation

### 7.4 Performance Monitoring

```cpp
struct PerformanceStats {
    double total_crossover_time;
    double total_mutation_time;
    double total_evaluation_time;
    double total_sorting_time;
    double total_local_search_time;
    int operations_count;

    void print_summary() const {
        double total = total_crossover_time + total_mutation_time +
                      total_evaluation_time + total_sorting_time;

        std::cout << "Performance Breakdown:" << std::endl;
        std::cout << "  Evaluation: " << total_evaluation_time
                  << "s (" << (total_evaluation_time/total*100) << "%)" << std::endl;
        // ... etc
    }
};
```

---

## 8. Error Handling and Recovery

### 8.1 CUDA Error Checking

```cpp
namespace CudaUtils {
    inline void check_cuda_error(cudaError_t error, const char* message) {
        if (error != cudaSuccess) {
            throw std::runtime_error(
                std::string(message) + ": " + cudaGetErrorString(error)
            );
        }
    }

    inline void sync_and_check(const char* operation) {
        cudaError_t error = cudaDeviceSynchronize();
        check_cuda_error(error, operation);
    }
}

// Usage
cudaMalloc(&d_data, size);
CudaUtils::check_cuda_error(cudaGetLastError(), "Failed to allocate GPU memory");
```

### 8.2 LLM Error Recovery

```python
class CodeGenerator:
    def refine_config(self, config_path: str, error_message: str,
                     problem: ProblemStructure) -> str:
        """Refine configuration based on compilation errors."""

        with open(config_path, 'r') as f:
            current_code = f.read()

        refinement_prompt = f"""
        The following BRKGA configuration has errors.
        Please fix the issues while maintaining the structure.

        CURRENT CODE:
        ```cpp
        {current_code}
        ```

        ERROR MESSAGE:
        {error_message}

        Provide the corrected version of the ENTIRE config file.
        """

        response = self.client.messages.create(...)
        refined_code = self._extract_code(response.content[0].text)

        refined_path = config_path.replace('.hpp', '_refined.hpp')
        with open(refined_path, 'w') as f:
            f.write(refined_code)

        return refined_path
```

### 8.3 Graceful Degradation

```cpp
void Solver::determine_execution_strategy() {
    if (gpu_count == 0) {
        execution_mode = "cpu";
        use_gpu = false;
        if (verbose) {
            std::cout << "No GPU available, using CPU mode" << std::endl;
        }
    } else if (config->population_size < 100) {
        execution_mode = "cpu";
        use_gpu = false;
        if (verbose) {
            std::cout << "Population too small for GPU, using CPU" << std::endl;
        }
    } else if (use_gpu_resident && config->is_multi_objective()) {
        use_gpu_resident = false;
        if (verbose) {
            std::cout << "GPU-resident not supported for multi-objective, "
                      << "falling back to transfer mode" << std::endl;
        }
    }
}
```

---

## 9. Testing Framework

### 9.1 Unit Tests

**Test Categories:**
1. Individual operations (flatten/unflatten)
2. Population operations (sort, select)
3. Genetic operators (crossover, mutation)
4. CUDA kernels (correctness and performance)
5. Local search operators
6. Configuration validation

### 9.2 Integration Tests

**Test Scenarios:**
1. End-to-end TSP solving
2. LLM problem analysis accuracy
3. Code generation and compilation
4. Multi-GPU synchronization
5. Island migration correctness

### 9.3 Benchmark Tests

```bash
# Run TSP benchmark
./benchmark_tsp --instance berlin52 --population 200 --generations 1000

# Run fitness computation test
./test_tspj_fitness --verify-optimal

# Run GPU vs CPU comparison
./solver --benchmark-modes
```

---

## 10. Deployment Guide

### 10.1 System Requirements

**Minimum:**
- NVIDIA GPU with Compute Capability 3.5+
- CUDA Toolkit 11.0+
- g++ 9+ or clang++ 10+
- Python 3.8+
- 4 GB RAM

**Recommended:**
- NVIDIA GPU with Compute Capability 7.5+ (RTX 2000+)
- CUDA Toolkit 12.0+
- g++ 11+
- Python 3.10+
- 16 GB RAM
- Multiple GPUs for island model

### 10.2 Installation

```bash
# Clone repository
git clone https://github.com/user/llm_brkga.git
cd llm_brkga

# Set up Python environment
python -m venv venv
source venv/bin/activate
pip install anthropic numpy

# Set API key
export ANTHROPIC_API_KEY="your-key-here"

# Build framework
make all

# Run tests
./test_berlin52_fitness
```

### 10.3 Configuration

**Environment Variables:**
```bash
ANTHROPIC_API_KEY=...     # Required for LLM features
CUDA_VISIBLE_DEVICES=0,1  # Select GPUs
QUICK_TEST=10             # Limit generations for testing
```

**Runtime Options:**
```bash
./solver \
    --population 500 \
    --generations 1000 \
    --elite-pct 0.15 \
    --mutant-pct 0.15 \
    --elite-prob 0.7 \
    --gpu-resident \
    --verbose
```

---

## Appendix: Glossary

| Term | Definition |
|------|------------|
| **BRKGA** | Biased Random-Key Genetic Algorithm |
| **Chromosome** | Vector of random keys representing a solution |
| **Decoder** | Function converting random keys to problem solution |
| **Elite** | Top-performing individuals preserved across generations |
| **Fitness** | Objective function value for a solution |
| **Island Model** | Parallel GA with multiple sub-populations |
| **Migration** | Exchange of individuals between islands |
| **Mutant** | Completely random new individual |
| **NSGA-II** | Non-dominated Sorting Genetic Algorithm II |
| **Pareto Front** | Set of non-dominated solutions |
| **Random Key** | Real number in [0,1] used in chromosome |

---

*Technical Architecture Document for LLM-BRKGA Framework*
*Version 1.0*
