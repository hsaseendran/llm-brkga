# LLM-BRKGA: An LLM-Powered GPU-Accelerated Biased Random-Key Genetic Algorithm Framework

## Thesis Documentation

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Introduction and Motivation](#2-introduction-and-motivation)
3. [Background and Related Work](#3-background-and-related-work)
4. [System Architecture](#4-system-architecture)
5. [BRKGA Framework](#5-brkga-framework)
6. [GPU Acceleration](#6-gpu-acceleration)
7. [LLM Integration](#7-llm-integration)
8. [Local Search Integration](#8-local-search-integration)
9. [Supported Problem Types](#9-supported-problem-types)
10. [Implementation Details](#10-implementation-details)
11. [Experimental Evaluation](#11-experimental-evaluation)
12. [Conclusions and Future Work](#12-conclusions-and-future-work)

---

## 1. Executive Summary

This project presents **LLM-BRKGA**, a novel framework that combines three powerful paradigms in optimization:

1. **Biased Random-Key Genetic Algorithms (BRKGA)** - A robust metaheuristic for combinatorial optimization
2. **GPU Acceleration** - Massively parallel computation using CUDA for high performance
3. **Large Language Models (LLMs)** - Natural language problem specification and automatic solver generation

The framework enables users to describe optimization problems in natural language, which are then automatically analyzed, converted to working BRKGA solvers, compiled, and executed on GPU hardware. This dramatically reduces the barrier to entry for applying sophisticated optimization techniques to real-world problems.

### Key Contributions

- **Natural Language Problem Specification**: Users describe problems in plain English; the system automatically generates optimized solver code
- **GPU-Accelerated Genetic Operations**: Crossover, mutation, and population management run on NVIDIA GPUs with multi-GPU support
- **Island Model Parallelism**: Multiple sub-populations evolve independently on different GPUs with periodic migration
- **GPU-Resident Execution**: Zero-copy mode eliminates host-device data transfers for maximum performance
- **Integrated Local Search**: Hybrid approach combining genetic search with problem-specific local improvement
- **Multi-Objective Support**: NSGA-II implementation for Pareto optimization
- **Automatic Code Generation**: LLM generates complete, compilable C++/CUDA configurations

---

## 2. Introduction and Motivation

### 2.1 The Challenge of Combinatorial Optimization

Combinatorial optimization problems are ubiquitous in industry and science:
- Supply chain and logistics (vehicle routing, scheduling)
- Manufacturing (job shop scheduling, resource allocation)
- Finance (portfolio optimization)
- Telecommunications (network design)
- Bioinformatics (sequence alignment)

These problems are often NP-hard, meaning exact solutions become computationally intractable as problem size grows. Metaheuristics like genetic algorithms provide high-quality approximate solutions in reasonable time.

### 2.2 Barriers to Adoption

Despite their effectiveness, metaheuristic solvers face adoption barriers:

1. **Implementation Complexity**: Developing efficient GA implementations requires expertise in algorithm design, data structures, and performance optimization
2. **Problem-Specific Customization**: Each problem requires custom fitness functions, decoders, and constraint handling
3. **Performance Optimization**: Leveraging modern hardware (GPUs) requires additional specialized knowledge
4. **Parameter Tuning**: GAs have many hyperparameters that significantly affect performance

### 2.3 Our Solution

LLM-BRKGA addresses these barriers by:

1. **Abstracting complexity**: The BRKGA framework handles all algorithmic details
2. **Natural language interface**: Users describe problems without coding
3. **Automatic optimization**: GPU acceleration is automatic and transparent
4. **Intelligent defaults**: LLM estimates appropriate parameters based on problem characteristics

---

## 3. Background and Related Work

### 3.1 Genetic Algorithms

Genetic Algorithms (GAs) are population-based metaheuristics inspired by biological evolution. A population of candidate solutions evolves through:

- **Selection**: Fitter individuals are more likely to reproduce
- **Crossover**: Combining genetic material from two parents
- **Mutation**: Random changes to maintain diversity

### 3.2 Biased Random-Key Genetic Algorithm (BRKGA)

BRKGA, introduced by Gonçalves and Resende (2011), uses random keys (real numbers in [0,1]) as the genetic representation. Key innovations:

1. **Random-Key Representation**: Chromosomes are vectors of random numbers, decoded into solutions
2. **Biased Crossover**: Offspring inherit genes from the elite parent with higher probability
3. **Decoder-Based Approach**: Problem-specific decoders convert random keys to feasible solutions
4. **Elitism**: Best solutions are preserved across generations

**BRKGA Population Structure:**
```
Population (N individuals)
├── Elite (top p_e% by fitness)
├── Non-Elite (remaining individuals)
└── Mutants (p_m% new random individuals each generation)

Offspring = N - Elite - Mutants
```

### 3.3 GPU Computing for Evolutionary Algorithms

GPUs offer massive parallelism well-suited to evolutionary computation:
- **Population Parallelism**: Each individual can be processed independently
- **Fitness Evaluation**: Often the most expensive operation, highly parallelizable
- **Genetic Operations**: Crossover and mutation can be performed in parallel

### 3.4 Large Language Models for Code Generation

Recent LLMs (GPT-4, Claude) demonstrate remarkable ability to:
- Understand natural language specifications
- Generate working code in multiple languages
- Reason about algorithmic requirements
- Debug and refine code based on error messages

---

## 4. System Architecture

### 4.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     LLM-BRKGA System                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │  Natural Language │───▶│  Problem Analyzer │                  │
│  │  Problem Input    │    │  (LLM-powered)   │                   │
│  └──────────────────┘    └────────┬─────────┘                   │
│                                   │                              │
│                                   ▼                              │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │  Data Files      │───▶│  Code Generator  │                   │
│  │  (TSP, CSV, etc) │    │  (LLM-powered)   │                   │
│  └──────────────────┘    └────────┬─────────┘                   │
│                                   │                              │
│                                   ▼                              │
│                          ┌──────────────────┐                   │
│                          │   Validator      │                   │
│                          │  (Syntax check)  │                   │
│                          └────────┬─────────┘                   │
│                                   │                              │
│                                   ▼                              │
│                          ┌──────────────────┐                   │
│                          │  NVCC Compiler   │                   │
│                          │  (CUDA C++)      │                   │
│                          └────────┬─────────┘                   │
│                                   │                              │
│                                   ▼                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              GPU-Accelerated BRKGA Solver                 │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐     │  │
│  │  │  GPU 0  │  │  GPU 1  │  │  GPU 2  │  │  GPU N  │     │  │
│  │  │ Island  │◀─▶│ Island  │◀─▶│ Island  │◀─▶│ Island  │     │  │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                   │                              │
│                                   ▼                              │
│                          ┌──────────────────┐                   │
│                          │  Optimization    │                   │
│                          │  Results         │                   │
│                          └──────────────────┘                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Component Overview

| Component | Language | Description |
|-----------|----------|-------------|
| LLM Solver | Python | Orchestrates problem analysis and code generation |
| Problem Analyzer | Python | Extracts structured problem information using LLM |
| Code Generator | Python | Generates C++ BRKGA configurations |
| Validator | Python | Validates generated code structure |
| BRKGA Framework | C++/CUDA | Core genetic algorithm implementation |
| Solver | C++/CUDA | GPU-accelerated optimization engine |
| Local Search | C++ | Problem-specific local improvement operators |

### 4.3 Directory Structure

```
llm_brkga/
├── brkga/                      # Core BRKGA framework
│   ├── core/                   # Core algorithm components
│   │   ├── config.hpp          # Base configuration class
│   │   ├── solver.hpp          # Main solver with GPU support
│   │   ├── population.hpp      # Population management
│   │   ├── individual.hpp      # Individual representation
│   │   ├── genetic_operators.hpp # Crossover, mutation, selection
│   │   ├── cuda_kernels.cuh    # CUDA kernel implementations
│   │   ├── gpu_population.cuh  # GPU-resident population
│   │   ├── local_search.hpp    # Local search base class
│   │   ├── local_search_manager.hpp # Local search coordination
│   │   └── multi_gpu_manager.hpp # Multi-GPU coordination
│   ├── configs/                # Problem-specific configurations
│   │   └── tsp_config_impl.cu  # TSP implementation example
│   ├── data/                   # Benchmark problem instances
│   │   └── berlin52.tsp        # TSP benchmark
│   └── utils/                  # Utility functions
│       └── timer.hpp           # Performance timing
│
├── llm_solver/                 # LLM-powered solver system
│   ├── core/                   # Core Python components
│   │   ├── llm_brkga_solver.py # Main orchestrator
│   │   ├── problem_analyzer.py # LLM problem analysis
│   │   ├── code_generator.py   # LLM code generation
│   │   ├── validator.py        # Code validation
│   │   ├── execution_manager.py # Compilation and execution
│   │   ├── problem_structures.py # Data structures
│   │   └── data_parser.py      # Data file parsing
│   ├── context/                # LLM context package
│   │   └── context_for_llm_full.txt # Framework documentation for LLM
│   ├── generated/              # Generated solver configurations
│   └── results/                # Optimization results
│
├── Makefile                    # Build system
└── docs/                       # Documentation
```

---

## 5. BRKGA Framework

### 5.1 Random-Key Representation

Each individual in BRKGA is represented as a vector of random keys:

```
Individual = [r₁, r₂, r₃, ..., rₙ] where rᵢ ∈ [0, 1]
```

This representation is:
- **Problem-independent**: Same genetic operators work for all problems
- **Implicitly feasible**: Decoders convert keys to valid solutions
- **Continuous**: Enables smooth crossover without repair

### 5.2 Decoder Strategies

The decoder is the critical problem-specific component that converts random keys to solutions:

| Strategy | Description | Use Cases |
|----------|-------------|-----------|
| **Sorted Index** | Sort keys to get permutation | TSP, VRP, scheduling |
| **Threshold** | Select items where key > threshold | Knapsack, selection |
| **Assignment** | Discretize keys to assignments | Bin packing, clustering |
| **Priority** | Use keys as construction priorities | Scheduling, sequencing |

**Example: TSP Decoder (Sorted Index)**
```
Random Keys: [0.7, 0.2, 0.9, 0.1, 0.5]
Sorted Index: [3, 1, 4, 0, 2]  (indices that would sort the keys)
Tour: City 3 → City 1 → City 4 → City 0 → City 2
```

### 5.3 Genetic Operators

#### 5.3.1 Biased Crossover

Offspring inherit genes from elite parent with probability ρ_e (typically 0.7):

```
for each gene i:
    if random() < elite_prob:
        offspring[i] = elite_parent[i]
    else:
        offspring[i] = non_elite_parent[i]
```

#### 5.3.2 Mutation

Mutants are completely random individuals (not modified existing ones):

```
mutant = [random(), random(), ..., random()]
```

This maintains population diversity without complex mutation operators.

#### 5.3.3 Selection

Elitist selection preserves best individuals:
1. Rank population by fitness
2. Copy top p_e% (elite) to next generation
3. Fill remaining with offspring and mutants

### 5.4 Multi-Objective Support (NSGA-II)

For problems with multiple objectives, the framework implements NSGA-II:

1. **Non-Dominated Sorting**: Partition population into Pareto fronts
2. **Crowding Distance**: Measure solution spread within fronts
3. **Tournament Selection**: Compare by (rank, crowding distance)
4. **Pareto Archive**: Maintain set of non-dominated solutions

---

## 6. GPU Acceleration

### 6.1 GPU Architecture Overview

NVIDIA GPUs consist of:
- **Streaming Multiprocessors (SMs)**: Independent processing units
- **CUDA Cores**: Execute parallel threads
- **Global Memory**: Large but high-latency
- **Shared Memory**: Fast, per-block cache
- **Registers**: Fastest, per-thread storage

### 6.2 CUDA Kernel Implementations

#### 6.2.1 Population Initialization Kernel

```cuda
__global__ void initialize_population_kernel(T* population,
                                            curandState* states,
                                            int pop_size,
                                            int chrom_length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pop_size) {
        curand_init(1234 + idx, 0, 0, &states[idx]);
        for (int i = 0; i < chrom_length; i++) {
            population[idx * chrom_length + i] = curand_uniform(&states[idx]);
        }
    }
}
```

Each thread initializes one individual with random keys using cuRAND.

#### 6.2.2 Crossover Kernel

```cuda
__global__ void crossover_kernel(T* elite_pop, T* non_elite_pop,
                                T* offspring, curandState* states,
                                int num_offspring, int chrom_length,
                                double elite_prob, int elite_size,
                                int non_elite_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_offspring) {
        curandState localState = states[idx];

        int elite_parent = curand(&localState) % elite_size;
        int non_elite_parent = curand(&localState) % non_elite_size;

        for (int i = 0; i < chrom_length; i++) {
            if (curand_uniform(&localState) < elite_prob) {
                offspring[idx * chrom_length + i] =
                    elite_pop[elite_parent * chrom_length + i];
            } else {
                offspring[idx * chrom_length + i] =
                    non_elite_pop[non_elite_parent * chrom_length + i];
            }
        }
        states[idx] = localState;
    }
}
```

Each thread generates one offspring via biased uniform crossover.

#### 6.2.3 Mutation Kernel

```cuda
__global__ void mutation_kernel(T* population, curandState* states,
                               int num_mutants, int chrom_length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_mutants) {
        curandState localState = states[idx];
        for (int i = 0; i < chrom_length; i++) {
            population[idx * chrom_length + i] = curand_uniform(&localState);
        }
        states[idx] = localState;
    }
}
```

### 6.3 Execution Modes

The solver supports three execution modes:

#### 6.3.1 CPU Mode
- Traditional sequential execution
- Used when no GPU available or population too small
- Useful for debugging and baseline comparisons

#### 6.3.2 GPU Mode (with transfers)
- Genetic operations on GPU
- Population transferred between host and device each generation
- Good for moderate population sizes

#### 6.3.3 GPU-Resident Mode (Zero-Copy)
- Population stays on GPU memory permanently
- Eliminates host-device transfer overhead
- Maximum performance for large populations

### 6.4 Multi-GPU Island Model

For systems with multiple GPUs, the framework implements an island model:

```
┌─────────────────────────────────────────────────────────┐
│                    Island Model                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   GPU 0              GPU 1              GPU 2           │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐      │
│  │ Island 0 │      │ Island 1 │      │ Island 2 │      │
│  │          │      │          │      │          │      │
│  │ Pop: N/3 │◀────▶│ Pop: N/3 │◀────▶│ Pop: N/3 │      │
│  │          │      │          │      │          │      │
│  └──────────┘      └──────────┘      └──────────┘      │
│                                                          │
│  Migration: Every M generations, K elite individuals    │
│  are exchanged between neighboring islands              │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Island Model Parameters:**
- `migration_frequency`: Generations between migrations (default: 25)
- `migration_size`: Number of elites exchanged (default: 5)
- Ring topology: Island i exchanges with islands (i-1) and (i+1)

**Benefits:**
- Independent evolution maintains diversity
- Migration shares good solutions
- Linear scalability with GPU count
- Robust against premature convergence

### 6.5 GPU Memory Management

The `GPUWorkspace` structure manages memory for each GPU:

| Buffer | Size | Purpose |
|--------|------|---------|
| `d_population` | N × L × sizeof(T) | Full population chromosomes |
| `d_elite_pop` | E × L × sizeof(T) | Elite individuals for crossover |
| `d_non_elite_pop` | (N-E) × L × sizeof(T) | Non-elite for crossover |
| `d_offspring` | O × L × sizeof(T) | Generated offspring |
| `d_mutants` | M × L × sizeof(T) | Random mutants |
| `d_states` | N × sizeof(curandState) | RNG states |
| `d_objectives` | N × K × sizeof(T) | Multi-objective values |
| `d_ranks` | N × sizeof(int) | Pareto ranks |
| `d_crowding_dist` | N × sizeof(T) | Crowding distances |

Where: N=population, L=chromosome length, E=elite, O=offspring, M=mutants, K=objectives

### 6.6 Performance Optimization

#### 6.6.1 Coalesced Memory Access
Chromosomes stored in row-major order for coalesced global memory reads.

#### 6.6.2 Thread Block Configuration
```cpp
dim3 block(256);  // 256 threads per block (good occupancy)
dim3 grid = (total_threads + 255) / 256;
```

#### 6.6.3 Peer-to-Peer Access
When GPUs support P2P, direct memory access between devices:
```cpp
cudaDeviceEnablePeerAccess(other_device, 0);
```

---

## 7. LLM Integration

### 7.1 Problem Analysis Pipeline

The Problem Analyzer uses Claude (Sonnet) to extract structured information:

```python
class ProblemAnalyzer:
    def analyze_problem(self, description, clarifying_qa, data_files):
        # Build analysis prompt with context
        prompt = self._build_analysis_prompt(description, clarifying_qa, data_metadata)

        # Call LLM for analysis
        response = self.client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=4000,
            temperature=0.2,  # Low temperature for consistency
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse JSON response to ProblemStructure
        return self._parse_analysis(response.content[0].text)
```

### 7.2 Problem Structure Extraction

The LLM extracts:

```python
@dataclass
class ProblemStructure:
    problem_name: str              # "Traveling Salesman Problem"
    problem_type: ProblemType      # ROUTING, PACKING, SCHEDULING, etc.
    domain: str                    # Detailed description
    chromosome_length: int         # Number of decision variables
    decision_variables: List[DecisionVariable]  # What each gene represents
    objectives: List[Objective]    # Minimize/maximize what
    constraints: List[Constraint]  # Hard/soft constraints
    decoder_strategy: DecoderStrategy  # SORTED_INDEX, THRESHOLD, etc.
    local_search_recommended: bool # Whether local search helps
    local_search_type: LocalSearchType  # TWO_OPT, SWAP, etc.
```

### 7.3 Code Generation

The Code Generator creates complete C++ configurations:

```python
class CodeGenerator:
    def generate_config(self, problem, output_path, hyperparameters):
        # Build generation prompt with BRKGA context
        prompt = self._build_generation_prompt(problem, hyperparameters)

        # Call LLM with framework documentation as system prompt
        response = self.client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=8000,
            temperature=0.1,  # Very low for consistent code
            system=self.context,  # BRKGA framework documentation
            messages=[{"role": "user", "content": prompt}]
        )

        # Extract and save C++ code
        code = self._extract_code(response.content[0].text)
        with open(output_path, 'w') as f:
            f.write(code)
```

### 7.4 Generated Configuration Structure

The LLM generates configurations following this template:

```cpp
template<typename T>
class ProblemConfig : public BRKGAConfig<T> {
private:
    // Problem-specific data
    std::vector<std::vector<T>> distance_matrix;
    int num_cities;

public:
    ProblemConfig(const std::string& data_file)
        : BRKGAConfig<T>(chromosome_length) {

        // Load problem data
        load_data(data_file);

        // Set BRKGA parameters
        this->population_size = 200;
        this->elite_size = 30;
        this->mutant_size = 30;
        this->elite_prob = 0.7;
        this->max_generations = 1000;

        // Define fitness function
        this->fitness_function = [this](const Individual<T>& ind) {
            return calculate_tour_length(ind);
        };

        // Define decoder
        this->decoder = [this](const Individual<T>& ind) {
            return decode_to_tour(ind);
        };

        // Define comparator (minimization)
        this->comparator = [](T a, T b) { return a < b; };

        // Configure GPU
        this->threads_per_block = 256;
        update_cuda_grid_size();
    }

    // Decoder: Random keys → Tour
    std::vector<std::vector<T>> decode_to_tour(const Individual<T>& ind) {
        auto& keys = ind.get_chromosome();
        std::vector<std::pair<T, int>> indexed_keys;
        for (int i = 0; i < num_cities; i++) {
            indexed_keys.emplace_back(keys[i], i);
        }
        std::sort(indexed_keys.begin(), indexed_keys.end());

        std::vector<T> tour;
        for (auto& [key, city] : indexed_keys) {
            tour.push_back(static_cast<T>(city));
        }
        return {tour};
    }

    // Fitness: Tour length
    T calculate_tour_length(const Individual<T>& ind) {
        auto decoded = decode_to_tour(ind);
        auto& tour = decoded[0];
        T total = 0;
        for (size_t i = 0; i < tour.size(); i++) {
            int from = static_cast<int>(tour[i]);
            int to = static_cast<int>(tour[(i + 1) % tour.size()]);
            total += distance_matrix[from][to];
        }
        return total;
    }
};
```

### 7.5 Iterative Refinement

The system supports error-driven refinement:

```python
for iteration in range(max_iterations):
    # Generate code
    config_path = generator.generate_config(problem, output_path)

    # Validate
    valid, results = validator.full_validation(config_path, problem)
    if not valid:
        continue  # Will refine based on errors

    # Compile
    compilation_result = executor.compile_solver(config_path, executable_path)
    if not compilation_result.success:
        # Collect errors for refinement
        config_path = generator.refine_config(config_path, error_msg, problem)
        continue

    # Test execution
    test_result = executor.run_quick_test(executable_path)
    if test_result.success:
        break  # Success!
```

---

## 8. Local Search Integration

### 8.1 Hybrid Genetic Algorithm

Local search improves solution quality by exploiting neighborhood structure:

```
BRKGA Evolution (Exploration)
        │
        ▼
┌───────────────────┐
│  Local Search     │  ← Applied to elite individuals
│  (Exploitation)   │
└───────────────────┘
        │
        ▼
    Improved Solution
```

### 8.2 Local Search Base Class

```cpp
template<typename T>
class LocalSearch {
protected:
    int max_iterations;
    double improvement_threshold;
    std::function<bool(T, T)> comparator;  // For min/max problems

public:
    // Main interface
    virtual Individual<T> improve(const Individual<T>& individual) = 0;
    virtual bool should_apply(int generation, const Individual<T>& individual,
                             const std::vector<Individual<T>>& population) = 0;
    virtual void configure(const std::map<std::string, std::string>& params) = 0;

    // Statistics tracking
    mutable int applications_count;
    mutable int improvements_count;
    mutable double total_improvement;
    mutable double total_time_ms;
};
```

### 8.3 Application Strategies

| Strategy | Description |
|----------|-------------|
| `DISABLED` | No local search |
| `BEST_ONLY` | Apply only to best individual |
| `ELITE_ONLY` | Apply to all elite individuals |
| `RANDOM_SAMPLE` | Apply to random subset |
| `ALL_INDIVIDUALS` | Apply to entire population |
| `STAGNATION_ONLY` | Apply when no improvement for N generations |
| `ADAPTIVE` | Adjust based on improvement rate |

### 8.4 Timing Options

| Timing | When Applied |
|--------|--------------|
| `POST_CROSSOVER` | After generating offspring |
| `POST_MUTATION` | After generating mutants |
| `POST_EVALUATION` | After fitness evaluation |
| `END_GENERATION` | At end of each generation |
| `FINAL_POLISH` | Only at the very end |

### 8.5 Local Search Operators

| Operator | Problem Types | Description |
|----------|---------------|-------------|
| **2-opt** | TSP, VRP | Reverse segment between two edges |
| **3-opt** | TSP, VRP | More complex edge reconnections |
| **Swap** | Scheduling, Selection | Exchange two elements |
| **Shift** | Scheduling | Move element to new position |
| **Insert** | Sequencing | Insert element at different location |
| **Exchange** | Assignment | Swap assignments between groups |

### 8.6 Local Search Manager

Coordinates multiple local search operators:

```cpp
template<typename T>
class LocalSearchManager {
private:
    std::vector<std::unique_ptr<LocalSearch<T>>> local_searches;
    LocalSearchConfig<T> config;

public:
    // Apply local search based on configured strategy
    std::vector<Individual<T>> apply_to_population(
        const std::vector<Individual<T>>& population,
        int generation) {

        auto indices = select_individuals_for_search(population, generation);
        std::vector<Individual<T>> improved = population;

        for (int idx : indices) {
            for (auto& ls : local_searches) {
                if (ls->should_apply(generation, improved[idx], population)) {
                    improved[idx] = ls->apply(improved[idx]);
                }
            }
        }
        return improved;
    }

    // Intensive search for final polish
    Individual<T> intensive_search(const Individual<T>& individual) {
        Individual<T> current = individual;
        bool improved;
        do {
            improved = false;
            for (auto& ls : local_searches) {
                Individual<T> result = ls->apply(current);
                if (comparator(result.fitness, current.fitness)) {
                    current = result;
                    improved = true;
                }
            }
        } while (improved);
        return current;
    }
};
```

---

## 9. Supported Problem Types

### 9.1 Problem Type Classification

| Type | Description | Decoder Strategy |
|------|-------------|------------------|
| **Routing** | TSP, VRP, path problems | Sorted Index |
| **Packing** | Bin packing, knapsack | Threshold |
| **Scheduling** | Job shop, flow shop | Priority |
| **Assignment** | Task assignment, clustering | Assignment |
| **Selection** | Feature selection, subset sum | Threshold |
| **Sequencing** | Order-dependent problems | Sorted Index |
| **Partitioning** | Graph partitioning | Assignment |

### 9.2 Example: Traveling Salesman Problem (TSP)

**Problem**: Find shortest tour visiting all cities exactly once

**Representation**:
- Chromosome: N random keys (one per city)
- Decoder: Sort keys to get visit order
- Fitness: Total tour distance

**Local Search**: 2-opt (reverse segments to reduce crossings)

### 9.3 Example: 0/1 Knapsack

**Problem**: Maximize value of items in knapsack without exceeding capacity

**Representation**:
- Chromosome: N random keys (one per item)
- Decoder: Select items where key > threshold (greedy by value/weight)
- Fitness: Total value (with capacity constraint)

**Local Search**: Swap (exchange selected/unselected items)

### 9.4 Example: Job Shop Scheduling

**Problem**: Minimize makespan for jobs on machines

**Representation**:
- Chromosome: Operations ordered by random keys
- Decoder: Schedule operations in key order respecting precedence
- Fitness: Maximum completion time

**Local Search**: Shift (move operations to earlier slots)

---

## 10. Implementation Details

### 10.1 Individual Representation

```cpp
template<typename T>
class Individual {
private:
    std::vector<std::vector<T>> components;  // Multi-component support

public:
    T fitness;
    std::vector<T> objectives;  // For multi-objective
    int rank;                   // Pareto rank
    T crowding_distance;
    bool evaluated;

    // Access methods
    std::vector<T>& get_component(int idx) { return components[idx]; }
    std::vector<T> get_chromosome() const;  // Flattened view
    std::vector<T> flatten() const;         // For GPU transfer
    void unflatten(const std::vector<T>& data);  // From GPU transfer
};
```

### 10.2 Population Management

```cpp
template<typename T>
class Population {
private:
    std::vector<Individual<T>> individuals;
    std::vector<std::vector<int>> fronts;  // Pareto fronts

public:
    // BRKGA operations
    void initialize();
    std::vector<Individual<T>> get_elite();
    std::vector<Individual<T>> get_non_elite();
    void next_generation_step();
    void finalize_generation();

    // NSGA-II operations
    void fast_non_dominated_sort();
    void calculate_crowding_distance();
    void select_next_generation_nsga2();
    void ensure_minimum_diversity();

    // Access
    const Individual<T>& get_best() const;
    std::vector<Individual<T>> get_pareto_front() const;
};
```

### 10.3 Configuration Base Class

```cpp
template<typename T>
class BRKGAConfig {
public:
    // Population parameters
    int population_size;
    int elite_size;
    int mutant_size;
    double elite_prob;
    int max_generations;

    // Problem structure
    std::vector<int> component_lengths;
    int num_components;
    int num_objectives;

    // Function pointers (set by derived class)
    std::function<T(const Individual<T>&)> fitness_function;
    std::function<std::vector<std::vector<T>>(const Individual<T>&)> decoder;
    std::function<bool(T, T)> comparator;
    std::vector<std::function<T(const Individual<T>&)>> objective_functions;

    // GPU configuration
    int threads_per_block;
    dim3 block_size;
    dim3 grid_size;

    // Local search
    LocalSearchManager<T>* local_search_manager;

    // Methods
    bool is_multi_objective() const { return num_objectives > 1; }
    int get_total_chromosome_length() const;
    int get_offspring_size() const;
    void update_cuda_grid_size();
};
```

### 10.4 Build System

The Makefile supports multiple build configurations:

```makefile
# Compiler settings
NVCC = nvcc
NVCC_FLAGS = -std=c++17 -O3 -arch=sm_75

# Targets
all: main

main: main.cu
    $(NVCC) $(NVCC_FLAGS) -Xcompiler -fopenmp $< -o $@ -lcurand -lcudart -lpthread

clean:
    rm -f main *.o
```

---

## 11. Experimental Evaluation

### 11.1 Benchmark Problems

The framework is tested on standard benchmark instances:

| Problem | Instance | Size | Optimal |
|---------|----------|------|---------|
| TSP | berlin52 | 52 cities | 7542 |
| TSP | kroA100 | 100 cities | 21282 |
| TSP | att532 | 532 cities | 27686 |
| Knapsack | kp_100 | 100 items | - |
| Job Shop | ft10 | 10×10 | 930 |

### 11.2 Performance Metrics

1. **Solution Quality**: Gap from optimal/best known solution
2. **Convergence Speed**: Generations to reach target quality
3. **Execution Time**: Wall-clock time per generation
4. **GPU Utilization**: Occupancy and memory bandwidth
5. **Scalability**: Performance vs. problem size and GPU count

### 11.3 Expected Results

**GPU vs CPU Speedup** (population=1000, chromosome=100):
- Single GPU: ~10-50x speedup
- Dual GPU: ~15-80x speedup
- Quad GPU: ~25-150x speedup

**Solution Quality** (TSP berlin52):
- BRKGA alone: Within 1-5% of optimal
- BRKGA + 2-opt: Within 0.1-1% of optimal

---

## 12. Conclusions and Future Work

### 12.1 Summary of Contributions

1. **Unified Framework**: Complete BRKGA implementation with GPU acceleration
2. **LLM Integration**: Natural language problem specification and automatic code generation
3. **Multi-GPU Support**: Island model with efficient migration
4. **Hybrid Approach**: Seamless local search integration
5. **Extensibility**: Easy to add new problem types and operators

### 12.2 Limitations

1. **LLM Dependency**: Requires API access and may incur costs
2. **CUDA Requirement**: GPU acceleration requires NVIDIA hardware
3. **Fitness Evaluation**: Complex fitness functions may still need manual optimization
4. **Problem Coverage**: Not all optimization problems fit BRKGA well

### 12.3 Future Work

1. **Device-Side Fitness Evaluation**: Move fitness computation to GPU kernels
2. **Dynamic Parameter Adaptation**: Auto-tune BRKGA parameters during evolution
3. **Distributed Computing**: Extend island model across machines
4. **Problem Library**: Pre-built configurations for common problems
5. **Visualization**: Real-time evolution visualization
6. **More LLM Models**: Support for additional code generation models

---

## References

1. Gonçalves, J.F., Resende, M.G.C. (2011). "Biased random-key genetic algorithms for combinatorial optimization." Journal of Heuristics, 17(5), 487-525.

2. Deb, K., Pratap, A., Agarwal, S., Meyarivan, T. (2002). "A fast and elitist multiobjective genetic algorithm: NSGA-II." IEEE Transactions on Evolutionary Computation, 6(2), 182-197.

3. NVIDIA CUDA Programming Guide. https://docs.nvidia.com/cuda/

4. Anthropic Claude API Documentation. https://docs.anthropic.com/

---

## Appendix A: API Reference

### A.1 LLMBRKGASolver

```python
class LLMBRKGASolver:
    def __init__(self,
                 context_package_path: str = "llm_solver/context",
                 framework_path: str = "brkga",
                 output_dir: str = "llm_solver",
                 api_key: Optional[str] = None)

    def solve(self,
              problem_description: str,
              clarifying_qa: Optional[Dict[str, str]] = None,
              data_files: Optional[Dict[str, str]] = None,
              hyperparameters: Optional[Dict[str, Any]] = None,
              quick_test_only: bool = False,
              max_iterations: int = 3,
              verbose: bool = True) -> SolverSession
```

### A.2 BRKGA Hyperparameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `population_size` | int | 100-500 | Number of individuals |
| `elite_percentage` | float | 0.15 | Fraction of elite (0-1) |
| `mutant_percentage` | float | 0.15 | Fraction of mutants (0-1) |
| `elite_prob` | float | 0.70 | Bias toward elite parent |
| `max_generations` | int | 500-1500 | Maximum generations |

---

## Appendix B: Example Usage

### B.1 Python API

```python
from llm_solver.core.llm_brkga_solver import LLMBRKGASolver

# Initialize solver
solver = LLMBRKGASolver()

# Solve TSP from natural language
session = solver.solve(
    problem_description="""
    I have a traveling salesman problem with 52 cities in Berlin.
    I want to find the shortest tour that visits all cities exactly once
    and returns to the starting city.
    """,
    data_files={"cities": "brkga/data/berlin52.tsp"},
    hyperparameters={
        "population_size": 200,
        "max_generations": 500
    }
)

# Access results
print(f"Best tour length: {session.best_fitness}")
```

### B.2 Command Line

```bash
# Run the solver
python -m llm_solver.core.llm_brkga_solver \
    "Solve the TSP for 52 cities in berlin52.tsp" \
    --data brkga/data/berlin52.tsp \
    --quick-test
```

---

*Document generated for thesis work on LLM-BRKGA: An LLM-Powered GPU-Accelerated Genetic Algorithm Framework*
