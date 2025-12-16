# LLM BRKGA Solver - Usage Guide

## Quick Start

### 1. Set Your API Key
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

### 2. Run a Pre-configured Problem

#### Knapsack Problem (Simple Demo)
```bash
cd /mnt/drive2/users/harishj/llm_brkga
python llm_solver/demo.py
```

#### Berlin52 TSP (Bigger Problem)
```bash
python llm_solver/demo_tsp.py
```

#### Interactive Mode (Any Problem)
```bash
python llm_solver/interactive.py
```

Or with a problem file:
```bash
python llm_solver/interactive.py llm_solver/problems/berlin52_tsp.txt
```

---

## How It Works

```
Your Natural Language Description
          ↓
    LLM Analyzes Problem
          ↓
  Generates C++ BRKGA Config
          ↓
      Compiles Solver
          ↓
   Runs BRKGA Optimization
          ↓
  Prints & Exports Solution
```

---

## Solving Berlin52 TSP

### Method 1: Using the Demo Script

```bash
python llm_solver/demo_tsp.py
```

This will:
1. Send the TSP problem description to Claude
2. Generate a C++ config file
3. Compile the solver
4. Run a quick test (100 generations)
5. Save the executable

Then run the full optimization:
```bash
./llm_solver/generated/berlin52_tsp_solver
```

### Method 2: Using Interactive Mode

```bash
python llm_solver/interactive.py llm_solver/problems/berlin52_tsp.txt
```

### Method 3: Directly Use Existing TSP Config

If you already have a working TSP config:
```bash
# The framework already has TSP configs
cd brkga
nvcc -std=c++17 -O3 -Xcompiler -fopenmp -I. \
     configs/tsp_config.hpp main.cu -o tsp_solver \
     -lcurand -lcudart -lpthread

./tsp_solver --data data/berlin52.tsp --config tsp_config
```

---

## Problem Description Format

### Structure

```
PROBLEM TITLE

Brief description of what you want to optimize.

PROBLEM:
- List constraints
- List requirements

DATA (if applicable):
- File location
- Format description

OBJECTIVE:
- What to minimize/maximize
- How solution is represented

SOLUTION FORMAT:
- What the output should look like
```

### Example: Knapsack

```
0/1 Knapsack Problem

Select items to maximize value without exceeding weight capacity.

PROBLEM:
- 20 items with weights and values
- Knapsack capacity: 100 units
- Each item can be selected at most once

OBJECTIVE:
- Maximize total value
- Don't exceed weight capacity

SOLUTION:
- Which items to select (indices)
- Total weight and value
```

### Example: TSP

```
Traveling Salesman Problem

Find shortest tour visiting all cities exactly once.

DATA:
- City coordinates in: brkga/data/berlin52.tsp

OBJECTIVE:
- Minimize total tour distance
- Visit each city exactly once
- Return to start

SOLUTION:
- Tour order [city0, city1, ...]
- Total distance
```

---

## Output Files

### Generated Config
- Location: `llm_solver/generated/PROBLEM_NAME_config.hpp`
- What: C++ BRKGA configuration class
- Contains: Fitness function, decoder, problem data

### Executable
- Location: `llm_solver/generated/PROBLEM_NAME_solver`
- What: Compiled BRKGA solver
- Run with: `./llm_solver/generated/PROBLEM_NAME_solver`

### Solution
- Location: `llm_solver/results/solution.txt`
- What: Final solution with:
  * Best fitness
  * Decoded solution (actual answer)
  * Raw chromosome (BRKGA internal)

### Console Output
The solver prints detailed solution breakdown:
```
=== Solution Details ===
Selected items: 0 1 2 3 6 8 9 ...
Total weight: 100/100
Total value: 500
...
```

☝️ **This is your actual solution!**

---

## Customization

### Adjust BRKGA Parameters

Edit the generated config file:
```cpp
// In the constructor
this->population_size = 200;     // Increase for better quality
this->max_generations = 2000;    // More generations
this->elite_size = 40;           // 20% of population
this->mutant_size = 20;          // 10% of population
this->elite_prob = 0.7;          // Elite inheritance probability
```

### Run with Different Data

Modify the `create_default()` method or add a `load_from_file()` method in the config.

### GPU Acceleration

The solver automatically uses GPU if available. Check output:
```
Selected execution mode: single_gpu
Active GPUs: 1
```

---

## Troubleshooting

### API Key Issues
```
Error: Could not resolve authentication method
```
**Fix:** Set `export ANTHROPIC_API_KEY="your-key"`

### Compilation Fails
**Fix:** Check the generated config at `llm_solver/generated/PROBLEM_config.hpp`

Common issues:
- Wrong include path (should be `../../brkga/core/config.hpp`)
- Wrong constructor parameters
- Missing template parameter `<T>`

### Solution is Just Chromosomes
**Fix:** Check the `print_solution()` output on console - that's the actual solution!
The chromosome in the file is the internal BRKGA representation.

---

## Advanced: Creating Custom Problems

1. Create a problem description file in `llm_solver/problems/`
2. Run: `python llm_solver/interactive.py llm_solver/problems/your_problem.txt`
3. The LLM will generate a custom BRKGA config
4. Compile and run!

### Supported Problem Types
- **Selection**: Knapsack, set cover, feature selection
- **Permutation**: TSP, VRP, scheduling
- **Assignment**: Task allocation, bin packing, graph partitioning
- **Sequencing**: Job shop, flow shop
- **Partitioning**: Clustering, graph coloring

---

## Performance Tips

### For Large Problems (like Berlin52):
- Increase population size: `population_size = 500`
- Increase generations: `max_generations = 5000`
- Use GPU mode (automatic if available)
- Enable local search (if supported in config)

### For Quick Testing:
- Reduce generations: `max_generations = 100`
- Smaller population: `population_size = 50`
- Check quick test output first

---

## Example Workflow

```bash
# 1. Set API key
export ANTHROPIC_API_KEY="sk-ant-..."

# 2. Create problem description
cat > my_problem.txt << EOF
My optimization problem description...
