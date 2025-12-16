# LLM-Guided BRKGA Framework

A GPU-accelerated Biased Random-Key Genetic Algorithm (BRKGA) framework with LLM-powered problem configuration.

## Overview

This project combines:
- **BRKGA CUDA Library**: High-performance GPU-accelerated genetic algorithm implementation
- **LLM Solver Interface**: Natural language interface for configuring optimization problems

## Project Structure

```
llm_brkga/
├── brkga/                  # Core BRKGA CUDA library
│   ├── core/               # Core components (solver, population, genetic operators)
│   ├── configs/            # Problem-specific configurations (TSP, Knapsack, etc.)
│   ├── examples/           # Example implementations
│   └── utils/              # Utility functions
├── llm_solver/             # LLM-powered solver interface
│   ├── core/               # LLM interaction and code generation
│   ├── context/            # Problem context management
│   └── generated/          # Auto-generated solver code
├── docs/                   # Technical documentation
└── data/                   # Sample data files
```

## Quick Start

### Prerequisites

- CUDA Toolkit 11.0+
- C++17 compatible compiler (GCC 9+)
- NVIDIA GPU with Compute Capability 7.5+
- Python 3.8+ (for LLM solver)

### Build BRKGA Library

```bash
cd brkga
make
```

### Run LLM Solver

```bash
export ANTHROPIC_API_KEY="your-api-key"
python llm_solver/interactive.py
```

## Components

### BRKGA Library (`brkga/`)

GPU-accelerated genetic algorithm framework supporting:
- TSP (Traveling Salesman Problem)
- TSPJ (TSP with Job constraints)
- Knapsack Problem
- Multi-objective optimization (NSGA-II)
- Custom problem configurations

See [brkga/README.md](brkga/README.md) for detailed documentation.

### LLM Solver (`llm_solver/`)

Natural language interface for optimization:
- Describe your problem in plain English
- LLM generates appropriate CUDA configuration
- Automatic compilation and execution

See [llm_solver/README.md](llm_solver/README.md) for usage guide.

## Documentation

- [Technical Architecture](docs/TECHNICAL_ARCHITECTURE.md)
- [Thesis Documentation](docs/THESIS_DOCUMENTATION.md)
