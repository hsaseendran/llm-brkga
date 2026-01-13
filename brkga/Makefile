# Makefile for Unified Multi-GPU BRKGA Framework with NSGA-II

# Compiler settings
NVCC = nvcc
CXX = g++

# Directories
CORE_DIR = core
CONFIG_DIR = configs
UTILS_DIR = utils
EXAMPLES_DIR = examples
BUILD_DIR = build
DATA_DIR = data
SOLUTIONS_DIR = solutions

# CUDA architecture (adjust based on your GPU)
CUDA_ARCH = -arch=sm_75

# Compiler flags
NVCC_FLAGS = -std=c++17 $(CUDA_ARCH) -O3 -Xcompiler -fopenmp
CXX_FLAGS = -std=c++17 -O3 -fopenmp -Wall -Wextra

# Include directories
INCLUDES = -I. -I$(CORE_DIR) -I$(CONFIG_DIR) -I$(UTILS_DIR)

# CUDA libraries
CUDA_LIBS = -lcurand -lcudart -lpthread

# Target executables
TARGET = brkga_unified
ZDT1_TARGET = zdt1
MULTI_GPU_ZDT1 = multi_gpu_zdt1
MULTI_TSP_TARGET = multi_tsp
RCMADP_TARGET = rcmadp_solver

# Source files
MAIN_SRC = main.cu

# Header dependencies
CORE_HEADERS = $(wildcard $(CORE_DIR)/*.hpp) $(wildcard $(CORE_DIR)/*.cuh)
CONFIG_HEADERS = $(wildcard $(CONFIG_DIR)/*.hpp)
UTILS_HEADERS = $(wildcard $(UTILS_DIR)/*.hpp)
ALL_HEADERS = $(CORE_HEADERS) $(CONFIG_HEADERS) $(UTILS_HEADERS)

# Default target
all: directories $(TARGET)

# Create necessary directories
directories:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(DATA_DIR)
	@mkdir -p $(SOLUTIONS_DIR)

# Main target - unified solver
$(TARGET): $(MAIN_SRC) $(ALL_HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(MAIN_SRC) -o $(TARGET) $(CUDA_LIBS)

# Debug build
debug: NVCC_FLAGS += -g -G -DDEBUG
debug: directories $(TARGET)

# Release build (default)
release: NVCC_FLAGS += -DNDEBUG
release: directories $(TARGET)

# Performance optimized build
performance: NVCC_FLAGS += -DNDEBUG -use_fast_math --ptxas-options=-v
performance: directories $(TARGET)

# Clean build artifacts
clean:
	rm -f $(TARGET) $(ZDT1_TARGET) $(MULTI_GPU_ZDT1) $(MULTI_TSP_TARGET) $(RCMADP_TARGET) $(BASELINE_BENCHMARK) $(TEST_STREAMS) $(TEST_PHASE1_2) $(TEST_SEGMENTED_SORT) $(TEST_PHASE4_TSP) $(TEST_TRANSPOSE) $(TEST_INITIALIZE_LAYOUT) $(TEST_CROSSOVER_LAYOUT) $(TEST_MUTATION_LAYOUT) $(TEST_BRKGA_GEN_LAYOUT) $(TEST_REORDER_LAYOUT) $(TEST_GENE_MAJOR_BRKGA) $(BENCHMARK_ALL_PHASES)
	rm -rf $(BUILD_DIR)
	rm -f temp_main_*.cu
	rm -f *.o
	@echo "Cleaned build artifacts"

# Clean all generated files
clean-all: clean
	rm -rf $(SOLUTIONS_DIR)/*
	rm -f *_pareto_front.txt
	rm -f *.png
	@echo "Cleaned all generated files"

# ==========================================
# BrkgaCuda 2.0 Optimization Benchmarks
# ==========================================

BASELINE_BENCHMARK = benchmark_baseline
TEST_STREAMS = test_streams
TEST_PHASE1_2 = test_phase1_2
TEST_SEGMENTED_SORT = test_segmented_sort
TEST_PHASE4_TSP = test_phase4_tsp
TEST_TRANSPOSE = test_transpose
TEST_INITIALIZE_LAYOUT = test_initialize_layout
TEST_CROSSOVER_LAYOUT = test_crossover_layout
TEST_MUTATION_LAYOUT = test_mutation_layout
TEST_BRKGA_GEN_LAYOUT = test_brkga_generation_layout
TEST_REORDER_LAYOUT = test_reorder_layout
TEST_GENE_MAJOR_BRKGA = test_gene_major_brkga
BENCHMARK_ALL_PHASES = benchmark_all_phases

$(BASELINE_BENCHMARK): benchmark_baseline.cu $(ALL_HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) benchmark_baseline.cu -o $(BASELINE_BENCHMARK) $(CUDA_LIBS)

$(TEST_STREAMS): test_streams.cu core/cuda_streams.hpp
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) test_streams.cu -o $(TEST_STREAMS) $(CUDA_LIBS)

$(TEST_PHASE1_2): test_phase1_2.cu $(ALL_HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) test_phase1_2.cu -o $(TEST_PHASE1_2) $(CUDA_LIBS)

$(TEST_SEGMENTED_SORT): test_segmented_sort.cu utils/segmented_sort.hpp
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) test_segmented_sort.cu -o $(TEST_SEGMENTED_SORT) $(CUDA_LIBS)

$(TEST_PHASE4_TSP): test_phase4_tsp.cu $(ALL_HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) test_phase4_tsp.cu -o $(TEST_PHASE4_TSP) $(CUDA_LIBS)

$(TEST_TRANSPOSE): test_transpose.cu core/memory_layout.hpp
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) test_transpose.cu -o $(TEST_TRANSPOSE) $(CUDA_LIBS)

$(TEST_INITIALIZE_LAYOUT): test_initialize_layout.cu core/cuda_kernels.cuh core/memory_layout.hpp
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) test_initialize_layout.cu -o $(TEST_INITIALIZE_LAYOUT) $(CUDA_LIBS)

$(TEST_CROSSOVER_LAYOUT): test_crossover_layout.cu core/cuda_kernels.cuh core/memory_layout.hpp
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) test_crossover_layout.cu -o $(TEST_CROSSOVER_LAYOUT) $(CUDA_LIBS)

$(TEST_MUTATION_LAYOUT): test_mutation_layout.cu core/cuda_kernels.cuh core/memory_layout.hpp
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) test_mutation_layout.cu -o $(TEST_MUTATION_LAYOUT) $(CUDA_LIBS)

$(TEST_BRKGA_GEN_LAYOUT): test_brkga_generation_layout.cu core/cuda_kernels.cuh core/memory_layout.hpp
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) test_brkga_generation_layout.cu -o $(TEST_BRKGA_GEN_LAYOUT) $(CUDA_LIBS)

$(TEST_REORDER_LAYOUT): test_reorder_layout.cu core/cuda_kernels.cuh core/memory_layout.hpp
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) test_reorder_layout.cu -o $(TEST_REORDER_LAYOUT) $(CUDA_LIBS)

$(TEST_GENE_MAJOR_BRKGA): test_gene_major_brkga.cu core/gene_major_brkga.hpp core/cuda_kernels.cuh core/memory_layout.hpp
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) test_gene_major_brkga.cu -o $(TEST_GENE_MAJOR_BRKGA) $(CUDA_LIBS)

$(BENCHMARK_ALL_PHASES): benchmark_all_phases.cu core/gene_major_brkga.hpp core/cuda_kernels.cuh core/memory_layout.hpp
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) benchmark_all_phases.cu -o $(BENCHMARK_ALL_PHASES) $(CUDA_LIBS)

# Test stream infrastructure
test-streams: $(TEST_STREAMS)
	@echo "=== Testing CUDA Streams Infrastructure ==="
	./$(TEST_STREAMS)

# Test Phases 1 & 2 integration
test-phase1-2: $(TEST_PHASE1_2)
	@echo "=== Testing Phases 1 & 2 Stream-Based Solver ==="
	./$(TEST_PHASE1_2)

# Test Phase 4 segmented sorting
test-segmented-sort: $(TEST_SEGMENTED_SORT)
	@echo "=== Testing Phase 4 Segmented Sort ==="
	./$(TEST_SEGMENTED_SORT)

# Test Phase 4 with full TSP integration
test-phase4-tsp: $(TEST_PHASE4_TSP)
	@echo "=== Testing Phase 4 TSP Integration ==="
	./$(TEST_PHASE4_TSP)

# Test Phase 5 memory layout transpose
test-transpose: $(TEST_TRANSPOSE)
	@echo "=== Testing Phase 5 Memory Layout Transpose ==="
	./$(TEST_TRANSPOSE)

# Test Phase 5 Day 1: Initialize kernel
test-initialize-layout: $(TEST_INITIALIZE_LAYOUT)
	@echo "=== Testing Phase 5 Day 1: Initialize Kernel ==="
	./$(TEST_INITIALIZE_LAYOUT)

# Test Phase 5 Day 2: Crossover kernel
test-crossover-layout: $(TEST_CROSSOVER_LAYOUT)
	@echo "=== Testing Phase 5 Day 2: Crossover Kernel ==="
	./$(TEST_CROSSOVER_LAYOUT)

# Test Phase 5 Day 3: Mutation kernel
test-mutation-layout: $(TEST_MUTATION_LAYOUT)
	@echo "=== Testing Phase 5 Day 3: Mutation Kernel ==="
	./$(TEST_MUTATION_LAYOUT)

# Test Phase 5 Day 4: BRKGA generation kernel
test-brkga-gen-layout: $(TEST_BRKGA_GEN_LAYOUT)
	@echo "=== Testing Phase 5 Day 4: BRKGA Generation Kernel ==="
	./$(TEST_BRKGA_GEN_LAYOUT)

# Test Phase 5 Day 5: Reorder kernel
test-reorder-layout: $(TEST_REORDER_LAYOUT)
	@echo "=== Testing Phase 5 Day 5: Reorder Kernel ==="
	./$(TEST_REORDER_LAYOUT)

# Test Phase 5 Day 6: GeneLayoutBRKGA integration
test-gene-major-brkga: $(TEST_GENE_MAJOR_BRKGA)
	@echo "=== Testing Phase 5 Day 6: GeneLayoutBRKGA Integration ==="
	./$(TEST_GENE_MAJOR_BRKGA)

# Comprehensive benchmark: Baseline vs Gene-Major (Phases 1-5)
benchmark-all-phases: $(BENCHMARK_ALL_PHASES)
	@echo "=== BrkgaCuda 2.0 Comprehensive Benchmark ==="
	./$(BENCHMARK_ALL_PHASES)

# Run baseline benchmark (before optimizations)
benchmark-baseline: $(BASELINE_BENCHMARK)
	@echo "=== Running Baseline Performance Benchmark ==="
	@echo "This will record current performance before optimizations"
	./$(BASELINE_BENCHMARK) --pop 8000 --gens 500 --gpus 1 --mode baseline --knapsack-only

# Run baseline with TSP (requires TSP file)
benchmark-baseline-tsp: $(BASELINE_BENCHMARK)
	@if [ -f pla85900.tsp ]; then \
		./$(BASELINE_BENCHMARK) --tsp pla85900.tsp --pop 8000 --gens 500 --gpus 1 --mode baseline --tsp-only; \
	else \
		echo "Error: pla85900.tsp not found in current directory"; \
		echo "Please provide a TSP file or run benchmark-baseline for synthetic problems"; \
	fi

# Compare optimizations (run after implementing each optimization)
benchmark-streams: $(BASELINE_BENCHMARK)
	@echo "=== Benchmarking with CUDA Streams ==="
	./$(BASELINE_BENCHMARK) --pop 8000 --gens 500 --gpus 1 --mode streams --knapsack-only

benchmark-coalesced: $(BASELINE_BENCHMARK)
	@echo "=== Benchmarking with Coalesced Memory Access ==="
	./$(BASELINE_BENCHMARK) --pop 8000 --gens 500 --gpus 1 --mode coalesced --knapsack-only

benchmark-bbsegsort: $(BASELINE_BENCHMARK)
	@if [ -f pla85900.tsp ]; then \
		echo "=== Benchmarking with bb-segsort ==="; \
		./$(BASELINE_BENCHMARK) --tsp pla85900.tsp --pop 8000 --gens 500 --gpus 1 --mode bbsegsort --tsp-only; \
	else \
		echo "bb-segsort benchmark requires TSP file"; \
	fi

benchmark-full: $(BASELINE_BENCHMARK)
	@echo "=== Benchmarking with ALL Optimizations ==="
	./$(BASELINE_BENCHMARK) --pop 8000 --gens 500 --gpus 1 --mode full --knapsack-only

# ==========================================
# Multi-Objective / NSGA-II Targets
# ==========================================

# Single-GPU ZDT1 example
$(ZDT1_TARGET): $(EXAMPLES_DIR)/zdt1.cu $(CORE_HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(EXAMPLES_DIR)/zdt1.cu -o $(ZDT1_TARGET) $(CUDA_LIBS)

run-zdt1: $(ZDT1_TARGET)
	./$(ZDT1_TARGET)

# Multi-GPU ZDT1 example
$(MULTI_GPU_ZDT1): $(EXAMPLES_DIR)/multi_gpu_zdt1.cu $(CORE_HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(EXAMPLES_DIR)/multi_gpu_zdt1.cu -o $(MULTI_GPU_ZDT1) $(CUDA_LIBS)

run-multi-gpu-zdt1: $(MULTI_GPU_ZDT1)
	@echo "=== Running Multi-GPU NSGA-II on ZDT1 ==="
	./$(MULTI_GPU_ZDT1)

# Multi-GPU ZDT1 with custom parameters
run-multi-gpu-zdt1-large: $(MULTI_GPU_ZDT1)
	@echo "=== Running Multi-GPU NSGA-II with Large Population ==="
	./$(MULTI_GPU_ZDT1) --pop 2000 --gen 1000

# Benchmark multi-GPU vs single-GPU
benchmark-multi-gpu: $(MULTI_GPU_ZDT1)
	@echo "=== Benchmarking Multi-GPU NSGA-II ==="
	./$(MULTI_GPU_ZDT1) --benchmark --pop 1000 --gen 100

# Multi-objective TSP
$(MULTI_TSP_TARGET): $(EXAMPLES_DIR)/multi_tsp.cu $(CONFIG_DIR)/multi_tsp_config.hpp
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(EXAMPLES_DIR)/multi_tsp.cu -o $(MULTI_TSP_TARGET) $(CUDA_LIBS)

run-multi-tsp: $(MULTI_TSP_TARGET)
	@if [ -f data/berlin52.tsp ]; then \
		./$(MULTI_TSP_TARGET) --data data/berlin52.tsp; \
	else \
		echo "Error: data/berlin52.tsp not found"; \
		echo "Please download a TSPLIB instance or create sample data"; \
	fi

# ==========================================
# RCMADP (Resource-Constrained Multi-Agent Drop-and-Pick)
# ==========================================

$(RCMADP_TARGET): $(EXAMPLES_DIR)/rcmadp_example.cu $(CONFIG_DIR)/rcmadp_config.hpp $(CORE_HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(EXAMPLES_DIR)/rcmadp_example.cu -o $(RCMADP_TARGET) $(CUDA_LIBS)

run-rcmadp: $(RCMADP_TARGET)
	@if [ -f data/rcmadp_example/small_instance.tsp ]; then \
		./$(RCMADP_TARGET) data/rcmadp_example/small_instance.tsp data/rcmadp_example/small_processing_times.txt 3 2; \
	else \
		echo "Error: RCMADP example data not found"; \
		echo "Expected: data/rcmadp_example/small_instance.tsp and small_processing_times.txt"; \
	fi

run-rcmadp-custom: $(RCMADP_TARGET)
	@echo "Usage: ./$(RCMADP_TARGET) <tsp_file> <proc_times_file> [agents] [resources_per_agent]"
	@echo "   or: ./$(RCMADP_TARGET) --tspj <_TT.csv file> [agents] [resources_per_agent]"
	@echo ""
	@echo "Examples:"
	@echo "  ./$(RCMADP_TARGET) berlin52.tsp processing_times.txt 4 3"
	@echo "  ./$(RCMADP_TARGET) --tspj ../uploads/berlin52_TSPJ_TT.csv 4 3"

# Run with berlin52 TSPJ data (4 agents, 3 resources each)
run-rcmadp-berlin52: $(RCMADP_TARGET)
	@if [ -f data/berlin52_TSPJ_TT.csv ]; then \
		./$(RCMADP_TARGET) --tspj data/berlin52_TSPJ_TT.csv 4 3; \
	else \
		echo "Error: berlin52 TSPJ data not found"; \
		echo "Expected: data/berlin52_TSPJ_TT.csv and data/berlin52_TSPJ_JT.csv"; \
	fi

# ==========================================
# Testing Targets
# ==========================================

# Test single-objective BRKGA
test-brkga: $(TARGET)
	./$(TARGET) --create-samples
	@echo "\n=== Testing BRKGA (Single-Objective) ==="
	./$(TARGET) --data data/medium_tsp.tsp --config tsp_config

# Test multi-objective NSGA-II
test-nsga2: $(ZDT1_TARGET)
	@echo "=== Testing NSGA-II (Multi-Objective) ==="
	./$(ZDT1_TARGET)

# Test multi-GPU NSGA-II
test-multi-gpu-nsga2: $(MULTI_GPU_ZDT1)
	@echo "=== Testing Multi-GPU NSGA-II ==="
	./$(MULTI_GPU_ZDT1) --pop 500 --gen 250

# Comprehensive test suite
test-all-modes: $(TARGET) $(ZDT1_TARGET) $(MULTI_GPU_ZDT1)
	@echo "========================================="
	@echo "  Comprehensive Testing Suite"
	@echo "========================================="
	@echo "\n1. Single-objective BRKGA:"
	$(MAKE) test-brkga
	@echo "\n2. Single-GPU NSGA-II:"
	$(MAKE) test-nsga2
	@echo "\n3. Multi-GPU NSGA-II:"
	$(MAKE) test-multi-gpu-nsga2
	@echo "\n========================================="
	@echo "  All tests completed!"
	@echo "========================================="

# ==========================================
# Benchmarking Targets
# ==========================================

# Benchmark NSGA-II scaling
benchmark-nsga2-scaling: $(MULTI_GPU_ZDT1)
	@echo "=== NSGA-II Population Scaling Benchmark ==="
	@for pop in 100 500 1000 2000 5000; do \
		echo "\nPopulation size: $$pop"; \
		./$(MULTI_GPU_ZDT1) --pop $$pop --gen 100; \
	done

# Compare single vs multi-GPU performance
benchmark-gpu-comparison: $(MULTI_GPU_ZDT1)
	@echo "=== Single-GPU vs Multi-GPU Comparison ==="
	./$(MULTI_GPU_ZDT1) --benchmark --pop 2000 --gen 200

# Full benchmark suite
benchmark-comprehensive: $(TARGET) $(MULTI_GPU_ZDT1)
	@echo "========================================="
	@echo "  Comprehensive Benchmark Suite"
	@echo "========================================="
	@echo "\n1. BRKGA Single-Objective:"
	$(MAKE) benchmark
	@echo "\n2. NSGA-II Multi-Objective Scaling:"
	$(MAKE) benchmark-nsga2-scaling
	@echo "\n3. Multi-GPU Performance:"
	$(MAKE) benchmark-gpu-comparison
	@echo "\n========================================="
	@echo "  Benchmarks completed!"
	@echo "========================================="

# ==========================================
# Analysis and Visualization
# ==========================================

# Analyze Pareto front quality
analyze-pareto:
	@if [ -f zdt1_multi_gpu_pareto.txt ]; then \
		echo "=== Pareto Front Analysis ==="; \
		python3 -c "import numpy as np; \
		data = np.loadtxt('zdt1_multi_gpu_pareto.txt'); \
		print(f'Solutions: {len(data)}'); \
		print(f'f1 range: [{data[:,0].min():.4f}, {data[:,0].max():.4f}]'); \
		print(f'f2 range: [{data[:,1].min():.4f}, {data[:,1].max():.4f}]'); \
		true_f2 = 1 - np.sqrt(data[:,0]); \
		error = np.abs(data[:,1] - true_f2).mean(); \
		print(f'Average error: {error:.6f}')"; \
	else \
		echo "No Pareto front file found. Run a multi-objective optimization first."; \
	fi

# Visualize results
visualize-pareto:
	@if [ -f zdt1_multi_gpu_pareto.txt ]; then \
		python3 -c "import matplotlib.pyplot as plt; import numpy as np; \
		data=np.loadtxt('zdt1_multi_gpu_pareto.txt'); \
		plt.figure(figsize=(10,6)); \
		plt.scatter(data[:,0], data[:,1], alpha=0.6, s=30, label='Found solutions'); \
		x_true = np.linspace(0, 1, 200); \
		y_true = 1 - np.sqrt(x_true); \
		plt.plot(x_true, y_true, 'r--', linewidth=2, label='True Pareto front'); \
		plt.xlabel('Objective 1 (f1)', fontsize=12); \
		plt.ylabel('Objective 2 (f2)', fontsize=12); \
		plt.title('Multi-GPU NSGA-II: ZDT1 Benchmark Results', fontsize=14); \
		plt.legend(fontsize=10); \
		plt.grid(True, alpha=0.3); \
		plt.tight_layout(); \
		plt.savefig('pareto_visualization.png', dpi=300); \
		print('✓ Visualization saved to: pareto_visualization.png')"; \
	else \
		echo "No Pareto front file found. Run a multi-objective optimization first."; \
	fi

# ==========================================
# System Information
# ==========================================

# Show GPU information
gpu-info:
	@echo "=== GPU Information ==="
	@nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader
	@echo ""
	@echo "=== CUDA Version ==="
	@nvcc --version | grep "release"

# Check multi-GPU setup
check-multi-gpu:
	@echo "=== Multi-GPU System Check ==="
	@gpu_count=$$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l); \
	echo "GPUs detected: $$gpu_count"; \
	if [ $$gpu_count -gt 1 ]; then \
		echo "✓ Multi-GPU capable"; \
		echo ""; \
		echo "GPU Details:"; \
		nvidia-smi --query-gpu=index,name,memory.free --format=csv; \
	else \
		echo "✗ Only single GPU detected"; \
		echo "Multi-GPU features will not be available"; \
	fi

# System requirements check
check-requirements:
	@echo "=== System Requirements Check ==="
	@echo -n "CUDA Toolkit: "
	@if command -v nvcc >/dev/null 2>&1; then echo "✓ Installed"; else echo "✗ Not found"; fi
	@echo -n "C++17 Compiler: "
	@if $(CXX) -std=c++17 -x c++ -E /dev/null >/dev/null 2>&1; then echo "✓ Available"; else echo "✗ Not available"; fi
	@echo -n "Python 3: "
	@if command -v python3 >/dev/null 2>&1; then echo "✓ Installed"; else echo "✗ Not found"; fi
	@echo -n "Matplotlib: "
	@if python3 -c "import matplotlib" >/dev/null 2>&1; then echo "✓ Installed"; else echo "✗ Not found"; fi
	@echo -n "NumPy: "
	@if python3 -c "import numpy" >/dev/null 2>&1; then echo "✓ Installed"; else echo "✗ Not found"; fi
	@echo ""
	$(MAKE) check-multi-gpu

# ==========================================
# Documentation
# ==========================================

# Generate documentation
docs:
	@echo "=== Generating Documentation ==="
	@if command -v doxygen >/dev/null 2>&1; then \
		doxygen Doxyfile; \
		echo "✓ Documentation generated in docs/"; \
	else \
		echo "✗ Doxygen not found"; \
	fi

# Show usage examples
examples:
	@echo "========================================="
	@echo "  NSGA-II Multi-GPU Usage Examples"
	@echo "========================================="
	@echo ""
	@echo "1. Basic multi-objective optimization:"
	@echo "   make run-zdt1"
	@echo ""
	@echo "2. Multi-GPU optimization:"
	@echo "   make run-multi-gpu-zdt1"
	@echo ""
	@echo "3. Large population run:"
	@echo "   make run-multi-gpu-zdt1-large"
	@echo ""
	@echo "4. Performance benchmark:"
	@echo "   make benchmark-multi-gpu"
	@echo ""
	@echo "5. Visualize results:"
	@echo "   make visualize-pareto"
	@echo ""
	@echo "6. Custom parameters:"
	@echo "   ./multi_gpu_zdt1 --pop 3000 --gen 1000 --benchmark"
	@echo ""
	@echo "========================================="

# Help message
help:
	@echo "Unified Multi-GPU BRKGA Framework with NSGA-II"
	@echo "==============================================="
	@echo ""
	@echo "Build Targets:"
	@echo "  all              - Build main solver"
	@echo "  clean            - Clean build artifacts"
	@echo "  clean-all        - Clean everything"
	@echo ""
	@echo "Multi-Objective Targets:"
	@echo "  run-zdt1                  - Run single-GPU NSGA-II on ZDT1"
	@echo "  run-multi-gpu-zdt1        - Run multi-GPU NSGA-II on ZDT1"
	@echo "  run-multi-gpu-zdt1-large  - Large population multi-GPU run"
	@echo "  benchmark-multi-gpu       - Benchmark multi-GPU performance"
	@echo ""
	@echo "RCMADP (Resource-Constrained Multi-Agent Drop-and-Pick):"
	@echo "  run-rcmadp                - Run RCMADP with sample instance"
	@echo "  run-rcmadp-berlin52       - Run RCMADP with berlin52 TSPJ data"
	@echo "  run-rcmadp-custom         - Show usage for custom instances"
	@echo ""
	@echo "Testing:"
	@echo "  test-brkga               - Test single-objective BRKGA"
	@echo "  test-nsga2               - Test multi-objective NSGA-II"
	@echo "  test-multi-gpu-nsga2     - Test multi-GPU NSGA-II"
	@echo "  test-all-modes           - Run all tests"
	@echo ""
	@echo "Benchmarking:"
	@echo "  benchmark-nsga2-scaling  - Test population scaling"
	@echo "  benchmark-gpu-comparison - Compare single vs multi-GPU"
	@echo "  benchmark-comprehensive  - Full benchmark suite"
	@echo ""
	@echo "Analysis:"
	@echo "  analyze-pareto     - Analyze Pareto front quality"
	@echo "  visualize-pareto   - Create visualization"
	@echo ""
	@echo "System:"
	@echo "  gpu-info           - Show GPU information"
	@echo "  check-multi-gpu    - Check multi-GPU setup"
	@echo "  check-requirements - Check all requirements"
	@echo ""
	@echo "Documentation:"
	@echo "  docs               - Generate documentation"
	@echo "  examples           - Show usage examples"
	@echo "  help               - Show this help"
	@echo ""
	@echo "For more information, see README.md"

# Declare phony targets
.PHONY: all directories clean clean-all debug release performance \
        test-streams test-phase1-2 test-segmented-sort test-phase4-tsp test-transpose test-initialize-layout test-crossover-layout test-mutation-layout test-brkga-gen-layout test-reorder-layout test-gene-major-brkga benchmark-all-phases benchmark-baseline benchmark-baseline-tsp benchmark-streams benchmark-coalesced \
        benchmark-bbsegsort benchmark-full \
        run-zdt1 run-multi-gpu-zdt1 run-multi-gpu-zdt1-large \
        run-multi-tsp run-rcmadp run-rcmadp-berlin52 run-rcmadp-custom benchmark-multi-gpu \
        test-brkga test-nsga2 test-multi-gpu-nsga2 test-all-modes \
        benchmark-nsga2-scaling benchmark-gpu-comparison benchmark-comprehensive \
        analyze-pareto visualize-pareto \
        gpu-info check-multi-gpu check-requirements \
        docs examples help

# Default goal
.DEFAULT_GOAL := all