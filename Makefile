# ============================================================================
# BRKGA Framework + LLM Solver - Unified Makefile
# ============================================================================

# Compiler settings
NVCC := nvcc
CXX := g++
PYTHON := python3
PIP := pip3

# Directories
BRKGA_DIR := brkga
CORE_DIR := $(BRKGA_DIR)/core
CONFIGS_DIR := $(BRKGA_DIR)/configs
BUILD_DIR := $(BRKGA_DIR)/build
EXAMPLES_DIR := $(BRKGA_DIR)/examples
DATA_DIR := $(BRKGA_DIR)/data
SOLUTIONS_DIR := $(BRKGA_DIR)/solutions

LLM_DIR := llm_solver
LLM_CORE_DIR := $(LLM_DIR)/core
LLM_GENERATED_DIR := $(LLM_DIR)/generated
LLM_RESULTS_DIR := $(LLM_DIR)/results
LLM_CONTEXT_DIR := $(LLM_DIR)/context

# Compilation flags
CUDA_ARCH := sm_75
CPP_STD := c++17
OPT_LEVEL := O3
NVCC_FLAGS := -std=$(CPP_STD) -arch=$(CUDA_ARCH) -$(OPT_LEVEL) \
              -Xcompiler -fopenmp -I$(CORE_DIR)
LIBS := -lcurand -lcudart -lpthread

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# ============================================================================
# DEFAULT TARGET
# ============================================================================

.PHONY: all
all: help

# ============================================================================
# HELP / DOCUMENTATION
# ============================================================================

.PHONY: help
help:
	@echo "$(BLUE)╔══════════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║                                                                  ║$(NC)"
	@echo "$(BLUE)║         BRKGA Framework + LLM Solver - Makefile Help            ║$(NC)"
	@echo "$(BLUE)║                                                                  ║$(NC)"
	@echo "$(BLUE)╚══════════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(YELLOW)C++ BRKGA Framework Targets:$(NC)"
	@echo "  $(GREEN)make build-core$(NC)          - Build core BRKGA library"
	@echo "  $(GREEN)make build-examples$(NC)      - Build example programs"
	@echo "  $(GREEN)make example-tsp$(NC)         - Build and run TSP example"
	@echo "  $(GREEN)make example-knapsack$(NC)    - Build and run knapsack example"
	@echo "  $(GREEN)make test-cpp$(NC)            - Run C++ tests"
	@echo "  $(GREEN)make clean-cpp$(NC)           - Clean C++ build artifacts"
	@echo ""
	@echo "$(YELLOW)LLM Solver Targets:$(NC)"
	@echo "  $(GREEN)make demo-llm$(NC)            - Run quick LLM solver demo"
	@echo "  $(GREEN)make interactive-llm$(NC)     - Interactive problem solving mode"
	@echo "  $(GREEN)make examples-llm$(NC)        - Run all LLM solver examples"
	@echo "  $(GREEN)make test-llm$(NC)            - Run LLM solver tests"
	@echo "  $(GREEN)make install-llm$(NC)         - Install Python dependencies"
	@echo "  $(GREEN)make clean-llm$(NC)           - Clean LLM generated files"
	@echo ""
	@echo "$(YELLOW)Specific LLM Examples:$(NC)"
	@echo "  $(GREEN)make llm-knapsack$(NC)        - Solve knapsack with LLM"
	@echo "  $(GREEN)make llm-tsp$(NC)             - Solve TSP with LLM"
	@echo "  $(GREEN)make llm-scheduling$(NC)      - Solve scheduling with LLM"
	@echo "  $(GREEN)make llm-multi$(NC)           - Solve multi-objective with LLM"
	@echo "  $(GREEN)make llm-binpack$(NC)         - Solve bin packing with LLM"
	@echo ""
	@echo "$(YELLOW)Utility Targets:$(NC)"
	@echo "  $(GREEN)make setup$(NC)               - Complete project setup"
	@echo "  $(GREEN)make check-env$(NC)           - Check environment setup"
	@echo "  $(GREEN)make clean-all$(NC)           - Clean everything"
	@echo "  $(GREEN)make status$(NC)              - Show project status"
	@echo "  $(GREEN)make help$(NC)                - Show this help message"
	@echo ""
	@echo "$(YELLOW)Environment Variables:$(NC)"
	@echo "  ANTHROPIC_API_KEY  - Required for LLM solver"
	@echo "  CUDA_VISIBLE_DEVICES - GPU selection"
	@echo ""

# ============================================================================
# C++ BRKGA FRAMEWORK TARGETS
# ============================================================================

.PHONY: build-core
build-core:
	@echo "$(BLUE)Building BRKGA core library...$(NC)"
	@mkdir -p $(BUILD_DIR)
	@$(NVCC) $(NVCC_FLAGS) -c $(CORE_DIR)/*.cu -o $(BUILD_DIR)/brkga_core.o
	@echo "$(GREEN)✓ Core library built$(NC)"

.PHONY: build-examples
build-examples: build-core
	@echo "$(BLUE)Building example programs...$(NC)"
	@for example in $(EXAMPLES_DIR)/*.cu; do \
		base=$$(basename $$example .cu); \
		echo "  Building $$base..."; \
		$(NVCC) $(NVCC_FLAGS) $$example -o $(BUILD_DIR)/$$base $(LIBS); \
	done
	@echo "$(GREEN)✓ Examples built$(NC)"

.PHONY: example-tsp
example-tsp:
	@echo "$(BLUE)Building and running TSP example...$(NC)"
	@$(NVCC) $(NVCC_FLAGS) -I$(CORE_DIR) $(CONFIGS_DIR)/tsp_config.hpp \
		$(CORE_DIR)/main.cu -o $(BUILD_DIR)/tsp_solver $(LIBS)
	@$(BUILD_DIR)/tsp_solver
	@echo "$(GREEN)✓ TSP example completed$(NC)"

.PHONY: example-knapsack
example-knapsack:
	@echo "$(BLUE)Building and running knapsack example...$(NC)"
	@$(NVCC) $(NVCC_FLAGS) -I$(CORE_DIR) $(CONFIGS_DIR)/knapsack_config.hpp \
		$(CORE_DIR)/main.cu -o $(BUILD_DIR)/knapsack_solver $(LIBS)
	@$(BUILD_DIR)/knapsack_solver
	@echo "$(GREEN)✓ Knapsack example completed$(NC)"

.PHONY: test-cpp
test-cpp: build-core
	@echo "$(BLUE)Running C++ tests...$(NC)"
	@if [ -d "$(BRKGA_DIR)/tests" ]; then \
		cd $(BRKGA_DIR)/tests && make run-tests; \
	else \
		echo "$(YELLOW)No C++ tests found$(NC)"; \
	fi

.PHONY: clean-cpp
clean-cpp:
	@echo "$(BLUE)Cleaning C++ build artifacts...$(NC)"
	@rm -rf $(BUILD_DIR)/*
	@rm -f $(BRKGA_DIR)/*.o $(BRKGA_DIR)/*.out
	@echo "$(GREEN)✓ C++ artifacts cleaned$(NC)"

# ============================================================================
# LLM SOLVER TARGETS
# ============================================================================

.PHONY: demo-llm
demo-llm: check-api-key
	@echo "$(BLUE)Running LLM solver demo...$(NC)"
	@cd $(LLM_DIR) && $(PYTHON) demo.py
	@echo "$(GREEN)✓ Demo completed$(NC)"

.PHONY: interactive-llm
interactive-llm: check-api-key
	@echo "$(BLUE)Starting interactive LLM solver...$(NC)"
	@cd $(LLM_DIR) && $(PYTHON) examples.py interactive

.PHONY: examples-llm
examples-llm: check-api-key
	@echo "$(BLUE)Running all LLM solver examples...$(NC)"
	@cd $(LLM_DIR) && $(PYTHON) examples.py all
	@echo "$(GREEN)✓ All examples completed$(NC)"

.PHONY: test-llm
test-llm: check-api-key
	@echo "$(BLUE)Running LLM solver tests...$(NC)"
	@cd $(LLM_DIR) && $(PYTHON) examples.py knapsack
	@echo "$(GREEN)✓ Tests completed$(NC)"

.PHONY: llm-knapsack
llm-knapsack: check-api-key
	@echo "$(BLUE)Solving knapsack with LLM...$(NC)"
	@cd $(LLM_DIR) && $(PYTHON) examples.py knapsack

.PHONY: llm-tsp
llm-tsp: check-api-key
	@echo "$(BLUE)Solving TSP with LLM...$(NC)"
	@cd $(LLM_DIR) && $(PYTHON) examples.py tsp

.PHONY: llm-scheduling
llm-scheduling: check-api-key
	@echo "$(BLUE)Solving scheduling with LLM...$(NC)"
	@cd $(LLM_DIR) && $(PYTHON) examples.py scheduling

.PHONY: llm-multi
llm-multi: check-api-key
	@echo "$(BLUE)Solving multi-objective with LLM...$(NC)"
	@cd $(LLM_DIR) && $(PYTHON) examples.py multi

.PHONY: llm-binpack
llm-binpack: check-api-key
	@echo "$(BLUE)Solving bin packing with LLM...$(NC)"
	@cd $(LLM_DIR) && $(PYTHON) examples.py binpack

.PHONY: install-llm
install-llm:
	@echo "$(BLUE)Installing LLM solver dependencies...$(NC)"
	@$(PIP) install -r $(LLM_DIR)/requirements.txt
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

.PHONY: clean-llm
clean-llm:
	@echo "$(BLUE)Cleaning LLM generated files...$(NC)"
	@rm -rf $(LLM_GENERATED_DIR)/*
	@rm -rf $(LLM_RESULTS_DIR)/*
	@find $(LLM_DIR) -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find $(LLM_DIR) -name "*.pyc" -delete
	@echo "$(GREEN)✓ LLM artifacts cleaned$(NC)"

# ============================================================================
# SETUP AND INSTALLATION
# ============================================================================

.PHONY: setup
setup: create-dirs install-llm
	@echo "$(BLUE)Setting up project...$(NC)"
	@echo "$(GREEN)✓ Project setup complete$(NC)"
	@echo ""
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  1. Set ANTHROPIC_API_KEY: export ANTHROPIC_API_KEY='your-key'"
	@echo "  2. Test C++ framework: make example-tsp"
	@echo "  3. Test LLM solver: make demo-llm"

.PHONY: create-dirs
create-dirs:
	@echo "$(BLUE)Creating directory structure...$(NC)"
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(DATA_DIR)
	@mkdir -p $(SOLUTIONS_DIR)
	@mkdir -p $(LLM_GENERATED_DIR)
	@mkdir -p $(LLM_RESULTS_DIR)
	@mkdir -p $(LLM_CONTEXT_DIR)
	@touch $(LLM_GENERATED_DIR)/.gitkeep
	@touch $(LLM_RESULTS_DIR)/.gitkeep
	@echo "$(GREEN)✓ Directories created$(NC)"

# ============================================================================
# ENVIRONMENT CHECKS
# ============================================================================

.PHONY: check-env
check-env:
	@echo "$(BLUE)Checking environment...$(NC)"
	@echo ""
	@echo "$(YELLOW)Python:$(NC)"
	@$(PYTHON) --version || echo "$(RED)✗ Python not found$(NC)"
	@echo ""
	@echo "$(YELLOW)NVCC:$(NC)"
	@$(NVCC) --version | head -n 1 || echo "$(RED)✗ NVCC not found$(NC)"
	@echo ""
	@echo "$(YELLOW)CUDA:$(NC)"
	@nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "$(RED)✗ CUDA not available$(NC)"
	@echo ""
	@echo "$(YELLOW)Anthropic API Key:$(NC)"
	@if [ -z "$$ANTHROPIC_API_KEY" ]; then \
		echo "$(RED)✗ ANTHROPIC_API_KEY not set$(NC)"; \
	else \
		echo "$(GREEN)✓ ANTHROPIC_API_KEY is set$(NC)"; \
	fi
	@echo ""
	@echo "$(YELLOW)Python Packages:$(NC)"
	@$(PIP) show anthropic > /dev/null 2>&1 && echo "$(GREEN)✓ anthropic package installed$(NC)" || echo "$(RED)✗ anthropic package not installed$(NC)"

.PHONY: check-api-key
check-api-key:
	@if [ -z "$$ANTHROPIC_API_KEY" ]; then \
		echo "$(RED)Error: ANTHROPIC_API_KEY not set$(NC)"; \
		echo "Set it with: export ANTHROPIC_API_KEY='your-key-here'"; \
		exit 1; \
	fi

# ============================================================================
# STATUS AND INFORMATION
# ============================================================================

.PHONY: status
status:
	@echo "$(BLUE)╔══════════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║                     Project Status                               ║$(NC)"
	@echo "$(BLUE)╚══════════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(YELLOW)C++ Build Artifacts:$(NC)"
	@if [ -d "$(BUILD_DIR)" ] && [ "$$(ls -A $(BUILD_DIR) 2>/dev/null)" ]; then \
		ls -lh $(BUILD_DIR) | tail -n +2; \
	else \
		echo "  No build artifacts"; \
	fi
	@echo ""
	@echo "$(YELLOW)LLM Generated Configs:$(NC)"
	@if [ -d "$(LLM_GENERATED_DIR)" ] && [ "$$(ls -A $(LLM_GENERATED_DIR) 2>/dev/null | grep -v .gitkeep)" ]; then \
		ls -lh $(LLM_GENERATED_DIR) | tail -n +2 | grep -v .gitkeep; \
	else \
		echo "  No generated configs"; \
	fi
	@echo ""
	@echo "$(YELLOW)LLM Results:$(NC)"
	@if [ -d "$(LLM_RESULTS_DIR)" ] && [ "$$(ls -A $(LLM_RESULTS_DIR) 2>/dev/null | grep -v .gitkeep)" ]; then \
		ls -lh $(LLM_RESULTS_DIR) | tail -n +2 | grep -v .gitkeep; \
	else \
		echo "  No results yet"; \
	fi
	@echo ""

# ============================================================================
# CLEANING
# ============================================================================

.PHONY: clean-all
clean-all: clean-cpp clean-llm
	@echo "$(GREEN)✓ All artifacts cleaned$(NC)"

.PHONY: clean-data
clean-data:
	@echo "$(BLUE)Cleaning data directories...$(NC)"
	@rm -rf $(DATA_DIR)/*
	@rm -rf $(SOLUTIONS_DIR)/*
	@echo "$(GREEN)✓ Data cleaned$(NC)"

.PHONY: pristine
pristine: clean-all clean-data
	@echo "$(BLUE)Restoring to pristine state...$(NC)"
	@rm -rf $(BUILD_DIR)
	@echo "$(GREEN)✓ Project restored to pristine state$(NC)"

# ============================================================================
# BENCHMARKING AND TESTING
# ============================================================================

.PHONY: benchmark
benchmark: check-api-key
	@echo "$(BLUE)Running benchmarks...$(NC)"
	@cd $(LLM_DIR) && $(PYTHON) -c "from core import LLMBRKGASolver; \
		problems = ['knapsack', 'tsp', 'scheduling']; \
		for p in problems: print(f'Testing {p}...'); \
		# Add benchmark code here"

.PHONY: profile-cpp
profile-cpp: build-core
	@echo "$(BLUE)Profiling C++ code...$(NC)"
	@nvprof $(BUILD_DIR)/tsp_solver

# ============================================================================
# DOCUMENTATION
# ============================================================================

.PHONY: docs
docs:
	@echo "$(BLUE)Building documentation...$(NC)"
	@echo "Available documentation:"
	@echo "  - README.md"
	@echo "  - $(LLM_DIR)/README.md"
	@echo "  - ARCHITECTURE.md"
	@echo "  - GETTING_STARTED.md"

# ============================================================================
# VARIABLES INFO
# ============================================================================

.PHONY: vars
vars:
	@echo "$(BLUE)Makefile Variables:$(NC)"
	@echo "NVCC: $(NVCC)"
	@echo "PYTHON: $(PYTHON)"
	@echo "CUDA_ARCH: $(CUDA_ARCH)"
	@echo "CPP_STD: $(CPP_STD)"
	@echo "BRKGA_DIR: $(BRKGA_DIR)"
	@echo "LLM_DIR: $(LLM_DIR)"
	@echo "NVCC_FLAGS: $(NVCC_FLAGS)"

# ============================================================================
# DEVELOPMENT HELPERS
# ============================================================================

.PHONY: watch-llm
watch-llm:
	@echo "$(BLUE)Watching LLM generated files...$(NC)"
	@watch -n 2 'ls -lht $(LLM_GENERATED_DIR) | head -n 10'

.PHONY: tail-results
tail-results:
	@echo "$(BLUE)Tailing latest result...$(NC)"
	@tail -f $$(ls -t $(LLM_RESULTS_DIR)/* | head -n 1)

# ============================================================================