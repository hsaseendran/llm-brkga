#!/usr/bin/env python3
"""
Simple example: Using LLM BRKGA Solver with a data file

This is the simplest way to use the data file parsing feature.
"""

import os
from core.llm_brkga_solver import LLMBRKGASolver

# Initialize solver
solver = LLMBRKGASolver(
    context_package_path="./context",
    output_dir="./generated"
)

# Describe your problem
problem = """
Solve the Traveling Salesman Problem (TSP).
Find the shortest tour visiting all cities exactly once.
"""

# Path to your data file
data_file = "../brkga/data/berlin52.tsp"

# That's it! Just add data_file_path parameter
print("Solving TSP with data file...")
result = solver.solve(
    problem_description=problem,
    data_file_path=data_file,  # <-- NEW: Just add this line!
    quick_test_only=True,
    verbose=True
)

# Check results
if result.success:
    print(f"\n✅ Success! Best tour length: {result.best_fitness:.2f}")
else:
    print(f"\n❌ Failed: {result.errors}")
