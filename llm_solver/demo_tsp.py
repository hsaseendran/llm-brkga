#!/usr/bin/env python3
"""
Demo script for solving TSP problems with LLM BRKGA Solver
"""

from core.llm_brkga_solver import LLMBRKGASolver
import os

def main():
    print("=" * 70)
    print("LLM BRKGA Solver - TSP Demo")
    print("=" * 70)
    print()
    
    # Problem description for Berlin52 TSP
    tsp_problem = """
    Traveling Salesman Problem (TSP)
    
    I have a salesman who needs to visit 52 cities in Berlin and return to 
    the starting city. Each city has x,y coordinates. The goal is to find 
    the shortest tour that visits each city exactly once and returns to 
    the start.
    
    The distance between cities is calculated using Euclidean distance:
    distance = sqrt((x2-x1)^2 + (y2-y1)^2)
    
    I want to:
    - Minimize the total tour distance
    - Visit each city exactly once
    - Return to the starting city
    
    The city coordinates will be loaded from a file: brkga/data/berlin52.tsp
    
    The file format is:
    - Line 1-6: Header information
    - Line 7+: city_id x_coordinate y_coordinate
    
    Example:
    1 565.0 575.0
    2 25.0 185.0
    3 345.0 750.0
    ...
    """
    
    print("Problem Description:")
    print("-" * 70)
    print(tsp_problem)
    print("-" * 70)
    print()
    
    # Initialize solver
    solver = LLMBRKGASolver(
        context_path="llm_solver/context",
        framework_path="brkga",
        output_dir="llm_solver"
    )
    
    print("Starting solver...")
    print()
    
    # Solve the problem
    result = solver.solve(
        problem_description=tsp_problem,
        max_iterations=3,
        output_name="berlin52_tsp"
    )
    
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    if result.success:
        print(f"✅ Success!")
        print(f"   Config: {result.config_path}")
        print(f"   Executable: {result.executable_path}")
        print(f"   Best fitness: {result.best_fitness}")
        print()
        print("The solver has been generated and tested.")
        print("You can now run the full optimization:")
        print(f"   ./{result.executable_path}")
        print()
        print("Results will be saved to:")
        print("   llm_solver/results/solution.txt")
    else:
        print(f"❌ Failed after {result.iterations} iterations")
        print(f"   Last config: {result.config_path}")
        print()
        print("Try:")
        print("  - Check API key is set: export ANTHROPIC_API_KEY=...")
        print("  - Review the problem description for clarity")
        print("  - Check the generated config for errors")
    
    print("=" * 70)

if __name__ == "__main__":
    main()
