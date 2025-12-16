#!/usr/bin/env python3
"""
Generate a custom TSP solver for Berlin52 using LLM
"""

from core.llm_brkga_solver import LLMBRKGASolver

def main():
    print("=" * 70)
    print("LLM BRKGA Solver - Berlin52 TSP")
    print("=" * 70)
    print()
    
    # Problem description that tells the LLM to load from file
    problem_description = """
    Traveling Salesman Problem (TSP)
    
    I need to solve the TSP for cities with given coordinates.
    
    PROBLEM DESCRIPTION:
    - Find the shortest tour that visits each city exactly once and returns to the start
    - Cities have (x, y) coordinates
    - Distance between cities is Euclidean: sqrt((x2-x1)^2 + (y2-y1)^2)
    
    DATA FILE:
    - Path: brkga/data/berlin52.tsp
    - Format: TSPLIB format
      * Header lines (NAME, COMMENT, TYPE, DIMENSION, EDGE_WEIGHT_TYPE)
      * Line with "NODE_COORD_SECTION"
      * Then: city_id x_coordinate y_coordinate (one per line)
      * Example:
        NAME : berlin52
        COMMENT : 52 locations in Berlin
        TYPE : TSP
        DIMENSION : 52
        EDGE_WEIGHT_TYPE : EUC_2D
        NODE_COORD_SECTION
        1 565.0 575.0
        2 25.0 185.0
        3 345.0 750.0
        ...
    
    REQUIREMENTS:
    1. Create a static method load_from_file(filename) that:
       - Reads the TSPLIB format file
       - Parses city coordinates
       - Calculates distance matrix
       - Returns unique_ptr to config
    
    2. Decoder should:
       - Convert random keys to tour permutation
       - Use random-key based permutation decoder
       - Sort cities by their chromosome values to get tour order
    
    3. Fitness function should:
       - Calculate total tour distance
       - Sum distances between consecutive cities in tour
       - Include distance from last city back to first
    
    4. Solution output should show:
       - Tour order (city indices)
       - Total tour distance
       - Validate that each city visited exactly once
    
    OBJECTIVE:
    - Minimize total tour distance
    - Each city must be visited exactly once
    """
    
    print("Problem: TSP for Berlin52 (52 cities)")
    print("Data file: brkga/data/berlin52.tsp")
    print()
    print("Generating solver with LLM...")
    print()
    
    # Initialize solver
    solver = LLMBRKGASolver(
        context_path="llm_solver/context",
        framework_path="brkga",
        output_dir="llm_solver"
    )
    
    # Generate and test the solver
    result = solver.solve(
        problem_description=problem_description,
        max_iterations=3,
        output_name="berlin52_tsp"
    )
    
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    if result.success:
        print("✅ SUCCESS! TSP Solver generated and tested")
        print()
        print("📁 Generated files:")
        print(f"   Config: {result.config_path}")
        print(f"   Solver: {result.executable_path}")
        print()
        print("🧪 Quick test result:")
        print(f"   Best tour distance: {result.best_fitness}")
        print()
        print("🚀 To run full optimization on Berlin52:")
        print(f"   ./{result.executable_path}")
        print()
        print("   This will:")
        print("   - Load city coordinates from brkga/data/berlin52.tsp")
        print("   - Run BRKGA for 1000 generations")
        print("   - Print the best tour found")
        print("   - Save solution to llm_solver/results/solution.txt")
        print()
        print("📊 Expected results:")
        print("   - Optimal tour distance: 7542")
        print("   - BRKGA typically finds: 7542-7600")
        print()
    else:
        print("❌ Failed to generate solver")
        print(f"   Iterations attempted: {result.iterations}")
        print(f"   Last config: {result.config_path}")
        print()
        print("💡 Next steps:")
        print("   1. Check API key: export ANTHROPIC_API_KEY=...")
        print("   2. Review generated config for errors")
        print(f"   3. Check: cat {result.config_path}")
    
    print("=" * 70)

if __name__ == "__main__":
    main()
