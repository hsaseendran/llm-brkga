#!/usr/bin/env python3
"""
Generate a custom TSP solver for Berlin52 using LLM
WITH CUSTOMIZABLE HYPERPARAMETERS
"""

from core.llm_brkga_solver import LLMBRKGASolver
import sys

def main():
    print("=" * 70)
    print("LLM BRKGA Solver - Berlin52 TSP (with Custom Hyperparameters)")
    print("=" * 70)
    print()

    # Problem description
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

    REQUIREMENTS:
    1. Create a static method load_from_file(filename) or create_from_file(filename)
    2. Decoder: Convert random keys to tour permutation (sorted_index strategy)
    3. Fitness: Calculate total tour distance (minimize)
    4. Solution output: Tour order and total distance

    OBJECTIVE:
    - Minimize total tour distance
    - Each city visited exactly once
    """

    print("Problem: TSP for Berlin52 (52 cities)")
    print("Data file: brkga/data/berlin52.tsp")
    print()

    # Ask user about hyperparameters
    print("Would you like to customize BRKGA hyperparameters? (y/n, default=n): ", end="")
    customize = input().strip().lower()

    hyperparameters = None

    if customize in ['y', 'yes']:
        print()
        print("Customize hyperparameters (press Enter to use defaults):")
        print()

        hyperparameters = {}

        # Population size
        pop_input = input("  Population size (default: auto, recommended: 200-500 for TSP): ").strip()
        if pop_input:
            try:
                hyperparameters['population_size'] = int(pop_input)
            except ValueError:
                print("    Invalid, using default")

        # Elite percentage
        elite_input = input("  Elite percentage 0-1 (default: 0.15, recommended: 0.15-0.20): ").strip()
        if elite_input:
            try:
                hyperparameters['elite_percentage'] = float(elite_input)
            except ValueError:
                print("    Invalid, using default")

        # Mutant percentage
        mutant_input = input("  Mutant percentage 0-1 (default: 0.15, recommended: 0.10-0.15): ").strip()
        if mutant_input:
            try:
                hyperparameters['mutant_percentage'] = float(mutant_input)
            except ValueError:
                print("    Invalid, using default")

        # Elite bias
        bias_input = input("  Elite bias 0-1 (default: 0.7, recommended: 0.7-0.75): ").strip()
        if bias_input:
            try:
                hyperparameters['elite_prob'] = float(bias_input)
            except ValueError:
                print("    Invalid, using default")

        # Max generations
        gens_input = input("  Max generations (default: auto, recommended: 1000-2000 for Berlin52): ").strip()
        if gens_input:
            try:
                hyperparameters['max_generations'] = int(gens_input)
            except ValueError:
                print("    Invalid, using default")

        if not hyperparameters:
            hyperparameters = None
            print("\n  No customizations provided, using auto-estimated values")
        else:
            print("\n  ✓ Custom hyperparameters configured")

    print()
    print("Generating solver with LLM...")
    print()

    # Initialize solver
    solver = LLMBRKGASolver(
        context_package_path="llm_solver/context",
        framework_path="brkga",
        output_dir="llm_solver"
    )

    # Data file path
    data_file = "brkga/data/berlin52.tsp"

    # Generate and test the solver
    result = solver.solve(
        problem_description=problem_description,
        data_file_path=data_file,
        hyperparameters=hyperparameters,  # Pass custom hyperparameters
        max_iterations=3,
        quick_test_only=False  # Run full optimization
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
        print(f"   Executable: {result.executable_path}")
        print()

        if result.final_result:
            # Note: fitness is negative because TSP is minimization
            # The actual tour distance is the absolute value
            actual_distance = abs(result.final_result.best_fitness)
            print("🏆 Optimization Result:")
            print(f"   Best tour distance: {actual_distance:.2f}")
            print(f"   Generations: {result.final_result.generations}")
            print(f"   Time: {result.final_result.execution_time:.2f}s")
            print()
            print("   Note: Optimal tour for Berlin52 is 7542")
            quality_pct = (actual_distance / 7542.0 - 1.0) * 100
            if quality_pct < 1:
                print(f"   Quality: Excellent! Within {quality_pct:.2f}% of optimal")
            elif quality_pct < 5:
                print(f"   Quality: Very good! Within {quality_pct:.2f}% of optimal")
            else:
                print(f"   Quality: {quality_pct:.2f}% above optimal")

        print()
        print("🚀 To re-run the solver:")
        print(f"   ./{result.executable_path}")
        print()
        print("📄 Solution saved to:")
        print("   llm_solver/results/solution.txt")
        print()
    else:
        print("❌ Failed to generate solver")
        print(f"   Iterations attempted: {result.iterations}")
        if result.config_path:
            print(f"   Last config: {result.config_path}")
        print()
        print("💡 Next steps:")
        print("   1. Check API key: export ANTHROPIC_API_KEY=...")
        print("   2. Review error messages above")

    print("=" * 70)

if __name__ == "__main__":
    main()
