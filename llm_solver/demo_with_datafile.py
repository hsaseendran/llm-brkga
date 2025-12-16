#!/usr/bin/env python3
"""
Demo script showing how to use the LLM BRKGA solver with data files.

This script demonstrates:
1. How to provide a data file path (TSP, VRP, CSV, etc.)
2. How the LLM automatically parses and understands the data
3. How the generated code includes data loading logic
"""

import os
import sys

# Get the script's directory and project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Add llm_solver to path if needed
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from core.llm_brkga_solver import LLMBRKGASolver


def demo_tsp_with_datafile():
    """Demo: Solving TSP with a TSPLIB format data file."""

    print("=" * 70)
    print("DEMO: Traveling Salesman Problem with Data File")
    print("=" * 70)

    # Initialize solver with correct paths
    context_path = os.path.join(SCRIPT_DIR, "context")
    output_path = os.path.join(SCRIPT_DIR, "generated")

    solver = LLMBRKGASolver(
        context_package_path=context_path,
        output_dir=output_path
    )

    # Problem description (natural language)
    problem_description = """
    Solve the Traveling Salesman Problem (TSP) using the provided TSPLIB data file.

    The goal is to find the shortest tour that visits all cities exactly once and
    returns to the starting city.

    The data file contains city coordinates in TSPLIB format with EUC_2D distances.
    """

    # Path to data file (absolute path)
    data_file = os.path.join(PROJECT_ROOT, "brkga", "data", "berlin52.tsp")

    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        print("Please make sure the berlin52.tsp file exists in brkga/data/")
        return

    print(f"\n📁 Data file: {data_file}")
    print("🤖 Solving with LLM-generated BRKGA solver...\n")

    # Solve the problem (LLM will parse the data file automatically)
    result = solver.solve(
        problem_description=problem_description,
        data_file_path=data_file,  # NEW: Pass data file path
        quick_test_only=True,
        verbose=True
    )

    if result.success:
        print("\n✅ SUCCESS!")
        print(f"Best tour length: {result.best_fitness:.2f}")
    else:
        print("\n❌ FAILED")
        print("Errors:", result.errors)


def demo_vrp_with_datafile():
    """Demo: Solving VRP with a data file."""

    print("\n" + "=" * 70)
    print("DEMO: Vehicle Routing Problem with Data File")
    print("=" * 70)

    context_path = os.path.join(SCRIPT_DIR, "context")
    output_path = os.path.join(SCRIPT_DIR, "generated")

    solver = LLMBRKGASolver(
        context_package_path=context_path,
        output_dir=output_path
    )

    problem_description = """
    Solve the Capacitated Vehicle Routing Problem (CVRP).

    We have a fleet of vehicles with limited capacity that must deliver goods
    to multiple customers. Each customer has a specific demand. The goal is to
    minimize the total distance traveled while ensuring:
    1. Each customer is visited exactly once
    2. Vehicle capacity constraints are not violated
    3. All vehicles start and end at the depot
    """

    # Example VRP data file (if you have one)
    data_file = os.path.join(PROJECT_ROOT, "brkga", "data", "vrp_example.vrp")

    if not os.path.exists(data_file):
        print(f"⚠️  Data file not found: {data_file}")
        print("Skipping VRP demo (create a .vrp file to try this)")
        return

    print(f"\n📁 Data file: {data_file}")
    print("🤖 Solving with LLM-generated BRKGA solver...\n")

    result = solver.solve(
        problem_description=problem_description,
        data_file_path=data_file,
        quick_test_only=True,
        verbose=True
    )

    if result.success:
        print("\n✅ SUCCESS!")
        print(f"Best solution cost: {result.best_fitness:.2f}")


def demo_csv_matrix():
    """Demo: Using CSV matrix data (e.g., distance matrix)."""

    print("\n" + "=" * 70)
    print("DEMO: Problem with CSV Distance Matrix")
    print("=" * 70)

    context_path = os.path.join(SCRIPT_DIR, "context")
    output_path = os.path.join(SCRIPT_DIR, "generated")

    solver = LLMBRKGASolver(
        context_package_path=context_path,
        output_dir=output_path
    )

    problem_description = """
    Solve a TSP with time windows using pre-computed distance and time matrices.

    The CSV files contain:
    - Distance matrix: travel distances between all pairs of locations
    - Time matrix: travel times between all pairs of locations

    Find the tour that minimizes total travel time while respecting time windows.
    """

    # CSV data files
    distance_file = os.path.join(PROJECT_ROOT, "brkga", "data", "berlin52_TSPJ_JT.csv")
    time_file = os.path.join(PROJECT_ROOT, "brkga", "data", "berlin52_TSPJ_TT.csv")

    if not os.path.exists(distance_file):
        print(f"⚠️  Data file not found: {distance_file}")
        print("Skipping CSV demo")
        return

    print(f"\n📁 Distance matrix: {distance_file}")
    print(f"📁 Time matrix: {time_file}")
    print("🤖 Solving with LLM-generated BRKGA solver...\n")

    # For multi-file problems, we can mention additional files in the description
    enhanced_description = f"""{problem_description}

    DATA FILES:
    - Distance matrix: {distance_file}
    - Time matrix: {time_file}
    """

    result = solver.solve(
        problem_description=enhanced_description,
        data_file_path=distance_file,  # Primary data file
        quick_test_only=True,
        verbose=True
    )

    if result.success:
        print("\n✅ SUCCESS!")


def demo_custom_txt_format():
    """Demo: Using a custom text format data file."""

    print("\n" + "=" * 70)
    print("DEMO: Custom Text Format Data File")
    print("=" * 70)

    context_path = os.path.join(SCRIPT_DIR, "context")
    output_path = os.path.join(SCRIPT_DIR, "generated")

    solver = LLMBRKGASolver(
        context_package_path=context_path,
        output_dir=output_path
    )

    problem_description = """
    Solve a knapsack problem with items defined in a text file.

    The file format is:
    - First line: number of items
    - Following lines: value weight (space-separated)

    Goal: Maximize total value while staying within capacity constraint.
    """

    data_file = os.path.join(PROJECT_ROOT, "brkga", "data", "knapsack_100.txt")

    if not os.path.exists(data_file):
        print(f"⚠️  Data file not found: {data_file}")
        print("Skipping custom format demo")
        return

    print(f"\n📁 Data file: {data_file}")
    print("🤖 Solving with LLM-generated BRKGA solver...\n")

    result = solver.solve(
        problem_description=problem_description,
        data_file_path=data_file,
        quick_test_only=True,
        verbose=True
    )

    if result.success:
        print("\n✅ SUCCESS!")
        print(f"Best value: {result.best_fitness:.2f}")


def demo_data_file_parser_standalone():
    """Demo: Using the data file parser independently."""

    print("\n" + "=" * 70)
    print("DEMO: Standalone Data File Parser")
    print("=" * 70)

    from core.data_parser import DataFileParser

    parser = DataFileParser()

    # Parse a TSP file
    tsp_file = os.path.join(PROJECT_ROOT, "brkga", "data", "berlin52.tsp")

    if os.path.exists(tsp_file):
        print(f"\n📁 Parsing: {tsp_file}")
        metadata = parser.parse_file(tsp_file)

        print(f"\n📊 Metadata extracted:")
        print(f"   Format: {metadata.format.value}")
        print(f"   Problem size: {metadata.problem_size}")
        print(f"   Edge weight type: {metadata.edge_weight_type}")
        print(f"   Dimension info: {metadata.dimension_info}")

        # Generate LLM context
        context = parser.generate_llm_context(metadata)
        print(f"\n📝 LLM Context generated:")
        print(context)
    else:
        print(f"⚠️  File not found: {tsp_file}")


if __name__ == "__main__":
    print("🚀 LLM BRKGA Solver - Data File Parsing Demo\n")

    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("⚠️  WARNING: ANTHROPIC_API_KEY environment variable not set")
        print("   The LLM-based demos will not work without an API key.")
        print("   However, the standalone data parser demo will still work.\n")
        print("   To run the full demo, set your API key:")
        print("   export ANTHROPIC_API_KEY=your_key_here\n")

    # Run demos
    try:
        if api_key:
            # Main TSP demo (requires API key)
            demo_tsp_with_datafile()

            # Optional: Other format demos
            # demo_vrp_with_datafile()
            # demo_csv_matrix()
            # demo_custom_txt_format()
        else:
            print("Skipping LLM-based demos (no API key)\n")

        # Standalone parser demo (no API key needed)
        demo_data_file_parser_standalone()

        print("\n" + "=" * 70)
        print("✅ Demo completed!")
        print("=" * 70)

        if not api_key:
            print("\n💡 Tip: Set ANTHROPIC_API_KEY to try the full LLM-based demos")

    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
