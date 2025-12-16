"""
Example: Using Multiple Data Files with LLM BRKGA Solver
=========================================================

This example demonstrates how to solve optimization problems that require
multiple input data files, such as:
- Job scheduling with separate job times and travel times
- Vehicle routing with distance matrix and time windows
- Multi-objective problems with separate cost and constraint data

Author: LLM BRKGA Solver Team
Date: 2025
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_solver.core.llm_brkga_solver import LLMBRKGASolver


def example_1_job_shop_scheduling():
    """
    Example 1: Job Shop Scheduling with Multiple Data Files

    This problem requires:
    - job_times.csv: Processing times for each job on each machine
    - setup_times.csv: Setup times between different job types
    """

    print("\n" + "="*70)
    print("EXAMPLE 1: Job Shop Scheduling with Multiple Data Files")
    print("="*70)

    # Problem description
    problem_description = """
    I need to solve a job shop scheduling problem where:
    - We have 10 jobs that need to be processed on 5 machines
    - Each job has different processing times on different machines
    - There are setup times required when switching between certain job types
    - Goal: Minimize total completion time (makespan)
    - Constraints: Each machine can only process one job at a time
    """

    # Multiple data files
    data_files = {
        "job_times": "data/job_processing_times.csv",
        "setup_times": "data/machine_setup_times.csv"
    }

    # Note: You would need to create these CSV files for a real run
    # For this example, we're just showing the API usage

    print("\nProblem: Job Shop Scheduling")
    print("\nData files:")
    for name, path in data_files.items():
        print(f"  - {name}: {path}")

    # Create solver
    solver = LLMBRKGASolver()

    # Solve with multiple data files
    # Uncomment the following to run:
    # result = solver.solve(
    #     problem_description=problem_description,
    #     data_files=data_files,  # NEW: Pass dictionary of data files
    #     hyperparameters={
    #         "population_size": 200,
    #         "max_generations": 1000
    #     }
    # )

    print("\n✓ API usage example shown above")


def example_2_vehicle_routing_with_time_windows():
    """
    Example 2: Vehicle Routing with Time Windows

    This problem requires:
    - distances.csv: Distance matrix between locations
    - time_windows.csv: Allowed delivery time windows for each customer
    - demands.csv: Demand/capacity requirements
    """

    print("\n" + "="*70)
    print("EXAMPLE 2: Vehicle Routing with Time Windows")
    print("="*70)

    problem_description = """
    Vehicle routing problem with time windows:
    - 50 customer locations to visit
    - Each customer has a specific time window for delivery
    - Each customer has a demand that must be satisfied
    - Fleet of 5 vehicles with capacity constraint
    - Minimize total travel distance while respecting time windows and capacity
    """

    # Multiple data files for VRP
    data_files = {
        "distances": "data/customer_distances.csv",
        "time_windows": "data/delivery_time_windows.csv",
        "demands": "data/customer_demands.csv"
    }

    print("\nProblem: Vehicle Routing with Time Windows")
    print("\nData files:")
    for name, path in data_files.items():
        print(f"  - {name}: {path}")

    print("\n✓ Example configuration shown")


def example_3_backward_compatibility():
    """
    Example 3: Backward Compatibility - Single Data File

    Shows that the old API with data_file_path still works
    """

    print("\n" + "="*70)
    print("EXAMPLE 3: Backward Compatibility - Single Data File")
    print("="*70)

    problem_description = """
    Traveling Salesman Problem:
    - Visit all cities exactly once
    - Return to starting city
    - Minimize total travel distance
    """

    # Old way: single data file (still works!)
    data_file_path = "../brkga/data/berlin52.tsp"

    print("\nProblem: TSP")
    print(f"Data file: {data_file_path}")

    solver = LLMBRKGASolver()

    # Old API still works
    # result = solver.solve(
    #     problem_description=problem_description,
    #     data_file_path=data_file_path  # Old parameter still supported
    # )

    print("\n✓ Backward compatibility maintained")


def example_4_mixed_usage():
    """
    Example 4: Using both data_files and data_file_path

    If both are provided, they are merged together
    """

    print("\n" + "="*70)
    print("EXAMPLE 4: Mixed Usage (Not Recommended)")
    print("="*70)

    problem_description = "Multi-file optimization problem"

    # If you provide both, they get merged
    data_file_path = "data/primary_data.csv"
    data_files = {
        "secondary": "data/secondary_data.csv",
        "tertiary": "data/tertiary_data.csv"
    }

    # Result: All three files will be available
    # - "primary": data/primary_data.csv (from data_file_path)
    # - "secondary": data/secondary_data.csv
    # - "tertiary": data/tertiary_data.csv

    print("\nWhen both parameters are provided:")
    print(f"  data_file_path → Becomes 'primary': {data_file_path}")
    print("  data_files →")
    for name, path in data_files.items():
        print(f"    - {name}: {path}")

    print("\n⚠️  Note: It's clearer to use just data_files parameter")


def main():
    """Run all examples."""

    print("\n" + "="*70)
    print("MULTIPLE DATA FILES - USAGE EXAMPLES")
    print("="*70)
    print()
    print("This script shows how to use the new multiple data files feature.")
    print("The examples demonstrate different use cases and API patterns.")
    print()

    # Run examples
    example_1_job_shop_scheduling()
    example_2_vehicle_routing_with_time_windows()
    example_3_backward_compatibility()
    example_4_mixed_usage()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print()
    print("Key Points:")
    print("  1. Use data_files parameter to pass multiple named data files")
    print("  2. data_files is a dictionary: {'name': 'path', ...}")
    print("  3. Old data_file_path parameter still works (backward compatible)")
    print("  4. Maximum 10 data files per problem (configurable in config.yaml)")
    print("  5. All files are parsed and metadata is passed to the LLM")
    print()
    print("New API:")
    print("  solver.solve(")
    print("      problem_description='...',")
    print("      data_files={")
    print("          'job_times': 'path/to/jobs.csv',")
    print("          'travel_times': 'path/to/travel.csv'")
    print("      }")
    print("  )")
    print()
    print("="*70)


if __name__ == "__main__":
    main()
