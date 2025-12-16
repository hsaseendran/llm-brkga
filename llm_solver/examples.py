#!/usr/bin/env python3
"""
Example usage of the LLM BRKGA Solver system.
Demonstrates how to solve various optimization problems.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import LLMBRKGASolver


def example_1_simple_knapsack():
    """Example 1: Simple knapsack problem."""
    
    print("\n" + "="*70)
    print("EXAMPLE 1: SIMPLE KNAPSACK PROBLEM")
    print("="*70)
    
    problem = """
    I have a knapsack problem with 20 items. Each item has a weight and a value.
    The knapsack has a capacity of 50 units. I want to maximize the total value
    while staying within the weight capacity.
    
    Item details:
    - 20 items total
    - Weights range from 1 to 10
    - Values range from 5 to 50
    - Capacity constraint: 50 units
    """
    
    solver = LLMBRKGASolver()
    session = solver.solve(problem, quick_test_only=True)
    
    return session


def example_2_tsp():
    """Example 2: Traveling Salesman Problem."""
    
    print("\n" + "="*70)
    print("EXAMPLE 2: TRAVELING SALESMAN PROBLEM")
    print("="*70)
    
    problem = """
    I need to solve a Traveling Salesman Problem (TSP) with 30 cities.
    A salesman needs to visit all cities exactly once and return to the
    starting city, minimizing the total travel distance.
    
    Problem details:
    - 30 cities with x,y coordinates
    - Need to visit each city exactly once
    - Return to starting city
    - Minimize total Euclidean distance
    """
    
    solver = LLMBRKGASolver()
    session = solver.solve(problem, quick_test_only=True)
    
    return session


def example_3_job_scheduling():
    """Example 3: Job scheduling problem."""
    
    print("\n" + "="*70)
    print("EXAMPLE 3: JOB SCHEDULING PROBLEM")
    print("="*70)
    
    problem = """
    I have a job scheduling problem where I need to schedule 15 jobs on 3 machines.
    Each job has a processing time and a deadline. Jobs must be completed before
    their deadlines.
    
    Problem details:
    - 15 jobs to schedule
    - 3 parallel machines
    - Each job has processing time (5-30 minutes)
    - Each job has a deadline
    - Minimize total tardiness (lateness beyond deadline)
    - Hard constraint: jobs can't be interrupted once started
    """
    
    solver = LLMBRKGASolver()
    session = solver.solve(problem, quick_test_only=True)
    
    return session


def example_4_multi_objective_tsp():
    """Example 4: Multi-objective TSP."""
    
    print("\n" + "="*70)
    print("EXAMPLE 4: MULTI-OBJECTIVE TSP")
    print("="*70)
    
    problem = """
    I need to solve a multi-objective Traveling Salesman Problem with 25 cities.
    I want to optimize two objectives:
    
    1. Minimize total travel distance
    2. Minimize total travel time (some roads are faster than others)
    
    Problem details:
    - 25 cities
    - Each edge has both a distance and a travel time
    - Want Pareto front of non-dominated solutions
    - Multi-objective optimization needed
    """
    
    solver = LLMBRKGASolver()
    session = solver.solve(problem, quick_test_only=True)
    
    return session


def example_5_bin_packing():
    """Example 5: Bin packing problem."""
    
    print("\n" + "="*70)
    print("EXAMPLE 5: BIN PACKING PROBLEM")
    print("="*70)
    
    problem = """
    I have a bin packing problem where I need to pack 40 items into bins.
    Each item has a size, and each bin has capacity 100. I want to minimize
    the number of bins used.
    
    Problem details:
    - 40 items with sizes ranging from 10 to 80
    - Bin capacity: 100
    - Minimize number of bins needed
    - All items must be packed
    """
    
    solver = LLMBRKGASolver()
    session = solver.solve(problem, quick_test_only=True)
    
    return session


def example_6_custom_problem():
    """Example 6: Custom user-defined problem."""
    
    print("\n" + "="*70)
    print("EXAMPLE 6: CUSTOM PROBLEM (USER INPUT)")
    print("="*70)
    
    print("\nPlease describe your optimization problem:")
    problem = input("> ")
    
    if not problem.strip():
        print("No problem provided, skipping...")
        return None
    
    solver = LLMBRKGASolver()
    session = solver.solve(problem, quick_test_only=True)
    
    return session


def run_all_examples():
    """Run all examples."""
    
    examples = [
        ("Simple Knapsack", example_1_simple_knapsack),
        ("TSP", example_2_tsp),
        ("Job Scheduling", example_3_job_scheduling),
        ("Multi-objective TSP", example_4_multi_objective_tsp),
        ("Bin Packing", example_5_bin_packing),
    ]
    
    results = {}
    
    print("\n" + "="*70)
    print("RUNNING ALL EXAMPLES")
    print("="*70)
    
    for name, example_func in examples:
        try:
            print(f"\n\nRunning: {name}")
            session = example_func()
            results[name] = session
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            break
        except Exception as e:
            print(f"\nError in {name}: {str(e)}")
            results[name] = None
    
    # Summary
    print("\n\n" + "="*70)
    print("SUMMARY OF ALL EXAMPLES")
    print("="*70)
    
    for name, session in results.items():
        if session is None:
            status = "❌ Failed"
        elif session.test_result and session.test_result.success:
            status = "✅ Success"
        elif session.compilation_result and session.compilation_result.success:
            status = "⚠️  Compiled but test failed"
        else:
            status = "❌ Failed to compile"
        
        print(f"{name:30s} {status}")
    
    print("="*70)


def interactive_mode():
    """Run in interactive mode."""
    
    solver = LLMBRKGASolver()
    solver.interactive_solve()


def main():
    """Main entry point."""
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == 'all':
            run_all_examples()
        elif mode == 'interactive':
            interactive_mode()
        elif mode in ['1', 'knapsack']:
            example_1_simple_knapsack()
        elif mode in ['2', 'tsp']:
            example_2_tsp()
        elif mode in ['3', 'scheduling']:
            example_3_job_scheduling()
        elif mode in ['4', 'multi']:
            example_4_multi_objective_tsp()
        elif mode in ['5', 'binpack']:
            example_5_bin_packing()
        elif mode in ['6', 'custom']:
            example_6_custom_problem()
        else:
            print(f"Unknown mode: {mode}")
            print_usage()
    else:
        print_usage()


def print_usage():
    """Print usage information."""
    
    print("""
LLM BRKGA Solver - Examples

Usage:
    python examples.py <mode>

Modes:
    all            - Run all examples
    interactive    - Interactive problem solving
    1, knapsack    - Simple knapsack problem
    2, tsp         - Traveling salesman problem
    3, scheduling  - Job scheduling problem
    4, multi       - Multi-objective TSP
    5, binpack     - Bin packing problem
    6, custom      - Enter your own problem

Examples:
    python examples.py all
    python examples.py interactive
    python examples.py tsp
    """)


if __name__ == "__main__":
    main()
