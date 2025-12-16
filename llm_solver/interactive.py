#!/usr/bin/env python3
"""
Interactive LLM BRKGA Solver
Describe your optimization problem in natural language and get a working solver!
"""

from core.llm_brkga_solver import LLMBRKGASolver
from core.problem_analyzer import ProblemAnalyzer
from core.data_parser import DataFileParser
import sys
import os

def get_multiline_input(prompt):
    """Get multi-line input from user."""
    print(prompt)
    print("(Enter your problem description. Type 'DONE' on a new line when finished)")
    print("-" * 70)
    lines = []
    while True:
        try:
            line = input()
            if line.strip() == "DONE":
                break
            lines.append(line)
        except EOFError:
            break
    return "\n".join(lines)

def print_section(title, width=70):
    """Print a formatted section header."""
    print()
    print("=" * width)
    print(title.center(width))
    print("=" * width)
    print()

def print_subsection(title, width=70):
    """Print a formatted subsection header."""
    print()
    print(title)
    print("-" * width)

def main():
    print_section("Interactive LLM BRKGA Solver")

    print("This tool generates custom BRKGA solvers from natural language.")
    print("The AI will ask clarifying questions to understand your problem better.")
    print()

    # Check for problem description as argument
    if len(sys.argv) > 1:
        problem_file = sys.argv[1]
        print(f"Reading problem from: {problem_file}")
        with open(problem_file, 'r') as f:
            problem_description = f.read()
    else:
        # Interactive mode
        print("Describe your optimization problem in natural language.")
        print()
        print("Examples:")
        print("  - TSP: 'I want to travel to all cities starting from city 1")
        print("          and then get back to city 1. I want to visit each")
        print("          city exactly once optimally'")
        print("  - Knapsack: 'Select items to maximize value without exceeding capacity'")
        print("  - Scheduling: 'Assign tasks to machines to minimize makespan'")
        print()

        problem_description = get_multiline_input("Enter your problem:")

    print_subsection("Your Problem Description")
    print(problem_description)
    print("-" * 70)

    # Initialize components
    print()
    print("Initializing AI analyzer...")
    analyzer = ProblemAnalyzer(
        context_package_path="llm_solver/context"
    )
    data_parser = DataFileParser()

    # Step 1: Ask clarifying questions
    print_section("Step 1: Problem Clarification")
    print("Analyzing your problem to identify any ambiguities...")
    print()

    questions = analyzer.ask_clarifying_questions(problem_description)

    clarifying_qa = {}
    if questions:
        print(f"I have {len(questions)} questions to better understand your problem:")
        print()
        for i, question in enumerate(questions, 1):
            print(f"{i}. {question}")
            answer = input(f"   Your answer: ").strip()
            clarifying_qa[question] = answer
            print()
    else:
        print("Your problem description is clear! No clarifying questions needed.")

    # Step 2: Hyperparameters (Optional)
    print_section("Step 2: BRKGA Hyperparameters (Optional)")

    hyperparameters = None
    customize_hyperparams = input("Would you like to customize BRKGA hyperparameters? (y/n, default=n): ").strip().lower()

    if customize_hyperparams in ['y', 'yes']:
        print("\nYou can customize the following hyperparameters (press Enter to use defaults):")
        print("Note: Defaults will be automatically estimated based on your problem.")
        print()

        hyperparameters = {}

        # Population size
        pop_input = input("  Population size (default: auto): ").strip()
        if pop_input:
            try:
                hyperparameters['population_size'] = int(pop_input)
            except ValueError:
                print("    Invalid input, will use default")

        # Elite percentage
        elite_input = input("  Elite percentage 0-1 (default: 0.15): ").strip()
        if elite_input:
            try:
                hyperparameters['elite_percentage'] = float(elite_input)
            except ValueError:
                print("    Invalid input, will use default")

        # Mutant percentage
        mutant_input = input("  Mutant percentage 0-1 (default: 0.15): ").strip()
        if mutant_input:
            try:
                hyperparameters['mutant_percentage'] = float(mutant_input)
            except ValueError:
                print("    Invalid input, will use default")

        # Elite probability (bias)
        bias_input = input("  Elite bias 0-1 (default: 0.7): ").strip()
        if bias_input:
            try:
                hyperparameters['elite_prob'] = float(bias_input)
            except ValueError:
                print("    Invalid input, will use default")

        # Max generations
        gens_input = input("  Max generations (default: auto): ").strip()
        if gens_input:
            try:
                hyperparameters['max_generations'] = int(gens_input)
            except ValueError:
                print("    Invalid input, will use default")

        if not hyperparameters:
            hyperparameters = None  # No customizations provided
        else:
            print("\n✓ Custom hyperparameters configured")

    # Step 3: Check for data file
    print_section("Step 3: Data File (Optional)")

    data_file_path = None
    has_data_file = input("Do you have a data file for this problem? (y/n): ").strip().lower()

    if has_data_file in ['y', 'yes']:
        while True:
            data_file_path = input("Enter the path to your data file: ").strip()

            if not os.path.exists(data_file_path):
                print(f"File not found: {data_file_path}")
                retry = input("Try again? (y/n): ").strip().lower()
                if retry not in ['y', 'yes']:
                    data_file_path = None
                    break
            else:
                # Parse and show preview
                print()
                print("Parsing data file...")
                try:
                    metadata = data_parser.parse_file(data_file_path)
                    print()
                    print("Data File Summary:")
                    print(f"  Format: {metadata.format.value}")
                    print(f"  Size: {metadata.problem_size} items")
                    if metadata.dimension_info:
                        print(f"  Dimensions: {metadata.dimension_info}")
                    print()
                    print("Data Preview:")
                    print("-" * 70)

                    # Handle both list and string previews
                    if isinstance(metadata.data_preview, list):
                        preview_lines = metadata.data_preview[:10]
                        for line in preview_lines:
                            print(f"  {line}")
                        if len(metadata.data_preview) > 10:
                            print("  ...")
                    elif isinstance(metadata.data_preview, str):
                        preview_lines = metadata.data_preview.split('\n')[:10]
                        for line in preview_lines:
                            print(f"  {line}")
                        if len(metadata.data_preview.split('\n')) > 10:
                            print("  ...")
                    else:
                        print("  (No preview available)")

                    print("-" * 70)
                    print()
                    confirm = input("Use this data file? (y/n): ").strip().lower()
                    if confirm not in ['y', 'yes']:
                        data_file_path = None
                    break
                except Exception as e:
                    print(f"Error parsing file: {e}")
                    retry = input("Try a different file? (y/n): ").strip().lower()
                    if retry not in ['y', 'yes']:
                        data_file_path = None
                        break

    # Step 4: Generate solver
    print_section("Step 4: Generating Solver")

    print("Initializing solver...")

    # Initialize solver
    solver = LLMBRKGASolver(
        context_package_path="llm_solver/context",
        framework_path="brkga",
        output_dir="llm_solver"
    )

    print()
    print("Generating solver configuration...")
    print("(This may take a minute as the AI analyzes and generates code)")
    print()

    # Solve the problem with clarifications, hyperparameters, and data file
    result = solver.solve(
        problem_description=problem_description,
        clarifying_qa=clarifying_qa if clarifying_qa else None,
        data_file_path=data_file_path,
        hyperparameters=hyperparameters,
        max_iterations=3,
        quick_test_only=False  # Run FULL optimization in interactive mode
    )

    # Step 5: Results
    print_section("Results")

    if result.success:
        print("SUCCESS! Generated and executed working solver")
        print()
        print("Generated Files:")
        print(f"  Config:     {result.config_path}")
        print(f"  Executable: {result.executable_path}")
        print()

        # Show optimization results
        if result.final_result:
            print("Optimization Results:")
            print(f"  Best Fitness: {result.final_result.best_fitness:.2f}")
            print(f"  Generations: {result.final_result.generations}")
            print(f"  Execution Time: {result.final_result.execution_time:.2f}s")

            if result.final_result.pareto_front_size:
                print(f"  Pareto Front Size: {result.final_result.pareto_front_size}")

        # Check if solution file exists and show preview
        solution_file = "llm_solver/results/solution.txt"
        if os.path.exists(solution_file):
            print()
            print("Solution File: llm_solver/results/solution.txt")
            print()
            print("Solution Preview:")
            print("-" * 70)
            with open(solution_file, 'r') as f:
                content = f.read()

            # Show the solution, excluding the chromosome if present
            lines = content.split('\n')

            # Show solution but skip chromosome lines
            showing = True
            line_count = 0
            for line in lines:
                # Stop at chromosome section if present
                if 'Chromosome' in line or 'BRKGA random keys' in line:
                    break

                print(f"  {line}")
                line_count += 1

                # Limit preview to reasonable size
                if line_count >= 25:
                    if len(lines) > 25:
                        print("  ...")
                        print(f"  (See full solution in {solution_file})")
                    break

            print("-" * 70)

        print()
        print("Next Steps:")
        print(f"  1. View full solution: cat {solution_file}")
        print(f"  2. Re-run with different parameters: ./{result.executable_path}")
        print(f"  3. Modify config if needed: {result.config_path}")
    else:
        print(f"Failed after {result.iterations} iterations")
        print()
        print(f"Last generated config: {result.config_path}")
        print()
        print("Troubleshooting:")
        print("  - Ensure ANTHROPIC_API_KEY is set")
        print("  - Make problem description more specific")
        print("  - Check generated config for compilation errors")

    print()
    print("=" * 70)

if __name__ == "__main__":
    main()
