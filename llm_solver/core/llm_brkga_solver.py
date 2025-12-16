"""
Main orchestrator for the LLM-powered BRKGA solver system.
Coordinates problem analysis, code generation, validation, and execution.
"""

import os
import sys
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from .problem_structures import *
from .problem_analyzer import ProblemAnalyzer
from .code_generator import CodeGenerator
from .validator import Validator
from .execution_manager import ExecutionManager


@dataclass
class SolverSession:
    """Tracks a complete solver generation and execution session."""
    problem_description: str
    problem_structure: Optional[ProblemStructure] = None
    config_path: Optional[str] = None
    executable_path: Optional[str] = None
    validation_results: list = None
    compilation_result: Optional[CompilationResult] = None
    test_result: Optional[ExecutionResult] = None
    final_result: Optional[ExecutionResult] = None
    iterations: int = 0
    max_iterations: int = 3
    success: bool = False  # Overall success flag
    errors: list = None  # List of errors encountered

    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.errors is None:
            self.errors = []

    @property
    def best_fitness(self) -> Optional[float]:
        """Get best fitness from results."""
        if self.final_result and self.final_result.best_fitness is not None:
            return self.final_result.best_fitness
        elif self.test_result and self.test_result.best_fitness is not None:
            return self.test_result.best_fitness
        return None
    
    def summary(self) -> str:
        """Generate session summary."""
        lines = [
            "="*60,
            "SOLVER SESSION SUMMARY",
            "="*60,
            f"Problem: {self.problem_structure.problem_name if self.problem_structure else 'Unknown'}",
            f"Iterations: {self.iterations}",
        ]
        
        if self.config_path:
            lines.append(f"Config: {self.config_path}")
        
        if self.compilation_result:
            status = "✅" if self.compilation_result.success else "❌"
            lines.append(f"Compilation: {status}")
        
        if self.test_result:
            status = "✅" if self.test_result.success else "❌"
            lines.append(f"Quick Test: {status}")
        
        if self.final_result:
            status = "✅" if self.final_result.success else "❌"
            lines.append(f"Full Execution: {status}")
            if self.final_result.best_fitness is not None:
                lines.append(f"Best Fitness: {self.final_result.best_fitness:.6f}")
        
        lines.append("="*60)
        return "\n".join(lines)


class LLMBRKGASolver:
    """Main solver system that orchestrates the complete pipeline."""
    
    def __init__(self, 
                 context_package_path: str = "llm_solver/context",
                 framework_path: str = "brkga",
                 output_dir: str = "llm_solver",
                 api_key: Optional[str] = None):
        """
        Initialize the LLM BRKGA solver system.
        
        Args:
            context_package_path: Path to BRKGA context package
            framework_path: Path to BRKGA framework
            output_dir: Directory for generated files
            api_key: Anthropic API key
        """
        self.context_path = context_package_path
        self.framework_path = framework_path
        self.output_dir = output_dir
        
        # Initialize components
        self.analyzer = ProblemAnalyzer(context_package_path, api_key)
        self.generator = CodeGenerator(context_package_path, api_key)
        self.validator = Validator(framework_path)
        self.executor = ExecutionManager(framework_path)
        
        # Create output directories
        os.makedirs(os.path.join(output_dir, "generated"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
        
        print("🚀 LLM BRKGA Solver initialized")
        print(f"   Context: {context_package_path}")
        print(f"   Framework: {framework_path}")
        print(f"   Output: {output_dir}")
    
    def solve(self,
              problem_description: str,
              clarifying_qa: Optional[Dict[str, str]] = None,
              data_file_path: Optional[str] = None,
              data_files: Optional[Dict[str, str]] = None,
              hyperparameters: Optional[Dict[str, Any]] = None,
              quick_test_only: bool = False,
              max_iterations: int = 3,
              verbose: bool = True) -> SolverSession:
        """
        Main entry point: solve an optimization problem from natural language.

        Args:
            problem_description: Natural language problem description
            clarifying_qa: Optional clarifying Q&A
            data_file_path: Optional path to data file (DEPRECATED - use data_files instead)
            data_files: Optional dict of named data files {"job_times": "path/to/job_times.csv",
                                                          "travel_times": "path/to/travel.csv"}
            hyperparameters: Optional dict with BRKGA hyperparameters:
                - population_size: int (default: auto-estimated)
                - elite_percentage: float 0-1 (default: 0.15)
                - mutant_percentage: float 0-1 (default: 0.15)
                - elite_prob: float 0-1 (default: 0.7)
                - max_generations: int (default: auto-estimated)
            quick_test_only: If True, only run quick test, not full optimization
            max_iterations: Maximum refinement iterations
            verbose: Print detailed progress

        Returns:
            SolverSession with complete results
        """
        
        session = SolverSession(
            problem_description=problem_description,
            max_iterations=max_iterations
        )

        # Handle backward compatibility: merge data_file_path into data_files
        merged_data_files = {}
        if data_files:
            merged_data_files.update(data_files)
        if data_file_path:
            # Add single data file with default name "primary" if not already in data_files
            if "primary" not in merged_data_files:
                merged_data_files["primary"] = data_file_path

        print("\n" + "="*60)
        print("LLM BRKGA SOLVER - PROBLEM SOLVING PIPELINE")
        print("="*60)

        if merged_data_files:
            print(f"\n📁 Data files ({len(merged_data_files)}):")
            for name, path in merged_data_files.items():
                print(f"   - {name}: {path}")

        # Step 1: Analyze problem
        print("\n📋 STEP 1: PROBLEM ANALYSIS")
        print("-" * 60)

        try:
            # Check if we need clarifying questions
            if not clarifying_qa and verbose:
                questions = self.analyzer.ask_clarifying_questions(problem_description)
                if questions:
                    print("🤔 Clarifying questions needed:")
                    for i, q in enumerate(questions, 1):
                        print(f"   {i}. {q}")
                    print("\n   Please provide answers and call solve() again with clarifying_qa parameter")
                    return session

            # Analyze the problem (with optional data file(s))
            session.problem_structure = self.analyzer.analyze_problem(
                problem_description,
                clarifying_qa,
                data_file_path,  # Keep for backward compatibility
                merged_data_files  # NEW: pass all data files
            )
            
            print(f"\n✅ Problem analyzed:")
            print(session.problem_structure.summary())
            
            # Estimate complexity
            complexity = self.analyzer.estimate_problem_complexity(session.problem_structure)
            print(f"\n📊 Complexity estimates:")
            for key, value in complexity.items():
                print(f"   {key}: {value}")

            # Get default hyperparameters and merge with user-provided ones
            default_hyperparameters = self.analyzer.get_default_hyperparameters(session.problem_structure)
            if hyperparameters:
                # Merge user hyperparameters with defaults (user values take precedence)
                final_hyperparameters = {**default_hyperparameters, **hyperparameters}
                print(f"\n⚙️  BRKGA Hyperparameters (user-customized):")
            else:
                final_hyperparameters = default_hyperparameters
                print(f"\n⚙️  BRKGA Hyperparameters (auto-estimated):")

            for key, value in final_hyperparameters.items():
                if "percentage" in key:
                    print(f"   {key}: {value} ({value*100:.0f}%)")
                else:
                    print(f"   {key}: {value}")
            
        except Exception as e:
            print(f"\n❌ Problem analysis failed: {str(e)}")
            session.errors.append(str(e))
            return session
        
        # Step 2: Generate code (with refinement loop)
        print("\n💻 STEP 2: CODE GENERATION")
        print("-" * 60)
        
        for iteration in range(max_iterations):
            session.iterations = iteration + 1
            
            if iteration > 0:
                print(f"\n🔄 Refinement iteration {iteration + 1}/{max_iterations}")
            
            try:
                # Generate config
                config_name = session.problem_structure.problem_name.lower().replace(" ", "_")
                session.config_path = os.path.join(
                    self.output_dir, 
                    "generated",
                    f"{config_name}_config.hpp"
                )
                
                if iteration == 0:
                    # Initial generation
                    self.generator.generate_config(
                        session.problem_structure,
                        session.config_path,
                        hyperparameters=final_hyperparameters
                    )
                else:
                    # Refinement based on previous errors
                    error_msg = self._collect_error_messages(session)
                    session.config_path = self.generator.refine_config(
                        session.config_path,
                        error_msg,
                        session.problem_structure,
                        hyperparameters=final_hyperparameters
                    )
                
                # Step 3: Validate
                print("\n🔍 STEP 3: VALIDATION")
                print("-" * 60)
                
                valid, validation_results = self.validator.full_validation(
                    session.config_path,
                    session.problem_structure
                )
                
                session.validation_results = validation_results
                
                if not valid:
                    print(f"\n⚠️  Validation failed, will try to refine...")
                    continue
                
                # Step 4: Compile
                print("\n🔨 STEP 4: COMPILATION")
                print("-" * 60)
                
                session.executable_path = os.path.join(
                    self.output_dir,
                    "generated",
                    f"{config_name}_solver"
                )
                
                session.compilation_result = self.executor.compile_solver(
                    session.config_path,
                    session.executable_path,
                    data_files=merged_data_files if merged_data_files else None
                )
                
                if not session.compilation_result.success:
                    print(f"\n⚠️  Compilation failed, will try to refine...")
                    continue
                
                # Step 5: Quick test
                print("\n🧪 STEP 5: QUICK TEST")
                print("-" * 60)

                # Scale timeout based on population size (larger populations need more time)
                # Base timeout: 30s for pop_size=200, scale linearly
                base_timeout = 30
                base_pop_size = 200
                pop_size = final_hyperparameters.get('population_size', base_pop_size)
                scaled_timeout = int(base_timeout * (pop_size / base_pop_size))
                # Ensure minimum of 30s and maximum of 300s (5 minutes)
                scaled_timeout = max(30, min(300, scaled_timeout))

                session.test_result = self.executor.run_quick_test(
                    session.executable_path,
                    timeout=scaled_timeout
                )
                
                if not session.test_result.success:
                    print(f"\n⚠️  Quick test failed, will try to refine...")
                    continue
                
                # Success! Break out of refinement loop
                print(f"\n✅ Configuration validated and tested successfully!")
                break
                
            except Exception as e:
                print(f"\n❌ Error in iteration {iteration + 1}: {str(e)}")
                if iteration == max_iterations - 1:
                    print(f"\n❌ Max iterations reached, giving up")
                    return session
                continue
        
        # Check if we succeeded
        if not session.test_result or not session.test_result.success:
            print(f"\n❌ Failed to generate working solver after {max_iterations} iterations")
            session.success = False
            return session

        # Mark quick test success
        session.success = True

        # Step 6: Full optimization (if requested)
        if not quick_test_only:
            print("\n🚀 STEP 6: FULL OPTIMIZATION")
            print("-" * 60)

            try:
                session.final_result = self.executor.run_full_optimization(
                    session.executable_path
                )

                if session.final_result.success:
                    print(f"\n✅ Optimization completed successfully!")
                    session.success = True
                    self._print_results(session)
                else:
                    print(f"\n⚠️  Optimization encountered issues")
                    session.success = False

            except Exception as e:
                print(f"\n❌ Optimization failed: {str(e)}")
                session.errors.append(str(e))
                session.success = False

        # Print final summary
        print(f"\n{session.summary()}")

        return session
    
    def _collect_error_messages(self, session: SolverSession) -> str:
        """Collect all error messages for refinement."""
        errors = []
        
        if session.validation_results:
            for result in session.validation_results:
                if hasattr(result, 'errors'):
                    errors.extend(result.errors)
        
        if session.compilation_result and not session.compilation_result.success:
            errors.extend(session.compilation_result.errors)
        
        if session.test_result and not session.test_result.success:
            errors.extend(session.test_result.errors)
        
        return "\n".join(errors)
    
    def _print_results(self, session: SolverSession):
        """Print optimization results in a nice format."""
        
        result = session.final_result
        if not result:
            return
        
        print("\n" + "="*60)
        print("OPTIMIZATION RESULTS")
        print("="*60)
        
        if result.best_fitness is not None:
            print(f"Best Fitness: {result.best_fitness:.6f}")
        
        if result.pareto_front_size is not None:
            print(f"Pareto Front Size: {result.pareto_front_size}")
        
        print(f"Generations: {result.generations}")
        print(f"Time: {result.execution_time:.2f}s")
        
        if result.solution_valid:
            print("Solution: ✅ Valid")
        
        print("="*60)
    
    def interactive_solve(self):
        """Interactive mode for solving problems."""
        
        print("\n" + "="*60)
        print("LLM BRKGA SOLVER - INTERACTIVE MODE")
        print("="*60)
        print("\nDescribe your optimization problem in natural language.")
        print("Type 'quit' to exit.\n")
        
        while True:
            try:
                description = input("Problem description: ").strip()
                
                if description.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not description:
                    continue
                
                # Solve the problem
                session = self.solve(description)
                
                # Ask if user wants to try another
                again = input("\nSolve another problem? (y/n): ").strip().lower()
                if again != 'y':
                    break
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                continue


def main():
    """Main entry point for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="LLM-powered BRKGA solver for optimization problems"
    )
    
    parser.add_argument(
        'problem',
        nargs='?',
        help='Problem description in natural language'
    )
    
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Only run quick test, not full optimization'
    )
    
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=3,
        help='Maximum refinement iterations (default: 3)'
    )
    
    parser.add_argument(
        '--api-key',
        help='Anthropic API key (or set ANTHROPIC_API_KEY env var)'
    )
    
    args = parser.parse_args()
    
    # Create solver
    solver = LLMBRKGASolver(api_key=args.api_key)
    
    if args.interactive:
        solver.interactive_solve()
    elif args.problem:
        session = solver.solve(
            args.problem,
            quick_test_only=args.quick_test,
            max_iterations=args.max_iterations
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
