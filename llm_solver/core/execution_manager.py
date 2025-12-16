"""
Execution manager for compiling and running generated BRKGA solvers.
"""

import os
import subprocess
import tempfile
import time
import re
from typing import Optional, Dict, Any
from .problem_structures import *


class ExecutionManager:
    """Manages compilation and execution of BRKGA solvers."""
    
    def __init__(self, framework_path: str = "brkga"):
        """
        Initialize execution manager.

        Args:
            framework_path: Path to BRKGA framework files
        """
        self.framework_path = os.path.abspath(framework_path)
        self.nvcc_path = "nvcc"
        self.default_arch = "sm_75"
    
    def compile_solver(self, config_path: str,
                      output_executable: str,
                      optimization_level: str = "O3",
                      data_files: Optional[Dict[str, str]] = None) -> CompilationResult:
        """
        Compile a complete BRKGA solver.

        Args:
            config_path: Path to config file
            output_executable: Path for output executable
            optimization_level: Optimization level (O0, O2, O3)
            data_files: Optional dict of data files {"name": "path"}

        Returns:
            CompilationResult
        """
        
        print(f"\n🔨 Compiling solver...")
        print(f"   Config: {config_path}")
        print(f"   Output: {output_executable}")
        
        start_time = time.time()

        # Create main file
        main_file = self._create_main_file(config_path, data_files)
        
        try:
            # Compile command
            compile_cmd = [
                self.nvcc_path,
                '-std=c++17',
                f'-arch={self.default_arch}',
                f'-{optimization_level}',
                '-Xcompiler', '-fopenmp',
                '-I', self.framework_path,
                main_file,
                '-o', output_executable,
                '-lcurand', '-lcudart', '-lpthread'
            ]
            
            print(f"   Command: {' '.join(compile_cmd)}")
            
            result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            compilation_time = time.time() - start_time
            
            errors = self._parse_errors(result.stderr)
            warnings = self._parse_warnings(result.stderr)
            
            success = result.returncode == 0 and os.path.exists(output_executable)
            
            if success:
                print(f"✅ Compilation successful ({compilation_time:.2f}s)")
                print(f"   Executable: {output_executable}")
            else:
                print(f"❌ Compilation failed ({compilation_time:.2f}s)")
                if errors:
                    print("   Errors:")
                    for err in errors[:5]:  # Show first 5 errors
                        print(f"      - {err}")
            
            return CompilationResult(
                success=success,
                output=result.stdout + result.stderr,
                errors=errors,
                warnings=warnings,
                compilation_time=compilation_time
            )
            
        except subprocess.TimeoutExpired:
            return CompilationResult(
                success=False,
                output="Compilation timeout",
                errors=["Compilation took too long (>120s)"],
                compilation_time=120.0
            )
        except Exception as e:
            return CompilationResult(
                success=False,
                output=str(e),
                errors=[f"Compilation error: {str(e)}"],
                compilation_time=time.time() - start_time
            )
        finally:
            # Cleanup temporary main file
            try:
                os.unlink(main_file)
            except:
                pass
    
    def _create_main_file(self, config_path: str, data_files: Optional[Dict[str, str]] = None) -> str:
        """Create a main.cu file for the config."""

        # Extract config class name from file
        with open(config_path, 'r') as f:
            content = f.read()

        # Look for class that extends BRKGAConfig specifically (not LocalSearch or other classes)
        class_match = re.search(r'class\s+(\w+)\s*:\s*public\s+BRKGAConfig', content)
        if not class_match:
            # Fallback: look for any class ending with "Config"
            class_match = re.search(r'class\s+(\w+Config)\s*:', content)
        if not class_match:
            # Last resort: first class (but this might pick wrong one like LocalSearch)
            class_match = re.search(r'class\s+(\w+)\s*:', content)
        config_name = class_match.group(1) if class_match else "CustomConfig"

        # Check if the config has a static factory method
        # Look for methods that return unique_ptr
        has_create_default_unique = 'std::unique_ptr<' in content and 'create_default()' in content
        has_create_standard_unique = 'std::unique_ptr<' in content and 'create_standard_instance()' in content
        has_create_from_file = 'std::unique_ptr<' in content and 'create_from_file' in content
        has_create_from_files = 'std::unique_ptr<' in content and 'create_from_files' in content

        # Legacy: methods that return by value
        has_create_default = 'create_default()' in content and not has_create_default_unique
        has_create_standard = 'create_standard_instance()' in content and not has_create_standard_unique
        has_load_from_file = 'load_from_file' in content

        # Create config initialization code
        if has_create_from_files and data_files:
            # Config expects multiple files - extract parameter names from factory method
            # Use DOTALL flag to match across newlines
            files_pattern = r'create_from_files\s*\(((?:\s*const\s+std::string\s*&\s*\w+\s*,?\s*)+)\)'
            match = re.search(files_pattern, content, re.DOTALL)

            if match:
                # Extract parameter names - look for identifiers after '&' (reference parameters)
                params_str = match.group(1)
                # Match parameter names that come after '&' in 'const std::string& param_name'
                param_names = re.findall(r'&\s*(\w+)', params_str)
                expected_param_count = len(param_names)

                # Filter out "primary" if it's a duplicate and we have other named files
                filtered_files = {}
                for name, path in data_files.items():
                    if name != "primary" or len(data_files) == 1:
                        filtered_files[name] = path

                # Smart matching: match file names to parameter names
                def match_file_to_param(file_name, param_name):
                    """Calculate match score between file name and parameter name"""
                    file_lower = file_name.lower()
                    param_lower = param_name.lower()
                    score = 0

                    # Check for keyword matches
                    keywords = {
                        'travel': ['travel', 'tt', 'distance', 'dist'],
                        'job': ['job', 'jt', 'task', 'work'],
                        'time': ['time', 'duration', 'cost']
                    }

                    for key, synonyms in keywords.items():
                        if key in param_lower:
                            for syn in synonyms:
                                if syn in file_lower:
                                    score += 10

                    # Check for acronym matches (e.g., "TT" for "travel_time")
                    param_parts = param_lower.replace('_', ' ').split()
                    if len(param_parts) >= 2:
                        acronym = ''.join([p[0] for p in param_parts])
                        if acronym in file_lower or f"_{acronym}" in file_lower:
                            score += 15

                    # Partial string matches
                    if param_lower in file_lower or file_lower in param_lower:
                        score += 5

                    return score

                # Match files to parameters
                matched_paths = []
                if len(filtered_files) == expected_param_count:
                    # Try smart matching
                    used_files = set()
                    for param_name in param_names:
                        best_file = None
                        best_score = -1

                        for file_name, file_path in filtered_files.items():
                            if file_name not in used_files:
                                score = match_file_to_param(file_name, param_name)
                                if score > best_score:
                                    best_score = score
                                    best_file = (file_name, file_path)

                        if best_file and best_score > 0:
                            print(f"   Matched '{best_file[0]}' to parameter '{param_name}' (score: {best_score})")
                            matched_paths.append(best_file[1])
                            used_files.add(best_file[0])
                        elif best_file:
                            # No good match found, use first available but warn
                            print(f"   Warning: Weak match '{best_file[0]}' to parameter '{param_name}'")
                            matched_paths.append(best_file[1])
                            used_files.add(best_file[0])

                    file_paths = matched_paths if matched_paths else list(filtered_files.values())
                else:
                    # Fallback to dict order if counts don't match
                    file_paths = list(filtered_files.values())
                    if len(file_paths) > expected_param_count:
                        print(f"   Warning: {len(file_paths)} files provided but factory method expects {expected_param_count}, using first {expected_param_count}")
                        file_paths = file_paths[:expected_param_count]
                    elif len(file_paths) < expected_param_count:
                        print(f"   Warning: {len(file_paths)} files provided but factory method expects {expected_param_count}")

                # Build the call with actual file paths
                file_paths_list = [f'"{os.path.abspath(path)}"' for path in file_paths]
                config_init = f'auto config = {config_name}<float>::create_from_files({", ".join(file_paths_list)});'
            else:
                # Fallback: assume it wants all files in order (excluding "primary" if duplicate)
                filtered_files = {name: path for name, path in data_files.items()
                                 if name != "primary" or len(data_files) == 1}
                file_paths_list = [f'"{os.path.abspath(path)}"' for path in filtered_files.values()]
                config_init = f'auto config = {config_name}<float>::create_from_files({", ".join(file_paths_list)});'
        elif has_create_from_file:
            # Config needs a single data file
            if data_files:
                # Use the uploaded file (take the first one)
                first_file = list(data_files.values())[0]
                config_init = f'auto config = {config_name}<float>::create_from_file("{os.path.abspath(first_file)}");'
            else:
                # No file provided - try to find one in common locations
                # Look for TSP files or other data files
                possible_data_files = [
                    'uploads/*berlin52*.tsp',
                    'data/berlin52.tsp',
                    'brkga/data/berlin52.tsp',
                    'llm_solver/data/berlin52.tsp',
                    '*/berlin52.tsp'
                ]

                import glob
                data_file = None
                for pattern in possible_data_files:
                    matches = glob.glob(pattern)
                    if matches:
                        data_file = os.path.abspath(matches[0])
                        break

                if data_file:
                    config_init = f'auto config = {config_name}<float>::create_from_file("{data_file}");'
                else:
                    # Fallback: use a placeholder path
                    config_init = f'auto config = {config_name}<float>::create_from_file("data/input.tsp");'
        elif has_create_default_unique:
            # Factory returns unique_ptr - use directly
            config_init = f"auto config = {config_name}<float>::create_default();"
        elif has_create_standard_unique:
            # Factory returns unique_ptr - use directly
            config_init = f"auto config = {config_name}<float>::create_standard_instance();"
        elif has_create_default or has_create_standard:
            # Legacy: factory returns by value
            # Extract and inline the method body to avoid move issues
            static_method_name = "create_default" if has_create_default else "create_standard_instance"

            # Try to extract the method implementation and inline it
            pattern = rf'static\s+{config_name}<T>\s+{static_method_name}\s*\([^)]*\)\s*{{([^}}]+(?:{{[^}}]*}}[^}}]*)*)}}'
            match = re.search(pattern, content, re.DOTALL)

            if match:
                # Found the method - extract the body
                method_body = match.group(1).strip()
                # Replace T with float
                method_body = method_body.replace('<T>', '<float>')
                config_init = f"// Inlined from {static_method_name}()\n        {method_body}\n        auto config = std::make_unique<{config_name}<float>>(weights, values, capacity);"
            else:
                # Fallback: try to call the constructor with empty vectors
                print(f"   Warning: Could not inline {static_method_name}(), using default constructor")
                config_init = f"auto config = std::make_unique<{config_name}<float>>(std::vector<int>(), std::vector<int>(), 0);"
        else:
            # No factory method - use default constructor
            config_init = f"auto config = std::make_unique<{config_name}<float>>();"

        # Detect if this is a multi-objective problem
        is_multi_objective = ('NUM_OBJECTIVES' in content and 'NUM_OBJECTIVES = 2' in content) or \
                            ('objective_functions.resize' in content) or \
                            ('objective_functions[0]' in content and 'objective_functions[1]' in content)

        if is_multi_objective:
            # Multi-objective problem - use Pareto front
            main_content = f"""// Standard library includes first
#include <iostream>
#include <fstream>
#include <memory>
#include <iomanip>
#include <set>
#include <cstdlib>

// Project includes
#include "{os.path.abspath(config_path)}"
#include "{os.path.join(self.framework_path, 'core/solver.hpp')}"

int main() {{
    try {{
        std::cout << "Creating configuration..." << std::endl;

        // Create configuration
        {config_init}

        std::cout << "Creating BRKGA solver..." << std::endl;

        // Create solver (verbose=true, print_freq=10)
        Solver<float> solver(std::move(config), true, 10);

        // Check for quick test mode
        const char* quick_test_env = std::getenv("QUICK_TEST");
        if (quick_test_env) {{
            int quick_test_gens = std::atoi(quick_test_env);
            auto* config_ptr = solver.get_config();
            if (config_ptr) {{
                config_ptr->max_generations = quick_test_gens;
                std::cout << "Quick test mode: limiting to " << quick_test_gens << " generations" << std::endl;
            }}
        }}

        std::cout << "Running optimization..." << std::endl;

        // Run the optimization
        solver.run();

        std::cout << "\\n=== Multi-Objective Optimization Complete! ===" << std::endl;

        // Export Pareto front
        solver.export_pareto_front("llm_solver/results/pareto_front.txt");
        std::cout << "\\nPareto front exported to: llm_solver/results/pareto_front.txt" << std::endl;

        // Get Pareto front
        auto pareto = solver.get_pareto_front();
        std::cout << "\\nPareto front size: " << pareto.size() << std::endl;

        if (!pareto.empty()) {{
            // Count unique solutions
            std::set<std::pair<float, float>> unique_solutions;
            for (const auto& ind : pareto) {{
                unique_solutions.insert({{ind.objectives[0], ind.objectives[1]}});
            }}
            std::cout << "Unique solutions: " << unique_solutions.size() << std::endl;

            // Display first 10 solutions
            std::cout << "\\nFirst 10 Pareto solutions:" << std::endl;
            std::cout << "   Objective 1    Objective 2" << std::endl;
            std::cout << "--------------------------------------" << std::endl;

            int count = std::min(10, static_cast<int>(pareto.size()));
            for (int i = 0; i < count; i++) {{
                std::cout << std::setw(12) << std::fixed << std::setprecision(6)
                          << pareto[i].objectives[0] << "  "
                          << std::setw(12) << std::setprecision(6)
                          << pareto[i].objectives[1] << std::endl;
            }}

            // Export to solution file
            std::ofstream solution_file("llm_solver/results/solution.txt");
            if (solution_file.is_open()) {{
                solution_file << "Multi-Objective Optimization Solution" << std::endl;
                solution_file << "====================================" << std::endl << std::endl;
                solution_file << "Pareto Front Size: " << pareto.size() << std::endl;
                solution_file << "Unique Solutions: " << unique_solutions.size() << std::endl << std::endl;

                solution_file << "Pareto Front (all solutions):" << std::endl;
                solution_file << "Objective 1\\tObjective 2" << std::endl;
                solution_file << "--------------------------------------" << std::endl;

                for (const auto& ind : pareto) {{
                    solution_file << std::fixed << std::setprecision(6)
                                  << ind.objectives[0] << "\\t"
                                  << ind.objectives[1] << std::endl;
                }}

                solution_file.close();
                std::cout << "\\n✓ Solution file saved to llm_solver/results/solution.txt" << std::endl;
            }}
        }}

        return 0;

    }} catch (const std::exception& e) {{
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }}
}}
"""
        else:
            # Single-objective problem - use best individual
            main_content = f"""// Standard library includes first
#include <iostream>
#include <fstream>
#include <memory>
#include <iomanip>
#include <cstdlib>

// Project includes
#include "{os.path.abspath(config_path)}"
#include "{os.path.join(self.framework_path, 'core/solver.hpp')}"

int main() {{
    try {{
        std::cout << "Creating configuration..." << std::endl;

        // Create configuration
        {config_init}

        std::cout << "Creating BRKGA solver..." << std::endl;

        // Create solver (verbose=true, print_freq=10)
        Solver<float> solver(std::move(config), true, 10);

        // Check for quick test mode
        const char* quick_test_env = std::getenv("QUICK_TEST");
        if (quick_test_env) {{
            int quick_test_gens = std::atoi(quick_test_env);
            auto* config_ptr = solver.get_config();
            if (config_ptr) {{
                config_ptr->max_generations = quick_test_gens;
                std::cout << "Quick test mode: limiting to " << quick_test_gens << " generations" << std::endl;
            }}
        }}

        std::cout << "Running optimization..." << std::endl;

        // Run the optimization
        solver.run();

        std::cout << "\\nOptimization complete!" << std::endl;

        // Get best solution
        const auto& best = solver.get_best_individual();
        std::cout << "Best fitness: " << best.fitness << std::endl;

        // Print solution details if config has print_solution method
        auto* config_ptr = solver.get_config();
        if (config_ptr) {{
            std::cout << "\\n";
            config_ptr->print_solution(best);
        }}

        // Export solution to file
        std::cout << "\\nExporting solution to: llm_solver/results/solution.txt" << std::endl;

        // Create results directory if it doesn't exist
        std::system("mkdir -p llm_solver/results");

        // Export solution to file using decoded representation
        std::ofstream solution_file("llm_solver/results/solution.txt");
        if (solution_file.is_open()) {{
            solution_file << "BRKGA Optimization Solution" << std::endl;
            solution_file << "===========================" << std::endl << std::endl;
            solution_file << "Best Fitness: " << best.fitness << std::endl << std::endl;

            // Redirect print_solution output to file
            std::streambuf* cout_backup = std::cout.rdbuf();
            std::cout.rdbuf(solution_file.rdbuf());

            // Use config's print_solution for detailed output
            config_ptr->print_solution(best);

            // Restore stdout
            std::cout.rdbuf(cout_backup);

            solution_file.close();
            std::cout << "✓ Solution saved to llm_solver/results/solution.txt" << std::endl;
        }}

        return 0;

    }} catch (const std::exception& e) {{
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }}
}}
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
            f.write(main_content)
            return f.name
    
    def run_quick_test(self, executable: str, 
                      test_generations: int = 10,
                      timeout: int = 30) -> ExecutionResult:
        """
        Run a quick test with few generations to validate the solver.
        
        Args:
            executable: Path to executable
            test_generations: Number of generations for test
            timeout: Timeout in seconds
            
        Returns:
            ExecutionResult
        """
        
        print(f"\n🧪 Running quick test ({test_generations} generations)...")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                [executable],
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, 'QUICK_TEST': str(test_generations)}
            )
            
            execution_time = time.time() - start_time
            
            # Parse output
            best_fitness = self._extract_fitness(result.stdout)
            generations = self._extract_generations(result.stdout)
            solution_valid = self._check_solution_validity(result.stdout)
            
            success = result.returncode == 0
            
            if success:
                print(f"✅ Quick test passed ({execution_time:.2f}s)")
                if best_fitness is not None:
                    print(f"   Best fitness: {best_fitness:.6f}")
            else:
                print(f"❌ Quick test failed ({execution_time:.2f}s)")
            
            return ExecutionResult(
                success=success,
                output=result.stdout + result.stderr,
                best_fitness=best_fitness,
                generations=generations,
                execution_time=execution_time,
                solution_valid=solution_valid,
                errors=self._extract_errors(result.stderr)
            )
            
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                output="Execution timeout",
                execution_time=timeout,
                errors=["Execution took too long"]
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                output=str(e),
                execution_time=time.time() - start_time,
                errors=[f"Execution error: {str(e)}"]
            )
    
    def run_full_optimization(self, executable: str,
                             data_file: Optional[str] = None,
                             max_time_minutes: Optional[int] = None) -> ExecutionResult:
        """
        Run full optimization.
        
        Args:
            executable: Path to executable
            data_file: Optional data file path
            max_time_minutes: Maximum runtime in minutes
            
        Returns:
            ExecutionResult
        """
        
        print(f"\n🚀 Running full optimization...")
        if max_time_minutes:
            print(f"   Time limit: {max_time_minutes} minutes")
        
        start_time = time.time()
        
        cmd = [executable]
        if data_file:
            cmd.extend(['--data', data_file])
        
        timeout = max_time_minutes * 60 if max_time_minutes else None
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            execution_time = time.time() - start_time
            
            # Parse results
            best_fitness = self._extract_fitness(result.stdout)
            generations = self._extract_generations(result.stdout)
            solution_valid = self._check_solution_validity(result.stdout)
            pareto_size = self._extract_pareto_size(result.stdout)
            
            success = result.returncode == 0
            
            if success:
                print(f"✅ Optimization completed ({execution_time:.2f}s)")
                if best_fitness is not None:
                    print(f"   Best fitness: {best_fitness:.6f}")
                if pareto_size is not None:
                    print(f"   Pareto front size: {pareto_size}")
                print(f"   Generations: {generations}")
            else:
                print(f"❌ Optimization failed")
            
            return ExecutionResult(
                success=success,
                output=result.stdout + result.stderr,
                best_fitness=best_fitness,
                generations=generations,
                execution_time=execution_time,
                solution_valid=solution_valid,
                pareto_front_size=pareto_size,
                errors=self._extract_errors(result.stderr)
            )
            
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                output="Execution timeout - optimization taking too long",
                execution_time=timeout if timeout else 0,
                errors=["Optimization exceeded time limit"]
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                output=str(e),
                execution_time=time.time() - start_time,
                errors=[f"Execution error: {str(e)}"]
            )
    
    def _parse_errors(self, stderr: str) -> list[str]:
        """Extract error messages."""
        errors = []
        for line in stderr.split('\n'):
            if 'error:' in line.lower():
                error = line.split('error:')[-1].strip()
                if error:
                    errors.append(error)
        return errors
    
    def _parse_warnings(self, stderr: str) -> list[str]:
        """Extract warning messages."""
        warnings = []
        for line in stderr.split('\n'):
            if 'warning:' in line.lower():
                warning = line.split('warning:')[-1].strip()
                if warning:
                    warnings.append(warning)
        return warnings
    
    def _extract_fitness(self, output: str) -> Optional[float]:
        """Extract best fitness from output."""
        # Try to find final fitness value after "Optimization complete!" message
        # This avoids matching intermediate or initial fitness values
        patterns = [
            r'Optimization complete.*?Best fitness:\s*([-\d.]+)',
            r'Final fitness:\s*([-\d.]+)',
            r'Best fitness:\s*([-\d.]+)',
            r'best:\s*([-\d.]+)',
            r'fitness:\s*([-\d.]+)'
        ]

        for pattern in patterns:
            # Use DOTALL flag so .* can match newlines
            matches = re.findall(pattern, output, re.IGNORECASE | re.DOTALL)
            if matches:
                try:
                    # Take the last match (most recent/final value)
                    return float(matches[-1])
                except:
                    pass
        return None
    
    def _extract_generations(self, output: str) -> int:
        """Extract number of generations from output."""
        pattern = r'Generation[s]?:\s*(\d+)'
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return 0
    
    def _extract_pareto_size(self, output: str) -> Optional[int]:
        """Extract Pareto front size for multi-objective."""
        pattern = r'Pareto front size:\s*(\d+)'
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None
    
    def _check_solution_validity(self, output: str) -> bool:
        """Check if solution is reported as valid."""
        return 'valid solution' in output.lower() or 'solution valid' in output.lower()
    
    def _extract_errors(self, stderr: str) -> list[str]:
        """Extract runtime errors."""
        errors = []
        for line in stderr.split('\n'):
            if any(kw in line.lower() for kw in ['error', 'exception', 'failed']):
                if line.strip():
                    errors.append(line.strip())
        return errors
