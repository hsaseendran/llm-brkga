"""
Validator for generated BRKGA configuration files.
Performs syntax, semantic, and compilation validation.
"""

import os
import re
import subprocess
import tempfile
from typing import List, Tuple
from .problem_structures import *


class Validator:
    """Validates generated BRKGA configuration files."""
    
    def __init__(self, framework_path: str = "brkga"):
        """
        Initialize validator.

        Args:
            framework_path: Path to BRKGA framework files
        """
        self.framework_path = os.path.abspath(framework_path)
    
    def validate_syntax(self, config_path: str) -> ValidationResult:
        """
        Perform syntactic validation of the configuration file.
        
        Args:
            config_path: Path to the config file
            
        Returns:
            ValidationResult with syntax check results
        """
        
        print("🔍 Validating syntax...")
        
        errors = []
        warnings = []
        suggestions = []
        
        with open(config_path, 'r') as f:
            code = f.read()
        
        # Check for header guards
        if '#ifndef' not in code or '#define' not in code:
            warnings.append("Missing header guards")
        
        # Check for required includes (accept various valid paths to config.hpp)
        has_config = ('core/config.hpp' in code or
                     '../core/config.hpp' in code or
                     '../../brkga/core/config.hpp' in code or
                     'brkga_config.hpp' in code)
        if not has_config:
            errors.append("Missing required include: config.hpp (should be ../../brkga/core/config.hpp)")
        
        # Check for template declaration
        if 'template<typename T>' not in code and 'template <typename T>' not in code:
            errors.append("Missing template declaration")
        
        # Check for base class
        if 'BRKGAConfig<T>' not in code and 'BRKGAConfig<float>' not in code:
            errors.append("Must inherit from BRKGAConfig<T>")
        
        # Check for balanced braces
        open_braces = code.count('{')
        close_braces = code.count('}')
        if open_braces != close_braces:
            errors.append(f"Unbalanced braces: {open_braces} {{ vs {close_braces} }}")
        
        # Check for get_chromosome() usage
        if 'chromosome[' in code or 'chromosome.at' in code:
            if 'get_chromosome()' not in code:
                warnings.append("Direct chromosome access detected - should use get_chromosome()")
        
        # Check for lambda syntax
        if 'fitness_function =' in code:
            if '[this]' not in code:
                warnings.append("Lambda should capture [this]")
        
        # Check for const references
        if 'Individual<T>&' in code and 'const Individual<T>&' not in code:
            warnings.append("Should use const references for Individual parameters")
        
        # Check for vector reserve
        if 'vector<' in code and 'push_back' in code:
            if 'reserve(' not in code:
                suggestions.append("Consider reserving vector capacity before filling")
        
        success = len(errors) == 0
        
        result = ValidationResult(
            success=success,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
        
        if success:
            print("✅ Syntax validation passed")
        else:
            print("❌ Syntax validation failed")
        
        return result
    
    def validate_semantics(self, config_path: str, 
                          problem: ProblemStructure) -> ValidationResult:
        """
        Perform semantic validation of the configuration logic.
        
        Args:
            config_path: Path to the config file
            problem: Original problem structure
            
        Returns:
            ValidationResult with semantic check results
        """
        
        print("🔍 Validating semantics...")
        
        errors = []
        warnings = []
        suggestions = []
        
        with open(config_path, 'r') as f:
            code = f.read()
        
        # Check chromosome length matches problem
        chrom_length_pattern = r'BRKGAConfig<\w+>\((\d+)\)'
        match = re.search(chrom_length_pattern, code)
        if match:
            declared_length = int(match.group(1))
            if declared_length != problem.chromosome_length:
                errors.append(
                    f"Chromosome length mismatch: declared {declared_length}, "
                    f"expected {problem.chromosome_length}"
                )
        else:
            warnings.append("Could not verify chromosome length")
        
        # Check for decoder implementation
        if 'decoder =' not in code and 'this->decoder =' not in code:
            errors.append("Decoder function not assigned")
        
        # Check for fitness or objective functions based on problem type
        if problem.is_multi_objective:
            if 'objective_functions' not in code:
                errors.append("Multi-objective problem requires objective_functions")
            if 'fitness_function' in code:
                warnings.append("Multi-objective should not use fitness_function")
        else:
            if 'fitness_function =' not in code and 'this->fitness_function =' not in code:
                errors.append("Single-objective problem requires fitness_function")
        
        # Check for comparator
        if 'comparator =' not in code and 'this->comparator =' not in code:
            warnings.append("Comparator not assigned (will use default)")
        
        # Check decoder return type
        if 'std::vector<std::vector<' in code or 'vector<vector<' in code:
            # Good - proper return type
            pass
        else:
            warnings.append("Decoder should return vector<vector<T>>")
        
        # Check for data loading if required
        if problem.requires_data_file:
            if 'load_from_file' not in code:
                warnings.append("Problem requires data file but no load method found")
        
        success = len(errors) == 0
        
        result = ValidationResult(
            success=success,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
        
        if success:
            print("✅ Semantic validation passed")
        else:
            print("❌ Semantic validation failed")
        
        return result
    
    def compile_test(self, config_path: str, 
                    quick_test: bool = True) -> CompilationResult:
        """
        Attempt to compile the configuration file.
        
        Args:
            config_path: Path to config file
            quick_test: If True, only check compilation, don't link
            
        Returns:
            CompilationResult with compilation status
        """
        
        print("🔨 Compiling configuration...")
        
        import time
        start_time = time.time()
        
        # Create a temporary test file with absolute include path
        abs_config_path = os.path.abspath(config_path)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as tmp:
            tmp.write(f"""
#include "{abs_config_path}"
int main() {{ return 0; }}
""")
            test_file = tmp.name

        try:
            # Get the parent directory of the config file to add to include path
            # This allows relative includes from the config to work
            config_dir = os.path.dirname(abs_config_path)
            project_root = os.path.dirname(os.path.dirname(config_dir))  # Go up to project root

            # Compile command
            compile_cmd = [
                'nvcc',
                '-std=c++17',
                '-I', project_root,  # Include project root so ../../brkga/... works
                '-I', self.framework_path,
                '-c' if quick_test else '',
                test_file,
                '-o', test_file.replace('.cu', '.o')
            ]
            
            # Remove empty strings from command
            compile_cmd = [arg for arg in compile_cmd if arg]
            
            result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            compilation_time = time.time() - start_time
            
            # Parse output
            errors = self._parse_compiler_errors(result.stderr)
            warnings = self._parse_compiler_warnings(result.stderr)
            
            success = result.returncode == 0
            
            if success:
                print(f"✅ Compilation successful ({compilation_time:.2f}s)")
            else:
                print(f"❌ Compilation failed ({compilation_time:.2f}s)")
                if errors:
                    print("   Compilation errors:")
                    for err in errors[:10]:  # Show first 10 errors
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
                errors=["Compilation took too long (>60s)"],
                compilation_time=60.0
            )
        except Exception as e:
            return CompilationResult(
                success=False,
                output=str(e),
                errors=[f"Compilation error: {str(e)}"],
                compilation_time=time.time() - start_time
            )
        finally:
            # Cleanup
            try:
                os.unlink(test_file)
                os.unlink(test_file.replace('.cu', '.o'))
            except:
                pass
    
    def _parse_compiler_errors(self, stderr: str) -> List[str]:
        """Extract error messages from compiler output."""
        errors = []
        for line in stderr.split('\n'):
            if 'error:' in line.lower():
                # Clean up the error message
                error = line.split('error:')[-1].strip()
                if error and error not in errors:
                    errors.append(error)
        return errors
    
    def _parse_compiler_warnings(self, stderr: str) -> List[str]:
        """Extract warning messages from compiler output."""
        warnings = []
        for line in stderr.split('\n'):
            if 'warning:' in line.lower():
                # Clean up the warning message
                warning = line.split('warning:')[-1].strip()
                if warning and warning not in warnings:
                    warnings.append(warning)
        return warnings
    
    def full_validation(self, config_path: str, 
                       problem: ProblemStructure) -> Tuple[bool, List[ValidationResult]]:
        """
        Perform complete validation pipeline.
        
        Args:
            config_path: Path to config file
            problem: Original problem structure
            
        Returns:
            Tuple of (success, list of validation results)
        """
        
        print("\n" + "="*60)
        print("VALIDATION PIPELINE")
        print("="*60)
        
        results = []
        
        # Stage 1: Syntax
        syntax_result = self.validate_syntax(config_path)
        results.append(syntax_result)
        
        if not syntax_result.success:
            print("\n⚠️  Syntax errors detected - stopping validation")
            return False, results
        
        # Stage 2: Semantics
        semantic_result = self.validate_semantics(config_path, problem)
        results.append(semantic_result)
        
        if not semantic_result.success:
            print("\n⚠️  Semantic errors detected - stopping validation")
            return False, results
        
        # Stage 3: Compilation
        compile_result = self.compile_test(config_path)
        results.append(compile_result)
        
        if not compile_result.success:
            print("\n⚠️  Compilation failed")
            return False, results
        
        print("\n✅ All validation stages passed!")
        return True, results
