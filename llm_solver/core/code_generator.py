"""
Code generator that transforms problem structures into valid C++ BRKGA configurations.
"""

import os
from typing import Optional, Dict, Any
from anthropic import Anthropic
from .problem_structures import (
    ProblemStructure, ProblemType, DecisionVariable, Objective,
    Constraint, ConstraintType, ConstraintHandling, DecoderStrategy,
    LocalSearchType
)
from .data_parser import DataFileParser


class CodeGenerator:
    """Generates C++ BRKGA configuration files from problem structures."""
    
    def __init__(self, context_package_path: str, api_key: Optional[str] = None):
        """
        Initialize the code generator.

        Args:
            context_package_path: Path to BRKGA context package
            api_key: Anthropic API key
        """
        self.context_path = context_package_path
        self.client = Anthropic(api_key=api_key) if api_key else Anthropic()
        self.context = self._load_context()
        self.framework_path = "brkga"  # Project files location (relative to root)
        self.data_parser = DataFileParser()
    
    def _load_context(self) -> str:
        """Load the complete BRKGA context."""
        # Try to load from various locations
        possible_paths = [
            os.path.join(self.context_path, "context_for_llm_full.txt"),
            "context/context_for_llm_full.txt",
            "llm_solver/context/context_for_llm_full.txt",
            os.path.join(os.path.dirname(__file__), "..", "context", "context_for_llm_full.txt")
        ]

        for context_file in possible_paths:
            if os.path.exists(context_file):
                with open(context_file, 'r') as f:
                    return f.read()

        raise FileNotFoundError(
            f"Cannot find BRKGA context file. Tried:\n" +
            "\n".join(f"  - {p}" for p in possible_paths)
        )
    
    def generate_config(self, problem: ProblemStructure,
                       output_path: str,
                       hyperparameters: Optional[Dict[str, Any]] = None,
                       include_test_data_generator: bool = False) -> str:
        """
        Generate a complete BRKGA configuration file.

        Args:
            problem: The analyzed problem structure
            output_path: Where to save the generated file
            hyperparameters: Optional dict of BRKGA hyperparameters (population_size, elite_percentage, etc.)
            include_test_data_generator: Whether to include test data generation

        Returns:
            Path to the generated file
        """

        print(f"🔧 Generating config for: {problem.problem_name}")
        print(f"   Type: {problem.problem_type.value}")
        print(f"   Decoder: {problem.decoder_strategy.value}")
        print(f"   Objectives: {len(problem.objectives)}")
        if problem.local_search_recommended:
            print(f"   Local search: {problem.local_search_type.value}")

        # Build the generation prompt
        generation_prompt = self._build_generation_prompt(problem, hyperparameters, include_test_data_generator)
        
        # Call LLM to generate code
        print("🤖 Calling LLM for code generation...")
        response = self.client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=8000,
            temperature=0.1,  # Low temperature for consistent code generation
            system=self.context,  # Provide BRKGA context as system prompt
            messages=[{
                "role": "user",
                "content": generation_prompt
            }]
        )
        
        generated_code = response.content[0].text
        
        # Extract C++ code from response
        code = self._extract_code(generated_code)
        
        # Save to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(code)
        
        print(f"✅ Config generated: {output_path}")
        
        return output_path
    
    def _build_generation_prompt(self, problem: ProblemStructure,
                                 hyperparameters: Optional[Dict[str, Any]],
                                 include_test_data: bool) -> str:
        """Build the prompt for code generation."""

        prompt = f"""Generate a complete BRKGA configuration file for the following optimization problem.

PROBLEM SPECIFICATION:
=====================

Name: {problem.problem_name}
Type: {problem.problem_type.value}
Description: {problem.domain}

DECISION VARIABLES:
"""
        
        for dv in problem.decision_variables:
            prompt += f"- {dv.name}: {dv.count} variables ({dv.semantics})\n"
        
        prompt += f"\nChromosome Length: {problem.chromosome_length}\n"
        
        prompt += "\nOBJECTIVES:\n"
        for obj in problem.objectives:
            prompt += f"- {obj.type.upper()} {obj.name}: {obj.description}\n"
            if obj.formula_hint:
                prompt += f"  Formula hint: {obj.formula_hint}\n"
        
        if problem.constraints:
            prompt += "\nCONSTRAINTS:\n"
            for c in problem.constraints:
                prompt += f"- {c.type.value.upper()}: {c.description}\n"
                prompt += f"  Handling: {c.handling.value}\n"
                if c.code_hint:
                    prompt += f"  Implementation: {c.code_hint}\n"
        
        prompt += f"\nDECODER STRATEGY: {problem.decoder_strategy.value}\n"
        prompt += f"Rationale: {problem.decoder_rationale}\n"
        
        if problem.expected_scale:
            prompt += "\nEXPECTED SCALE:\n"
            for k, v in problem.expected_scale.items():
                prompt += f"- {k}: {v}\n"
        
        prompt += f"\nMulti-objective: {problem.is_multi_objective}\n"

        # Add local search information
        prompt += f"\nLOCAL SEARCH:\n"
        prompt += f"- Recommended: {problem.local_search_recommended}\n"
        if problem.local_search_recommended:
            prompt += f"- Type: {problem.local_search_type.value}\n"
            prompt += f"- Rationale: {problem.local_search_rationale}\n"

        if problem.notes:
            prompt += "\nADDITIONAL NOTES:\n"
            for note in problem.notes:
                prompt += f"- {note}\n"

        # Add hyperparameters if provided
        if hyperparameters:
            prompt += "\nBRKGA HYPERPARAMETERS (USER-SPECIFIED):\n"
            prompt += "=" * 50 + "\n"
            if "population_size" in hyperparameters:
                prompt += f"Population Size: {hyperparameters['population_size']}\n"
            if "elite_percentage" in hyperparameters:
                elite_pct = hyperparameters['elite_percentage']
                prompt += f"Elite Percentage: {elite_pct} ({elite_pct*100:.0f}%)\n"
            if "mutant_percentage" in hyperparameters:
                mutant_pct = hyperparameters['mutant_percentage']
                prompt += f"Mutant Percentage: {mutant_pct} ({mutant_pct*100:.0f}%)\n"
            if "elite_prob" in hyperparameters:
                prompt += f"Elite Bias (elite_prob): {hyperparameters['elite_prob']}\n"
            if "max_generations" in hyperparameters:
                prompt += f"Maximum Generations: {hyperparameters['max_generations']}\n"
            prompt += "\n" + "=" * 50 + "\n"
            prompt += "CRITICAL: You MUST use these EXACT values in the config constructor:\n"
            prompt += "=" * 50 + "\n"
            if "population_size" in hyperparameters:
                prompt += f"  this->population_size = {hyperparameters['population_size']};\n"
            if "elite_percentage" in hyperparameters:
                elite_pct = hyperparameters['elite_percentage']
                prompt += f"  this->elite_size = static_cast<int>(this->population_size * {elite_pct});\n"
            if "mutant_percentage" in hyperparameters:
                mutant_pct = hyperparameters['mutant_percentage']
                prompt += f"  this->mutant_size = static_cast<int>(this->population_size * {mutant_pct});\n"
            if "elite_prob" in hyperparameters:
                prompt += f"  this->elite_prob = {hyperparameters['elite_prob']};\n"
            if "max_generations" in hyperparameters:
                prompt += f"  this->max_generations = {hyperparameters['max_generations']};\n"
            prompt += "\nCopy these lines EXACTLY into your config constructor!\n"
            prompt += "=" * 50 + "\n\n"

        # Add data file information if available
        if problem.data_files or (problem.data_file_path and problem.data_metadata):
            # Handle multiple data files
            if problem.data_files and len(problem.data_files) > 0:
                prompt += f"\nDATA FILES PROVIDED ({len(problem.data_files)}):\n"
                prompt += "="*60 + "\n"

                for file_name, file_path in problem.data_files.items():
                    prompt += f"\n--- File '{file_name}': {file_path} ---\n"

                    # Get metadata for this specific file
                    if problem.data_metadata and file_name in problem.data_metadata:
                        file_metadata = problem.data_metadata[file_name]

                        prompt += f"- Format: {file_metadata.get('format', 'unknown')}\n"
                        prompt += f"- Problem size: {file_metadata.get('problem_size', 'unknown')}\n"

                        # Add format-specific details
                        dim_info = file_metadata.get('dimension_info', {})
                        if dim_info:
                            prompt += "- Dimensions:\n"
                            for key, value in dim_info.items():
                                prompt += f"  - {key}: {value}\n"

                        # Add edge weight type for TSP/VRP
                        if file_metadata.get('edge_weight_type'):
                            prompt += f"- Edge weight type: {file_metadata['edge_weight_type']}\n"

                        # Add data preview
                        data_preview = file_metadata.get('data_preview', [])
                        if data_preview:
                            prompt += "\n- Data preview (first few lines):\n"
                            prompt += "```\n"
                            for line in data_preview[:5]:  # Show first 5 lines
                                prompt += f"{line}\n"
                            prompt += "```\n"

                prompt += "\n" + "="*60 + "\n"
                prompt += f"\nIMPORTANT: Your generated code must read and parse ALL {len(problem.data_files)} data file(s):\n"
                for file_name, file_path in problem.data_files.items():
                    prompt += f"  - {file_name}: {file_path}\n"
                prompt += "Make sure to include proper file reading logic for each file in the constructor.\n"

            # Backward compatibility: single data file
            elif problem.data_file_path and problem.data_metadata:
                prompt += "\nDATA FILE INFORMATION:\n"
                prompt += f"- File path: {problem.data_file_path}\n"
                prompt += f"- Format: {problem.data_metadata.get('format', 'unknown')}\n"
                prompt += f"- Problem size: {problem.data_metadata.get('problem_size', 'unknown')}\n"

                # Add format-specific details
                dim_info = problem.data_metadata.get('dimension_info', {})
                if dim_info:
                    prompt += "- Dimensions:\n"
                    for key, value in dim_info.items():
                        prompt += f"  - {key}: {value}\n"

                # Add edge weight type for TSP/VRP
                if problem.data_metadata.get('edge_weight_type'):
                    prompt += f"- Edge weight type: {problem.data_metadata['edge_weight_type']}\n"

                # Add data preview
                data_preview = problem.data_metadata.get('data_preview', [])
                if data_preview:
                    prompt += "\n- Data preview (first few lines):\n"
                    prompt += "```\n"
                    for line in data_preview[:5]:  # Show first 5 lines
                        prompt += f"{line}\n"
                    prompt += "```\n"

                prompt += f"\nIMPORTANT: Your generated code must read and parse the data file at: {problem.data_file_path}\n"
                prompt += "Make sure to include proper file reading logic in the constructor.\n"

        prompt += """

GENERATION REQUIREMENTS:
========================

1. Create a complete, compilable C++ header file
2. Follow the BRKGA framework patterns exactly as shown in the context
3. Use the appropriate decoder strategy for this problem type
4. Include all necessary data structures as private members
5. Implement proper constructor initialization
6. Use lambdas for function pointer assignment
7. Always use get_chromosome() method, never direct access
8. Use const references to avoid unnecessary copying
9. Reserve vector space before filling
10. Include helpful comments explaining the approach

CRITICAL - Framework API Requirements:
- Include statement: #include "../../brkga/core/config.hpp"
  (This is the correct path from llm_solver/generated/ to brkga/core/)
- Decoder return type: std::vector<std::vector<T>> (NOT DecodedSolution)
- Fitness function signature: T function_name(const Individual<T>&)
- Decoder signature: std::vector<std::vector<T>> function_name(const Individual<T>&)
- Comparator signature: bool function_name(T, T)
- Use get_chromosome() method to access genes
- Template parameter should be typename T (not hardcoded float)

Example signatures:
    this->fitness_function = [this](const Individual<T>& individual) {
        return calculate_fitness(individual);
    };

    this->decoder = [this](const Individual<T>& individual) {
        return decode_to_solution(individual);
    };

    // Comparator for minimization problems (TSP, VRP, scheduling, etc.)
    this->comparator = [](T a, T b) { return a < b; };  // Smaller is better

    // Comparator for maximization problems (knapsack, profit, coverage, etc.)
    this->comparator = [](T a, T b) { return a > b; };  // Larger is better

IMPORTANT - Fitness Values and Comparators:
- For MINIMIZATION problems (TSP, VRP, makespan, cost, etc.):
  * Return the ACTUAL VALUE (e.g., tour distance, not negative)
  * Set comparator: this->comparator = [](T a, T b) { return a < b; };
  * Example: fitness = 7542 for a TSP tour of distance 7542

- For MAXIMIZATION problems (knapsack, profit, value, etc.):
  * Return the ACTUAL VALUE (e.g., total value, not negative)
  * Set comparator: this->comparator = [](T a, T b) { return a > b; };
  * Example: fitness = 150 for knapsack with value 150

DO NOT use negative fitness values! The comparator handles min vs max.

"""

        # Add local search requirements if recommended
        if problem.local_search_recommended:
            prompt += f"""
LOCAL SEARCH REQUIREMENTS:
- Include local search to improve solution quality
- Local search type: {problem.local_search_type.value}
- Include #include "../../brkga/core/local_search.hpp" header
- Create a nested LocalSearch<T> subclass specific to this problem
- Implement the improve() method with appropriate moves
- Call add_local_searches() method in constructor after GPU setup
- Configure LocalSearchConfig with appropriate strategy (ELITE_ONLY recommended)

Example local search setup (add at end of constructor):
```cpp
void add_local_searches() {{
    auto local_search = std::make_unique<YourLocalSearch>(problem_data);
    this->add_local_search(std::move(local_search));

    LocalSearchConfig<T> ls_config;
    ls_config.strategy = LocalSearchStrategy::ELITE_ONLY;
    ls_config.frequency = 5;
    ls_config.probability = 0.8;
    ls_config.apply_to_best = true;
    this->set_local_search_config(ls_config);
}}
```

"""

        if problem.is_multi_objective:
            prompt += """
MULTI-OBJECTIVE REQUIREMENTS:
- Use the multi-objective constructor: BRKGAConfig<T>(component_lengths, n_objectives)
- Implement multiple objective functions
- Do NOT implement a fitness function (multi-objective doesn't use it)
- Ensure objective_functions vector has correct size

"""
        
        prompt += """
STRUCTURE:
Your response should be a complete .hpp file with:

1. Header guards
2. #include "../../brkga/core/config.hpp" (EXACTLY this path - and other standard headers as needed)
3. Template class declaration: template<typename T> class YourConfig : public BRKGAConfig<T>
4. Private data members for problem-specific data
5. Public constructor that:
   - Takes ONLY problem-specific parameters (e.g., weights, values, capacity)
   - DO NOT add BRKGA parameters (population_size, elite_percentage, etc.)
   - Calls base class constructor: BRKGAConfig<T>(chromosome_length) with ONLY chromosome length
   - Initializes data members
   - Assigns fitness_function lambda returning T
   - Assigns decoder lambda returning std::vector<std::vector<T>>
   - Assigns comparator lambda returning bool
   - Sets threads_per_block and calls update_cuda_grid_size()
6. Private helper methods for fitness calculation and decoding
7. DECODER OUTPUT FORMAT (CRITICAL):
   - The decoder MUST return the COMPLETE, READY-TO-PRINT solution
   - NO post-processing should be needed by the execution framework
   - The decoder is problem-specific and must handle ALL formatting
   - Examples:
     * TSP with depot: return [0, city1, city2, ..., cityN, 0] (depot at start AND end)
     * TSP without depot: return [city1, city2, ..., cityN] (just the permutation)
     * TSPJ: return [[city_sequence], [job_sequence]] (two components)
     * Assignment: return [assignment_values] (one per task/item)
   - The execution framework will write decoder output AS-IS with no modifications
8. REQUIRED: print_solution(const Individual<T>&) method that shows INTERPRETED solution
   - NOT just chromosome values
   - For knapsack: show selected item indices, total weight, total value
   - For TSP: show tour order, total distance
   - For scheduling: show task assignments, completion times
9. Optional: validate_solution, export_solution methods
10. Optional: static factory methods that return std::unique_ptr<YourConfig<T>>

CRITICAL REQUIREMENTS:
- The first include MUST be exactly: #include "../../brkga/core/config.hpp"

- Base class constructor call MUST use ONE of these forms:
  * Single component: BRKGAConfig<T>(chromosome_length)
  * Multi component: BRKGAConfig<T>({{length1, length2, ...}})
  * NEVER use: BRKGAConfig<T>(length, num_objectives, num_constraints) ❌ WRONG!

- Function signature requirements:
  * fitness_function: [this](const Individual<T>& individual) -> T { ... }
  * decoder: [this](const Individual<T>& individual) -> std::vector<std::vector<T>> { ... }
  * comparator: [](T a, T b) -> bool { return a < b; }
  * NEVER use: const std::vector<T>& chromosome ❌ WRONG!
  * ALWAYS access chromosome via: individual.get_chromosome()

- If config requires a data file, include factory method:
  static std::unique_ptr<YourConfig<T>> create_from_file(const std::string& path) {
      return std::make_unique<YourConfig<T>>(path);
  }

- If providing a static factory method (like create_default), it MUST:
  1. Return std::unique_ptr<YourConfig<T>>
  2. Take NO parameters (or all parameters have default values)
  3. Contain hardcoded test data
  Example:
    static std::unique_ptr<YourConfig<T>> create_default() {
        std::vector<int> weights = {10, 8, 7, ...};
        std::vector<int> values = {50, 40, 35, ...};
        int capacity = 100;
        return std::make_unique<YourConfig<T>>(weights, values, capacity);
    }

"""
        
        if include_test_data:
            prompt += """
Also include a static method to generate test data for this problem.

"""
        
        prompt += """
Generate the complete code now. Provide ONLY the C++ code, no explanations before or after.
Make sure to follow ALL the critical rules from the BRKGA context, especially:
- Use get_chromosome() method
- Use lambdas for function pointers with [this] capture
- Reserve vectors before filling
- Const references for individuals
- Correct return types
"""
        
        return prompt
    
    def _extract_code(self, response: str) -> str:
        """Extract C++ code from LLM response."""
        import re
        
        # Look for code blocks
        code_block_pattern = r'```(?:cpp|c\+\+)?\n(.*?)```'
        matches = re.findall(code_block_pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # If no code blocks, look for #ifndef or #include at start
        lines = response.split('\n')
        code_started = False
        code_lines = []
        
        for line in lines:
            if not code_started:
                if line.strip().startswith(('#ifndef', '#define', '#include', 'template', 'class')):
                    code_started = True
                    code_lines.append(line)
            else:
                code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines).strip()
        
        # Last resort: return the whole response
        return response.strip()
    
    def refine_config(self, config_path: str, error_message: str,
                     problem: ProblemStructure,
                     hyperparameters: Optional[Dict[str, Any]] = None) -> str:
        """
        Refine a configuration file based on compilation/validation errors.

        Args:
            config_path: Path to the config file with errors
            error_message: The error message to fix
            problem: Original problem structure
            hyperparameters: Optional hyperparameters to preserve

        Returns:
            Path to the refined config file
        """
        
        print(f"🔧 Refining config based on errors...")
        
        # Read the current config
        with open(config_path, 'r') as f:
            current_code = f.read()
        
        # Build refinement prompt
        refinement_prompt = f"""The following BRKGA configuration has compilation or validation errors.
Please fix the issues while maintaining the overall structure and logic.

ORIGINAL PROBLEM:
{problem.summary()}

CURRENT CODE:
```cpp
{current_code}
```

ERROR MESSAGE:
{error_message}

Please provide the corrected version of the ENTIRE config file.

CRITICAL REQUIREMENTS:
1. Base class constructor: MUST use ONLY one of these forms:
   - Single component: BRKGAConfig<T>(chromosome_length)
   - Multi component: BRKGAConfig<T>({{length1, length2, ...}})
   - NEVER use: BRKGAConfig<T>(length, num_objectives, num_constraints) ❌

2. Function signatures MUST be:
   - fitness_function: [this](const Individual<T>& individual) -> T
   - decoder: [this](const Individual<T>& individual) -> std::vector<std::vector<T>>
   - comparator: [](T a, T b) -> bool
   - NEVER use: const std::vector<T>& chromosome ❌

3. Access chromosome via: individual.get_chromosome()

4. If config needs a data file, include a factory method:
   static std::unique_ptr<ConfigName<T>> create_from_file(const std::string& path)

Focus on:
- Fixing syntax errors
- Ensuring all required methods are properly implemented
- Correct lambda syntax
- Correct return types
"""

        # Add hyperparameters if provided (CRITICAL for refinement!)
        if hyperparameters:
            refinement_prompt += "\n" + "=" * 50 + "\n"
            refinement_prompt += "CRITICAL: PRESERVE THESE EXACT HYPERPARAMETER VALUES!\n"
            refinement_prompt += "=" * 50 + "\n"
            refinement_prompt += "The original config had these user-specified values.\n"
            refinement_prompt += "You MUST keep them exactly as shown:\n\n"
            if "population_size" in hyperparameters:
                refinement_prompt += f"  this->population_size = {hyperparameters['population_size']};\n"
            if "elite_percentage" in hyperparameters:
                elite_pct = hyperparameters['elite_percentage']
                refinement_prompt += f"  this->elite_size = static_cast<int>(this->population_size * {elite_pct});\n"
            if "mutant_percentage" in hyperparameters:
                mutant_pct = hyperparameters['mutant_percentage']
                refinement_prompt += f"  this->mutant_size = static_cast<int>(this->population_size * {mutant_pct});\n"
            if "elite_prob" in hyperparameters:
                refinement_prompt += f"  this->elite_prob = {hyperparameters['elite_prob']};\n"
            if "max_generations" in hyperparameters:
                refinement_prompt += f"  this->max_generations = {hyperparameters['max_generations']};\n"
            refinement_prompt += "\nDO NOT change these values! Copy them exactly!\n"
            refinement_prompt += "=" * 50 + "\n\n"

        refinement_prompt += "\nProvide ONLY the corrected C++ code, no explanations.\n"
        
        response = self.client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=8000,
            temperature=0.1,
            system=self.context,
            messages=[{
                "role": "user",
                "content": refinement_prompt
            }]
        )
        
        refined_code = self._extract_code(response.content[0].text)
        
        # Save refined version
        refined_path = config_path.replace('.hpp', '_refined.hpp')
        with open(refined_path, 'w') as f:
            f.write(refined_code)
        
        print(f"✅ Refined config: {refined_path}")
        
        return refined_path
    
    def generate_test_data(self, problem: ProblemStructure, 
                          output_path: str) -> str:
        """
        Generate test data file for the problem.
        
        Args:
            problem: The problem structure
            output_path: Where to save the test data
            
        Returns:
            Path to the generated data file
        """
        
        if not problem.requires_data_file:
            return ""
        
        prompt = f"""Generate test data for the following optimization problem:

Problem: {problem.problem_name}
Type: {problem.problem_type.value}
Scale: {problem.expected_scale}
Format: {problem.data_file_format}

Generate a small test instance suitable for quick validation.
Provide the data in the specified format.
"""
        
        response = self.client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=2000,
            temperature=0.3,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        data = response.content[0].text
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(data)
        
        return output_path
