"""
Problem analyzer that uses LLM to parse natural language problem descriptions
and extract structured information for BRKGA code generation.
"""

import json
import os
from typing import Dict, Any, Optional
from anthropic import Anthropic
from .problem_structures import (
    ProblemStructure, ProblemType, DecisionVariable, Objective,
    Constraint, ConstraintType, ConstraintHandling, DecoderStrategy,
    LocalSearchType
)
from .data_parser import DataFileParser, DataMetadata, generate_data_context


class ProblemAnalyzer:
    """Analyzes natural language problem descriptions using LLM."""
    
    def __init__(self, context_package_path: str, api_key: Optional[str] = None):
        """
        Initialize the analyzer.

        Args:
            context_package_path: Path to the BRKGA context package
            api_key: Anthropic API key (optional, will use env var if not provided)
        """
        self.context_path = context_package_path
        self.client = Anthropic(api_key=api_key) if api_key else Anthropic()
        self.context = self._load_context()
        self.data_parser = DataFileParser()
    
    def _load_context(self) -> str:
        """Load the BRKGA context for the LLM."""

        # Try to load the full context from various locations
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

        # If not found, provide helpful error message
        raise FileNotFoundError(
            f"Cannot find BRKGA context file. Tried:\n" +
            "\n".join(f"  - {p}" for p in possible_paths)
        )
    
    def analyze_problem(self, problem_description: str,
                       clarifying_qa: Optional[Dict[str, str]] = None,
                       data_file_path: Optional[str] = None,
                       data_files: Optional[Dict[str, str]] = None) -> ProblemStructure:
        """
        Analyze a problem description and extract structured information.

        Args:
            problem_description: Natural language description of the problem
            clarifying_qa: Optional Q&A for clarification
            data_file_path: Optional path to data file (DEPRECATED - use data_files)
            data_files: Optional dict of named data files {"job_times": "path.csv", ...}

        Returns:
            ProblemStructure with extracted information
        """

        # Parse data files if provided
        all_data_metadata = {}
        files_to_parse = data_files if data_files else {}

        # Backward compatibility: handle single data_file_path
        if data_file_path and not data_files:
            files_to_parse = {"primary": data_file_path}

        # Parse each data file
        for file_name, file_path in files_to_parse.items():
            if os.path.exists(file_path):
                metadata = self.data_parser.parse_file(file_path)
                all_data_metadata[file_name] = metadata
            else:
                print(f"⚠️  Warning: Data file '{file_name}' not found at: {file_path}")

        # Build the analysis prompt
        analysis_prompt = self._build_analysis_prompt(
            problem_description, clarifying_qa, all_data_metadata
        )

        # Call LLM for analysis
        response = self.client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=4000,
            temperature=0.2,  # Lower temperature for more consistent analysis
            messages=[{
                "role": "user",
                "content": analysis_prompt
            }]
        )

        # Parse the response
        analysis_text = response.content[0].text
        problem_structure = self._parse_analysis(analysis_text)

        # Attach data metadata to problem structure if available
        if all_data_metadata:
            # Store all data files in the new format
            problem_structure.data_files = files_to_parse.copy()

            # For backward compatibility, store first file as data_file_path
            if data_file_path:
                problem_structure.data_file_path = data_file_path
            elif files_to_parse:
                problem_structure.data_file_path = list(files_to_parse.values())[0]

            # Combine all metadata
            combined_metadata = {}
            for name, metadata in all_data_metadata.items():
                combined_metadata[name] = metadata.to_dict()
            problem_structure.data_metadata = combined_metadata

        return problem_structure
    
    def _build_analysis_prompt(self, description: str,
                               clarifying_qa: Optional[Dict[str, str]] = None,
                               data_metadata: Optional[Dict[str, any]] = None) -> str:
        """Build the prompt for problem analysis.

        Args:
            description: Problem description
            clarifying_qa: Q&A pairs
            data_metadata: Either a single DataMetadata or dict of {name: DataMetadata}
        """

        prompt = f"""You are an expert in optimization and genetic algorithms. Analyze the following optimization problem description and extract structured information needed to generate a BRKGA solver.

PROBLEM DESCRIPTION:
{description}
"""

        if clarifying_qa:
            prompt += "\n\nCLARIFYING Q&A:\n"
            for q, a in clarifying_qa.items():
                prompt += f"Q: {q}\nA: {a}\n"

        # Add data file context if available
        if data_metadata:
            if isinstance(data_metadata, dict):
                # Multiple data files
                if len(data_metadata) > 0:
                    prompt += f"\n\nDATA FILES PROVIDED ({len(data_metadata)}):\n"
                    for file_name, metadata in data_metadata.items():
                        data_context = self.data_parser.generate_llm_context(metadata)
                        prompt += f"\n--- File: {file_name} ---\n{data_context}\n"
            else:
                # Single data file (backward compatibility)
                data_context = self.data_parser.generate_llm_context(data_metadata)
                prompt += f"\n\n{data_context}\n"

        prompt += """

Please analyze this problem and provide a structured analysis in JSON format with the following fields:

{
  "problem_name": "short descriptive name",
  "problem_type": "routing|packing|scheduling|assignment|selection|sequencing|partitioning|custom",
  "domain": "detailed problem description",
  "chromosome_length": <integer>,
  "decision_variables": [
    {
      "name": "variable name",
      "count": <integer>,
      "semantics": "what this represents"
    }
  ],
  "objectives": [
    {
      "name": "objective name",
      "type": "minimize|maximize",
      "description": "what to optimize",
      "formula_hint": "how to compute this"
    }
  ],
  "constraints": [
    {
      "description": "constraint description",
      "type": "hard|soft",
      "handling": "penalty|repair|decoder",
      "severity": <float 0-1>,
      "code_hint": "implementation hint"
    }
  ],
  "decoder_strategy": "sorted_index|threshold|assignment|priority|custom",
  "decoder_rationale": "why this decoder was chosen",
  "expected_scale": {
    "key": <value>
  },
  "requires_data_file": <boolean>,
  "data_file_format": "description if true",
  "complexity_estimate": "simple|moderate|complex",
  "gpu_beneficial": <boolean>,
  "local_search_recommended": <boolean>,
  "local_search_type": "none|two_opt|three_opt|swap|shift|insert|exchange",
  "local_search_rationale": "why local search is/isn't recommended",
  "notes": ["any important observations"]
}

Key considerations:
1. Chromosome length = total number of decision variables
2. Choose decoder strategy based on problem type:
   - sorted_index: for permutation problems (TSP, routing)
   - threshold: for selection problems (knapsack, item selection)
   - assignment: for binning/grouping problems
   - priority: for construction/scheduling problems
3. For multi-objective problems, list all objectives
4. Be specific about what each chromosome gene represents
5. Consider whether constraints are best handled in decoder, as penalties, or with repair
6. For local search, recommend based on problem type:
   - routing (TSP, VRP): two_opt or three_opt
   - scheduling: shift or swap
   - packing/selection (knapsack): swap
   - assignment: exchange
   - continuous optimization: none (not beneficial)
   - very large problems (>10000 variables): none (too slow)

Provide only the JSON response, no additional text.
"""
        
        return prompt
    
    def _parse_analysis(self, analysis_text: str) -> ProblemStructure:
        """Parse the LLM's analysis response into a ProblemStructure."""
        
        # Extract JSON from response
        import re
        json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
        if not json_match:
            raise ValueError("Could not find JSON in analysis response")
        
        analysis = json.loads(json_match.group())
        
        # Convert to ProblemStructure
        problem = ProblemStructure(
            problem_name=analysis["problem_name"],
            problem_type=ProblemType(analysis["problem_type"]),
            domain=analysis["domain"],
            chromosome_length=analysis["chromosome_length"],
            decision_variables=[
                DecisionVariable(**dv) for dv in analysis["decision_variables"]
            ],
            objectives=[
                Objective(**obj) for obj in analysis["objectives"]
            ],
            constraints=[
                Constraint(
                    description=c["description"],
                    type=ConstraintType(c["type"]),
                    handling=ConstraintHandling(c["handling"]),
                    severity=c.get("severity", 1.0),
                    code_hint=c.get("code_hint", "")
                )
                for c in analysis.get("constraints", [])
            ],
            decoder_strategy=DecoderStrategy(analysis["decoder_strategy"]),
            decoder_rationale=analysis["decoder_rationale"],
            expected_scale=analysis.get("expected_scale", {}),
            requires_data_file=analysis.get("requires_data_file", False),
            data_file_format=analysis.get("data_file_format", ""),
            complexity_estimate=analysis.get("complexity_estimate", "moderate"),
            gpu_beneficial=analysis.get("gpu_beneficial", False),
            notes=analysis.get("notes", []),
            local_search_recommended=analysis.get("local_search_recommended", False),
            local_search_type=LocalSearchType(analysis.get("local_search_type", "none")),
            local_search_rationale=analysis.get("local_search_rationale", "")
        )

        return problem
    
    def ask_clarifying_questions(self, problem_description: str) -> list[str]:
        """
        Generate clarifying questions if the problem description is ambiguous.

        Args:
            problem_description: The user's problem description

        Returns:
            List of clarifying questions to ask the user
        """

        prompt = f"""Analyze this optimization problem description and identify any ambiguities or missing information needed to generate a solver:

PROBLEM DESCRIPTION:
{problem_description}

IMPORTANT: The solver will use BRKGA (Biased Random-Key Genetic Algorithm), which is a metaheuristic approach that finds high-quality approximate solutions. Do NOT ask about exact vs. heuristic methods or solution approach - this is already determined.

Generate a list of clarifying questions (maximum 5) to resolve ambiguities. Focus on:
1. Problem scale (how many items, cities, jobs, etc.)
2. Specific constraints and their importance
3. Objective function details
4. Input data format if needed
5. Any domain-specific requirements

If the description is sufficiently clear, respond with "NO_QUESTIONS_NEEDED".

Provide questions as a JSON array:
["Question 1?", "Question 2?", ...]
"""
        
        response = self.client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1000,
            temperature=0.3,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        response_text = response.content[0].text.strip()
        
        if "NO_QUESTIONS_NEEDED" in response_text:
            return []
        
        # Extract JSON array
        import re
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        
        return []
    
    def estimate_problem_complexity(self, problem: ProblemStructure) -> Dict[str, Any]:
        """
        Estimate computational complexity and provide recommendations.

        Args:
            problem: The analyzed problem structure

        Returns:
            Dictionary with complexity estimates and recommendations
        """

        return {
            "chromosome_length": problem.chromosome_length,
            "complexity_class": problem.complexity_estimate,
            "recommended_population": self._estimate_population_size(problem),
            "recommended_generations": self._estimate_generations(problem),
            "estimated_time_minutes": self._estimate_runtime(problem),
            "gpu_recommended": problem.gpu_beneficial,
            "multi_objective": problem.is_multi_objective
        }

    def get_default_hyperparameters(self, problem: ProblemStructure) -> Dict[str, Any]:
        """
        Get default BRKGA hyperparameters based on problem characteristics.

        Args:
            problem: The analyzed problem structure

        Returns:
            Dictionary with default hyperparameter values
        """

        return {
            "population_size": self._estimate_population_size(problem),
            "elite_percentage": 0.15,  # 15% elite
            "mutant_percentage": 0.15,  # 15% mutants
            "elite_prob": 0.7,  # 70% bias towards elite parent
            "max_generations": self._estimate_generations(problem)
        }
    
    def _estimate_population_size(self, problem: ProblemStructure) -> int:
        """Estimate appropriate population size."""
        base_size = 100
        
        if problem.chromosome_length < 50:
            return base_size
        elif problem.chromosome_length < 200:
            return base_size * 2
        elif problem.chromosome_length < 500:
            return base_size * 3
        else:
            return base_size * 5
    
    def _estimate_generations(self, problem: ProblemStructure) -> int:
        """Estimate appropriate number of generations."""
        base_gens = 500
        
        if problem.complexity_estimate == "simple":
            return base_gens
        elif problem.complexity_estimate == "moderate":
            return base_gens * 2
        else:
            return base_gens * 3
    
    def _estimate_runtime(self, problem: ProblemStructure) -> float:
        """Estimate runtime in minutes."""
        pop = self._estimate_population_size(problem)
        gens = self._estimate_generations(problem)
        
        # Rough estimate: 1000 evaluations per second
        total_evals = pop * gens
        seconds = total_evals / 1000
        
        # Adjust for problem complexity
        if problem.complexity_estimate == "complex":
            seconds *= 2
        
        # Adjust for multi-objective (NSGA-II is slower)
        if problem.is_multi_objective:
            seconds *= 1.5
        
        return seconds / 60  # Convert to minutes
