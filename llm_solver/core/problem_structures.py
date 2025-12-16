"""
Data structures for representing optimization problems in the LLM BRKGA solver.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class ProblemType(Enum):
    """Classification of optimization problem types."""
    ROUTING = "routing"
    PACKING = "packing"
    SCHEDULING = "scheduling"
    ASSIGNMENT = "assignment"
    SELECTION = "selection"
    SEQUENCING = "sequencing"
    PARTITIONING = "partitioning"
    CUSTOM = "custom"


class DecoderStrategy(Enum):
    """Decoder strategy patterns for BRKGA."""
    SORTED_INDEX = "sorted_index"  # For permutation problems (TSP, routing)
    THRESHOLD = "threshold"  # For selection problems (knapsack)
    ASSIGNMENT = "assignment"  # For binning problems
    PRIORITY = "priority"  # For construction/scheduling
    CUSTOM = "custom"  # For novel problems


class ConstraintType(Enum):
    """Types of constraints."""
    HARD = "hard"  # Must be satisfied
    SOFT = "soft"  # Prefer to satisfy


class ConstraintHandling(Enum):
    """How to handle constraints."""
    PENALTY = "penalty"  # Add penalty to fitness
    REPAIR = "repair"  # Fix solutions after decoding
    DECODER = "decoder"  # Build valid solutions in decoder


class LocalSearchType(Enum):
    """Types of local search operators."""
    NONE = "none"  # No local search
    TWO_OPT = "two_opt"  # 2-opt for routing problems
    THREE_OPT = "three_opt"  # 3-opt for routing problems
    SWAP = "swap"  # Swap moves for selection/assignment
    SHIFT = "shift"  # Shift moves for scheduling
    INSERT = "insert"  # Insert moves for sequencing
    EXCHANGE = "exchange"  # Exchange moves for assignment


@dataclass
class Constraint:
    """Represents a problem constraint."""
    description: str
    type: ConstraintType
    handling: ConstraintHandling
    severity: float = 1.0  # Penalty weight for soft constraints
    code_hint: str = ""  # Hint for code generation


@dataclass
class Objective:
    """Represents an optimization objective."""
    name: str
    type: str  # "minimize" or "maximize"
    description: str
    formula_hint: str = ""  # Natural language formula
    complexity: str = "moderate"  # "simple", "moderate", "complex"


@dataclass
class DecisionVariable:
    """Represents a decision variable or group of variables."""
    name: str
    count: int  # Number of such variables
    semantics: str  # What this variable represents
    range: tuple = (0.0, 1.0)  # Valid range


@dataclass
class ProblemStructure:
    """Complete structured representation of an optimization problem."""
    
    # Basic identification
    problem_name: str
    problem_type: ProblemType
    domain: str  # Natural language description
    
    # Problem dimensions
    decision_variables: List[DecisionVariable]
    chromosome_length: int
    
    # Optimization goals
    objectives: List[Objective]
    is_multi_objective: bool = False
    
    # Constraints
    constraints: List[Constraint] = field(default_factory=list)
    
    # Decoder selection
    decoder_strategy: DecoderStrategy = DecoderStrategy.CUSTOM
    decoder_rationale: str = ""
    
    # Problem scale expectations
    expected_scale: Dict[str, int] = field(default_factory=dict)
    
    # Data requirements
    requires_data_file: bool = False
    data_file_format: str = ""
    data_file_path: Optional[str] = None  # DEPRECATED: Use data_files instead (kept for backward compatibility)
    data_files: Dict[str, str] = field(default_factory=dict)  # NEW: Multiple data files {"name": "path"}
    data_metadata: Optional[Dict[str, any]] = None  # Parsed metadata from data file(s)

    # Additional metadata
    complexity_estimate: str = "moderate"  # "simple", "moderate", "complex"
    gpu_beneficial: bool = False
    notes: List[str] = field(default_factory=list)

    # Local search configuration
    local_search_recommended: bool = False
    local_search_type: LocalSearchType = LocalSearchType.NONE
    local_search_rationale: str = ""
    
    def __post_init__(self):
        """Validate the problem structure."""
        if not self.objectives:
            raise ValueError("At least one objective is required")
        
        if self.chromosome_length <= 0:
            raise ValueError("Chromosome length must be positive")
        
        self.is_multi_objective = len(self.objectives) > 1
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Problem: {self.problem_name}",
            f"Type: {self.problem_type.value}",
            f"Chromosome length: {self.chromosome_length}",
            f"Objectives: {len(self.objectives)} ({'multi' if self.is_multi_objective else 'single'})",
            f"Constraints: {len(self.constraints)}",
            f"Decoder: {self.decoder_strategy.value}",
            f"Local search: {'Yes (' + self.local_search_type.value + ')' if self.local_search_recommended else 'No'}",
        ]
        return "\n".join(lines)


@dataclass
class ValidationResult:
    """Result of code validation."""
    success: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    def report(self) -> str:
        """Generate validation report."""
        lines = []
        if self.success:
            lines.append("✅ Validation passed")
        else:
            lines.append("❌ Validation failed")
        
        if self.errors:
            lines.append("\nErrors:")
            for err in self.errors:
                lines.append(f"  - {err}")
        
        if self.warnings:
            lines.append("\nWarnings:")
            for warn in self.warnings:
                lines.append(f"  - {warn}")
        
        if self.suggestions:
            lines.append("\nSuggestions:")
            for sug in self.suggestions:
                lines.append(f"  - {sug}")
        
        return "\n".join(lines)


@dataclass
class CompilationResult:
    """Result of compilation attempt."""
    success: bool
    output: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    compilation_time: float = 0.0
    
    def report(self) -> str:
        """Generate compilation report."""
        lines = []
        if self.success:
            lines.append(f"✅ Compilation successful ({self.compilation_time:.2f}s)")
        else:
            lines.append("❌ Compilation failed")
            if self.errors:
                lines.append("\nErrors:")
                for err in self.errors:
                    lines.append(f"  {err}")
        
        if self.warnings:
            lines.append(f"\n⚠️  {len(self.warnings)} warning(s)")
        
        return "\n".join(lines)


@dataclass
class ExecutionResult:
    """Result of solver execution."""
    success: bool
    output: str
    best_fitness: Optional[float] = None
    generations: int = 0
    execution_time: float = 0.0
    solution_valid: bool = False
    pareto_front_size: Optional[int] = None  # For multi-objective
    errors: List[str] = field(default_factory=list)
    
    def report(self) -> str:
        """Generate execution report."""
        lines = []
        if self.success:
            lines.append(f"✅ Execution successful ({self.execution_time:.2f}s)")
            if self.best_fitness is not None:
                lines.append(f"Best fitness: {self.best_fitness:.6f}")
            if self.pareto_front_size is not None:
                lines.append(f"Pareto front size: {self.pareto_front_size}")
            lines.append(f"Generations: {self.generations}")
        else:
            lines.append("❌ Execution failed")
            for err in self.errors:
                lines.append(f"  - {err}")
        
        return "\n".join(lines)
