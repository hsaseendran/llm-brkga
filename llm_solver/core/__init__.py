"""
LLM-powered BRKGA Solver System

A system that enables users to solve optimization problems by describing them
in natural language. The system analyzes the problem, generates custom BRKGA
configuration code, compiles and executes the solver, and presents results.
"""

from .problem_structures import (
    ProblemType,
    DecoderStrategy,
    ConstraintType,
    ConstraintHandling,
    Constraint,
    Objective,
    DecisionVariable,
    ProblemStructure,
    ValidationResult,
    CompilationResult,
    ExecutionResult
)

from .problem_analyzer import ProblemAnalyzer
from .code_generator import CodeGenerator
from .validator import Validator
from .execution_manager import ExecutionManager
from .llm_brkga_solver import LLMBRKGASolver, SolverSession

__version__ = "1.0.0"

__all__ = [
    # Main solver
    'LLMBRKGASolver',
    'SolverSession',
    
    # Components
    'ProblemAnalyzer',
    'CodeGenerator',
    'Validator',
    'ExecutionManager',
    
    # Data structures
    'ProblemType',
    'DecoderStrategy',
    'ConstraintType',
    'ConstraintHandling',
    'Constraint',
    'Objective',
    'DecisionVariable',
    'ProblemStructure',
    'ValidationResult',
    'CompilationResult',
    'ExecutionResult',
]
