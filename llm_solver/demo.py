#!/usr/bin/env python3
"""
Quick demo of the LLM BRKGA Solver system.
Solves a simple knapsack problem to demonstrate the system.
"""

import sys
import os

# Add to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import LLMBRKGASolver


def demo():
    """Run a simple demonstration."""
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║            LLM-POWERED BRKGA SOLVER DEMONSTRATION                ║
║                                                                  ║
║   Transform natural language into working optimization solvers   ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    # Simple knapsack problem
    problem = """
    I have a knapsack problem with 20 items.
    
    Each item has:
    - A weight (between 1 and 10 units)
    - A value (between 5 and 50 points)
    
    The knapsack has a capacity of 50 units.
    
    I want to maximize the total value of items in the knapsack
    without exceeding the weight capacity.
    """
    
    print("\n📝 PROBLEM DESCRIPTION:")
    print("-" * 70)
    print(problem)
    print("-" * 70)
    
    print("\n🚀 Starting LLM BRKGA Solver...")
    print("\nThis will:")
    print("  1. Analyze the problem using an LLM")
    print("  2. Generate custom C++ BRKGA configuration code")
    print("  3. Validate the generated code")
    print("  4. Compile the solver")
    print("  5. Run a quick test")
    print("\n" + "="*70)
    
    try:
        # Create solver
        solver = LLMBRKGASolver()
        
        # Solve with quick test only (for demo speed)
        session = solver.solve(
            problem,
            quick_test_only=True,
            max_iterations=3,
            verbose=True
        )
        
        # Print summary
        print("\n" + session.summary())
        
        if session.test_result and session.test_result.success:
            print("\n✨ SUCCESS! The system generated a working solver.")
            print("\n📁 Generated files:")
            print(f"   Config: {session.config_path}")
            print(f"   Executable: {session.executable_path}")
            print("\n💡 You can now use this solver for your knapsack problems!")
            
            # Offer to see the code
            print("\n" + "-"*70)
            show = input("Would you like to see the generated C++ code? (y/n): ").strip().lower()
            if show == 'y':
                print("\n" + "="*70)
                print("GENERATED C++ CODE")
                print("="*70)
                with open(session.config_path, 'r') as f:
                    print(f.read())
                print("="*70)
        else:
            print("\n⚠️  The solver generation encountered issues.")
            print("This can happen with complex problems or unclear descriptions.")
            print("The system attempted automatic refinement but needs manual review.")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nTroubleshooting:")
        print("  - Check that ANTHROPIC_API_KEY is set")
        print("  - Verify NVCC compiler is installed")
        print("  - Ensure CUDA toolkit is available")
        return
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\n📚 Next steps:")
    print("  - Try more examples: python examples.py all")
    print("  - Interactive mode: python examples.py interactive")
    print("  - Read README.md for full documentation")
    print("\n")


if __name__ == "__main__":
    demo()
