#!/usr/bin/env python3
"""
Run all RCMADP Interleaved benchmarks with MAKESPAN optimization.

Usage:
    python run_benchmarks_makespan.py
    python run_benchmarks_makespan.py --pop 6000 --gen 30000
    python run_benchmarks_makespan.py --output results_makespan
"""

import os
import sys
import subprocess
import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


# Configuration - USE MAKESPAN SOLVER
SOLVER_PATH = "brkga/rcmadp_interleaved_makespan_solver"
BENCHMARK_DIR = "brkga/data/Benchmarks"
REPORT_SCRIPT = "generate_html_report.py"

# Benchmark definitions: (tsp_file, proc_times_file, name)
BENCHMARKS = [
    ("berlin52.tsp", "berlin52_full_jobs_proc_times.txt", "berlin52"),
    ("berlin52.tsp", "berlin52_full_jobs_1R10_proc_times.txt", "berlin52_1R10"),
    ("berlin52.tsp", "berlin52_full_jobs_2_proc_times.txt", "berlin52_2x"),
    ("berlin52.tsp", "berlin52_full_jobs_5_proc_times.txt", "berlin52_5x"),
    ("eil51.tsp", "eil51_full_jobs_proc_times.txt", "eil51"),
    ("eil51.tsp", "eil51_full_jobs_1R10_proc_times.txt", "eil51_1R10"),
    ("eil51.tsp", "eil51_full_jobs_2_proc_times.txt", "eil51_2x"),
    ("eil51.tsp", "eil51_full_jobs_5_proc_times.txt", "eil51_5x"),
    ("kroA100.tsp", "kroA100_full_jobs_proc_times.txt", "kroA100"),
    ("kroA100.tsp", "kroA100_full_jobs_1R10_proc_times.txt", "kroA100_1R10"),
    ("kroA100.tsp", "kroA100_full_jobs_2_proc_times.txt", "kroA100_2x"),
    ("kroA100.tsp", "kroA100_full_jobs_5_proc_times.txt", "kroA100_5x"),
]

# Default parameters
DEFAULT_AGENTS = 6
DEFAULT_RESOURCES = 4
DEFAULT_POP = 6000
DEFAULT_GEN = 30000


def run_benchmark(tsp_file, proc_file, name, output_dir, pop, gen, agents, resources):
    """Run a single benchmark and return results."""
    tsp_path = os.path.join(BENCHMARK_DIR, tsp_file)
    proc_path = os.path.join(BENCHMARK_DIR, proc_file)

    # Create output subdirectory for this benchmark
    bench_dir = os.path.join(output_dir, name)
    os.makedirs(bench_dir, exist_ok=True)

    json_output = os.path.join(bench_dir, "soln.json")

    # Build command
    cmd = [
        f"./{SOLVER_PATH}",
        tsp_path,
        proc_path,
        str(agents),
        str(resources),
        "--pop", str(pop),
        "--gen", str(gen),
        "--output", json_output
    ]

    print(f"\n{'='*60}")
    print(f"Running: {name} (MAKESPAN OPTIMIZATION)")
    print(f"  TSP: {tsp_file}")
    print(f"  Processing times: {proc_file}")
    print(f"  Population: {pop}, Generations: {gen}")
    print(f"  Agents: {agents}, Resources/Agent: {resources}")
    print(f"  Objective: MINIMIZE MAKESPAN")
    print(f"{'='*60}")

    # Run solver with real-time output
    start_time = datetime.now()
    stdout_log = []
    stderr_log = []

    try:
        # Use Popen for real-time output streaming
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # Line buffered
        )

        # Stream stdout in real-time
        while True:
            # Check if process has finished
            retcode = process.poll()

            # Read available output
            if process.stdout:
                line = process.stdout.readline()
                if line:
                    print(line, end='', flush=True)
                    stdout_log.append(line)

            if retcode is not None:
                # Process finished, read remaining output
                remaining_stdout, remaining_stderr = process.communicate()
                if remaining_stdout:
                    print(remaining_stdout, end='', flush=True)
                    stdout_log.append(remaining_stdout)
                if remaining_stderr:
                    stderr_log.append(remaining_stderr)
                break

        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()

        # Save stdout/stderr to files
        with open(os.path.join(bench_dir, "stdout.txt"), "w") as f:
            f.write(''.join(stdout_log))
        with open(os.path.join(bench_dir, "stderr.txt"), "w") as f:
            f.write(''.join(stderr_log))

        # Check if JSON was created
        if os.path.exists(json_output):
            with open(json_output) as f:
                data = json.load(f)

            print(f"\n  SUCCESS: Makespan = {data['makespan']:.2f}, Time = {elapsed:.1f}s")
            return {
                "name": name,
                "success": True,
                "makespan": data["makespan"],
                "fitness": data["fitness"],
                "elapsed": elapsed,
                "json_path": json_output
            }
        else:
            print(f"  FAILED: No JSON output created")
            print(f"  stderr: {''.join(stderr_log)[:500]}")
            return {"name": name, "success": False, "error": "No output"}

    except Exception as e:
        print(f"  ERROR: {str(e)}")
        return {"name": name, "success": False, "error": str(e)}


def generate_reports(output_dir):
    """Generate HTML reports for all results."""
    print(f"\n{'='*60}")
    print("Generating HTML Reports")
    print(f"{'='*60}")

    # Generate individual reports for each benchmark
    for name in os.listdir(output_dir):
        bench_dir = os.path.join(output_dir, name)
        if os.path.isdir(bench_dir) and os.path.exists(os.path.join(bench_dir, "soln.json")):
            report_path = os.path.join(bench_dir, "report.html")
            cmd = ["python3", REPORT_SCRIPT, bench_dir, "-o", report_path]
            subprocess.run(cmd, capture_output=True)
            print(f"  Generated: {report_path}")

    # Generate combined report
    combined_report = os.path.join(output_dir, "combined_report.html")
    cmd = ["python3", REPORT_SCRIPT, output_dir, "-o", combined_report]
    subprocess.run(cmd, capture_output=True)
    print(f"  Generated combined report: {combined_report}")


def write_summary(output_dir, results, args):
    """Write a summary JSON and markdown file."""
    summary = {
        "run_date": datetime.now().isoformat(),
        "optimization_objective": "MAKESPAN (minimize completion time)",
        "parameters": {
            "population": args.pop,
            "generations": args.gen,
            "agents": args.agents,
            "resources_per_agent": args.resources
        },
        "results": results
    }

    # Write JSON summary
    summary_json = os.path.join(output_dir, "summary.json")
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    # Write markdown summary
    summary_md = os.path.join(output_dir, "README.md")
    with open(summary_md, "w") as f:
        f.write(f"# RCMADP Interleaved Benchmark Results (MAKESPAN OPTIMIZATION)\n\n")
        f.write(f"**Optimization Objective:** MINIMIZE MAKESPAN (completion time)\n\n")
        f.write(f"**Run Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Parameters\n\n")
        f.write(f"- Population: {args.pop}\n")
        f.write(f"- Generations: {args.gen}\n")
        f.write(f"- Agents: {args.agents}\n")
        f.write(f"- Resources per Agent: {args.resources}\n\n")
        f.write(f"## Results\n\n")
        f.write(f"| Benchmark | Makespan | Fitness | Time (s) | Status |\n")
        f.write(f"|-----------|----------|---------|----------|--------|\n")

        for r in results:
            if r["success"]:
                f.write(f"| {r['name']} | {r['makespan']:.2f} | {r['fitness']:.2f} | {r['elapsed']:.1f} | OK |\n")
            else:
                f.write(f"| {r['name']} | - | - | - | {r.get('error', 'Failed')} |\n")

        f.write(f"\n## Files\n\n")
        f.write(f"- `combined_report.html` - Combined HTML report with all benchmarks\n")
        f.write(f"- `summary.json` - Machine-readable summary\n")
        f.write(f"- `<benchmark>/soln.json` - Solution for each benchmark\n")
        f.write(f"- `<benchmark>/report.html` - Individual HTML report\n")

    print(f"\nSummary written to: {summary_md}")


def main():
    parser = argparse.ArgumentParser(description="Run RCMADP Interleaved benchmarks with MAKESPAN optimization")
    parser.add_argument("--pop", type=int, default=DEFAULT_POP,
                        help=f"Population size (default: {DEFAULT_POP})")
    parser.add_argument("--gen", type=int, default=DEFAULT_GEN,
                        help=f"Number of generations (default: {DEFAULT_GEN})")
    parser.add_argument("--agents", type=int, default=DEFAULT_AGENTS,
                        help=f"Number of agents (default: {DEFAULT_AGENTS})")
    parser.add_argument("--resources", type=int, default=DEFAULT_RESOURCES,
                        help=f"Resources per agent (default: {DEFAULT_RESOURCES})")
    parser.add_argument("--output", "-o", type=str,
                        default=f"benchmark_results_makespan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        help="Output directory")
    parser.add_argument("--benchmarks", "-b", nargs="+",
                        help="Specific benchmarks to run (e.g., berlin52 eil51)")
    parser.add_argument("--list", action="store_true",
                        help="List available benchmarks")
    args = parser.parse_args()

    if args.list:
        print("Available benchmarks:")
        for _, _, name in BENCHMARKS:
            print(f"  {name}")
        return

    # Change to the llm-brkga directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Check if solver exists
    if not os.path.exists(SOLVER_PATH):
        print(f"ERROR: Makespan solver not found at {SOLVER_PATH}")
        print("Please compile it first with:")
        print("  cd brkga && nvcc -O3 -std=c++17 -arch=sm_75 -Xcompiler -fopenmp -I. \\")
        print("      examples/rcmadp_interleaved_makespan_example.cu \\")
        print("      -o rcmadp_interleaved_makespan_solver -lcurand -lcudart")
        sys.exit(1)

    # Create output directory
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"# RCMADP Interleaved Benchmark Suite")
    print(f"# OPTIMIZATION OBJECTIVE: MINIMIZE MAKESPAN")
    print(f"# Population: {args.pop}, Generations: {args.gen}")
    print(f"# Agents: {args.agents}, Resources/Agent: {args.resources}")
    print(f"# Output: {output_dir}")
    print(f"{'#'*60}")

    # Filter benchmarks if specified
    benchmarks = BENCHMARKS
    if args.benchmarks:
        benchmarks = [b for b in BENCHMARKS if b[2] in args.benchmarks]
        if not benchmarks:
            print(f"ERROR: No matching benchmarks found for: {args.benchmarks}")
            print(f"Available: {[b[2] for b in BENCHMARKS]}")
            sys.exit(1)

    # Run benchmarks
    results = []
    total_start = datetime.now()

    for tsp_file, proc_file, name in benchmarks:
        result = run_benchmark(
            tsp_file, proc_file, name, output_dir,
            args.pop, args.gen, args.agents, args.resources
        )
        results.append(result)

    total_elapsed = (datetime.now() - total_start).total_seconds()

    # Generate reports
    generate_reports(output_dir)

    # Write summary
    write_summary(output_dir, results, args)

    # Print final summary
    print(f"\n{'#'*60}")
    print(f"# Benchmark Suite Complete (MAKESPAN OPTIMIZATION)")
    print(f"# Total time: {total_elapsed:.1f}s ({total_elapsed/3600:.2f} hours)")
    print(f"# Results saved to: {output_dir}")
    print(f"{'#'*60}")

    successful = sum(1 for r in results if r["success"])
    print(f"\nSuccessful: {successful}/{len(results)}")

    if successful > 0:
        best = min((r for r in results if r["success"]), key=lambda x: x["makespan"])
        print(f"Best result: {best['name']} with makespan {best['makespan']:.2f}")


if __name__ == "__main__":
    main()
