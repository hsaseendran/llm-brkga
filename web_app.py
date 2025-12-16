#!/usr/bin/env python3
"""
Web Frontend for LLM BRKGA Solver
Provides a user-friendly interface for creating and running optimization solvers
"""

from flask import Flask, render_template, request, jsonify, send_file, session
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
import os
import sys
import json
import threading
import time
from datetime import datetime
import uuid

# Add llm_solver to path
sys.path.insert(0, os.path.dirname(__file__))

from llm_solver.core.llm_brkga_solver import LLMBRKGASolver
from llm_solver.core.problem_analyzer import ProblemAnalyzer
from llm_solver.core.data_parser import DataFileParser

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

socketio = SocketIO(app, cors_allowed_origins="*")

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Store active sessions
active_sessions = {}


class ProgressCallback:
    """Callback class to track and emit progress updates"""

    def __init__(self, session_id, socketio_instance):
        self.session_id = session_id
        self.socketio = socketio_instance
        self.current_stage = ""
        self.progress = 0

    def emit_progress(self, stage, progress, message=""):
        """Emit progress update to frontend"""
        self.current_stage = stage
        self.progress = progress
        self.socketio.emit('progress_update', {
            'session_id': self.session_id,
            'stage': stage,
            'progress': progress,
            'message': message
        })

    def emit_log(self, message, level="info"):
        """Emit log message to frontend"""
        self.socketio.emit('log_message', {
            'session_id': self.session_id,
            'message': message,
            'level': level,
            'timestamp': datetime.now().isoformat()
        })


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze_problem():
    """Analyze problem and generate clarifying questions"""
    try:
        data = request.json
        problem_description = data.get('problem_description', '')

        if not problem_description:
            return jsonify({'error': 'Problem description is required'}), 400

        # Create session
        session_id = str(uuid.uuid4())

        # Initialize analyzer
        analyzer = ProblemAnalyzer(context_package_path="llm_solver/context")

        # Get clarifying questions
        questions = analyzer.ask_clarifying_questions(problem_description)

        # Store session
        active_sessions[session_id] = {
            'problem_description': problem_description,
            'questions': questions,
            'answers': {},
            'hyperparameters': None,
            'data_file': None,  # DEPRECATED: kept for backward compatibility
            'data_files': {},    # NEW: multiple data files {'name': 'path'}
            'created_at': datetime.now().isoformat()
        }

        return jsonify({
            'session_id': session_id,
            'questions': questions
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload_data', methods=['POST'])
def upload_data():
    """Handle data file upload (supports multiple files)"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        session_id = request.form.get('session_id')
        file_name = request.form.get('file_name', 'primary')  # NEW: optional file name/label

        if not session_id or session_id not in active_sessions:
            return jsonify({'error': 'Invalid session'}), 400

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Check max files limit (from config)
        max_files = 10  # Could be loaded from config.yaml
        current_file_count = len(active_sessions[session_id]['data_files'])
        if current_file_count >= max_files:
            return jsonify({'error': f'Maximum {max_files} files allowed per problem'}), 400

        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        # Parse and validate file
        data_parser = DataFileParser()
        metadata = data_parser.parse_file(filepath)

        # Store in session (NEW: supports multiple files)
        active_sessions[session_id]['data_files'][file_name] = filepath

        # Also store in old field for backward compatibility
        if not active_sessions[session_id]['data_file']:
            active_sessions[session_id]['data_file'] = filepath

        return jsonify({
            'success': True,
            'file_name': file_name,  # The label/name for this file
            'filename': filename,     # The original filename
            'total_files': len(active_sessions[session_id]['data_files']),
            'metadata': {
                'format': metadata.format.value,
                'size': metadata.problem_size,
                'dimension_info': metadata.dimension_info,
                'preview': metadata.data_preview[:10] if isinstance(metadata.data_preview, list) else metadata.data_preview.split('\n')[:10]
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/get_required_parameters', methods=['POST'])
def get_required_parameters():
    """Get required file parameters from generated config"""
    try:
        data = request.json
        session_id = data.get('session_id')
        problem_description = data.get('problem_description', '')

        if not session_id or session_id not in active_sessions:
            return jsonify({'error': 'Invalid session'}), 400

        # For now, return empty list - will be filled after config generation
        # In a more sophisticated implementation, this could analyze the problem
        # or check existing templates

        # Simple heuristic based on problem description
        parameters = []
        desc_lower = problem_description.lower()

        if 'tsp' in desc_lower and 'job' in desc_lower:
            # TSPJ problem
            parameters = ['travel_time_file', 'job_time_file']
        elif 'vrp' in desc_lower:
            # VRP problem
            parameters = ['distance_file', 'demand_file']
        elif 'tsp' in desc_lower:
            # Regular TSP
            parameters = ['distance_file']

        return jsonify({
            'success': True,
            'parameters': parameters
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/solve', methods=['POST'])
def solve_problem():
    """Start solver in background thread"""
    try:
        data = request.json
        session_id = data.get('session_id')
        answers = data.get('answers', {})
        hyperparameters = data.get('hyperparameters', None)
        file_mapping = data.get('file_mapping', {})  # Get explicit file-to-parameter mapping

        if not session_id or session_id not in active_sessions:
            return jsonify({'error': 'Invalid session'}), 400

        # Update session with answers, hyperparameters, and file mapping
        active_sessions[session_id]['answers'] = answers
        active_sessions[session_id]['hyperparameters'] = hyperparameters
        active_sessions[session_id]['file_mapping'] = file_mapping
        active_sessions[session_id]['status'] = 'running'

        # Start solver in background thread
        thread = threading.Thread(
            target=run_solver,
            args=(session_id,)
        )
        thread.daemon = True
        thread.start()

        return jsonify({
            'success': True,
            'message': 'Solver started'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def run_solver(session_id):
    """Run solver and emit progress updates"""
    try:
        session_data = active_sessions[session_id]
        callback = ProgressCallback(session_id, socketio)

        callback.emit_log("Initializing solver...", "info")
        callback.emit_progress("initialization", 10, "Setting up solver components")

        # Initialize solver components
        solver = LLMBRKGASolver(
            context_package_path="llm_solver/context",
            framework_path="brkga",
            output_dir="llm_solver"
        )

        callback.emit_progress("analysis", 20, "Analyzing problem description")

        # Prepare clarifying Q&A
        clarifying_qa = None
        if session_data['answers']:
            clarifying_qa = {}
            for q, a in session_data['answers'].items():
                clarifying_qa[q] = a

        # Prepare data files
        data_files = session_data.get('data_files', {})
        file_mapping = session_data.get('file_mapping', {})
        data_file_path = None

        # Apply explicit file mapping if provided
        if file_mapping:
            mapped_data_files = {}
            for param_name, file_name in file_mapping.items():
                if file_name in data_files:
                    mapped_data_files[param_name] = data_files[file_name]
                    callback.emit_log(f"Mapped '{file_name}' to parameter '{param_name}'", "info")
            data_files = mapped_data_files

        # Only use old data_file format if data_files is empty (backward compatibility)
        if not data_files and session_data.get('data_file'):
            data_file_path = session_data.get('data_file')

        # Analyze problem and emit results
        merged_data_files = data_files or {}
        if data_file_path and "primary" not in merged_data_files:
            merged_data_files["primary"] = data_file_path

        problem_structure = solver.analyzer.analyze_problem(
            session_data['problem_description'],
            clarifying_qa,
            data_file_path,
            merged_data_files
        )

        # Emit analysis results
        analysis_data = {
            'problem_type': problem_structure.problem_type.value,
            'chromosome_length': problem_structure.chromosome_length,
            'objectives': [
                {
                    'name': obj.name,
                    'type': obj.type,
                    'weight': 1.0
                } for obj in problem_structure.objectives
            ],
            'constraints': [c.description for c in problem_structure.constraints],
            'decoder_strategy': problem_structure.decoder_strategy.value,
            'estimated_complexity': problem_structure.complexity_estimate
        }
        socketio.emit('analysis_complete', {
            'session_id': session_id,
            'analysis': analysis_data
        })
        callback.emit_log("Problem analysis complete", "success")

        # Get hyperparameters
        default_hyperparameters = solver.analyzer.get_default_hyperparameters(problem_structure)
        if session_data.get('hyperparameters'):
            final_hyperparameters = {**default_hyperparameters, **session_data['hyperparameters']}
        else:
            final_hyperparameters = default_hyperparameters

        callback.emit_progress("generation", 35, "Generating solver code...")
        callback.emit_log("Generating custom solver for your problem", "info")

        # Generate code
        config_name = problem_structure.problem_name.lower().replace(" ", "_")
        config_path = os.path.join("llm_solver", "generated", f"{config_name}_config.hpp")

        solver.generator.generate_config(
            problem_structure,
            config_path,
            hyperparameters=final_hyperparameters
        )

        # Read and emit generated code
        with open(config_path, 'r') as f:
            generated_code = f.read()

        socketio.emit('code_generated', {
            'session_id': session_id,
            'code': generated_code,
            'config_path': config_path
        })
        callback.emit_log("Code generation complete", "success")

        callback.emit_progress("compilation", 55, "Compiling solver...")

        # Compile
        executable_path = os.path.join("llm_solver", "generated", f"{config_name}_solver")
        compilation_result = solver.executor.compile_solver(
            config_path,
            executable_path,
            data_files=merged_data_files if merged_data_files else None
        )

        if not compilation_result.success:
            callback.emit_log("Compilation failed", "error")
            raise Exception("Compilation failed")

        callback.emit_log("Compilation successful!", "success")
        callback.emit_progress("execution", 70, "Running genetic algorithm...")

        # Run full optimization with convergence tracking
        result = run_optimization_with_tracking(
            solver.executor,
            executable_path,
            callback,
            session_id
        )

        callback.emit_progress("finalizing", 95, "Finalizing results...")

        # Read solution file if it exists
        solution_content = None
        solution_file = "llm_solver/results/solution.txt"
        if os.path.exists(solution_file):
            with open(solution_file, 'r') as f:
                solution_content = f.read()

        callback.emit_progress("complete", 100, "Optimization complete!")

        # Create result object
        from llm_solver.core.llm_brkga_solver import SolverSession
        session_result = SolverSession(problem_description=session_data['problem_description'])
        session_result.problem_structure = problem_structure
        session_result.config_path = config_path
        session_result.executable_path = executable_path
        session_result.compilation_result = compilation_result
        session_result.final_result = result
        session_result.success = result.success if result else False

        # Store results
        session_data['result'] = session_result
        session_data['status'] = 'completed' if session_result.success else 'failed'

        # Emit final results
        socketio.emit('solver_complete', {
            'session_id': session_id,
            'success': session_result.success,
            'result': {
                'best_fitness': result.best_fitness if result else None,
                'generations': result.generations if result else None,
                'execution_time': result.execution_time if result else None,
                'config_path': config_path,
                'executable_path': executable_path,
                'solution': solution_content
            }
        })

        if session_result.success:
            callback.emit_log("Solver completed successfully!", "success")
        else:
            callback.emit_log("Solver failed", "error")

    except Exception as e:
        import traceback
        traceback.print_exc()
        callback.emit_log(f"Error: {str(e)}", "error")
        socketio.emit('solver_error', {
            'session_id': session_id,
            'error': str(e)
        })
        active_sessions[session_id]['status'] = 'error'


def run_optimization_with_tracking(executor, executable_path, callback, session_id):
    """Run optimization and track convergence in real-time"""
    import subprocess
    import re
    from llm_solver.core.problem_structures import ExecutionResult

    callback.emit_log("Starting optimization...", "info")

    try:
        # Run the executable and capture output in real-time
        process = subprocess.Popen(
            [executable_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Pattern to match generation output: "Generation   123: Best=45.67, Avg=50.12, Worst=55.89"
        pattern = re.compile(r'Generation\s+(\d+):\s+Best=([\d.]+),\s+Avg=([\d.]+),\s+Worst=([\d.]+)')

        output_lines = []
        best_fitness = None
        generations = 0

        # Read output line by line
        for line in process.stdout:
            output_lines.append(line)

            # Check for generation output
            match = pattern.search(line)
            if match:
                generation = int(match.group(1))
                best = float(match.group(2))
                avg = float(match.group(3))
                worst = float(match.group(4))

                # Emit convergence update
                socketio.emit('convergence_update', {
                    'session_id': session_id,
                    'generation': generation,
                    'best': best,
                    'avg': avg,
                    'worst': worst
                })

                best_fitness = best
                generations = generation

                # Update progress message every 10 generations
                if generation % 10 == 0:
                    callback.emit_log(f"Generation {generation}: Best={best:.2f}", "info")

        # Wait for process to complete
        return_code = process.wait(timeout=600)  # 10 minute timeout

        # Get stderr
        stderr_output = process.stderr.read()

        # Parse final output
        full_output = ''.join(output_lines)

        # Create execution result
        result = ExecutionResult(
            success=(return_code == 0),
            output=full_output,
            best_fitness=best_fitness,
            generations=generations,
            execution_time=0.0,  # Could parse from output
            solution_valid=True
        )

        return result

    except subprocess.TimeoutExpired:
        process.kill()
        callback.emit_log("Optimization timed out", "error")
        return ExecutionResult(success=False, output="Timeout", errors=["Execution timed out"])
    except Exception as e:
        callback.emit_log(f"Optimization error: {str(e)}", "error")
        return ExecutionResult(success=False, output=str(e), errors=[str(e)])


@app.route('/api/export/<session_id>')
def export_solution(session_id):
    """Export solution file"""
    try:
        if session_id not in active_sessions:
            return jsonify({'error': 'Invalid session'}), 400

        solution_file = "llm_solver/results/solution.txt"
        if not os.path.exists(solution_file):
            return jsonify({'error': 'Solution file not found'}), 404

        return send_file(
            solution_file,
            as_attachment=True,
            download_name=f'solution_{session_id[:8]}.txt'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/session/<session_id>')
def get_session(session_id):
    """Get session details"""
    if session_id not in active_sessions:
        return jsonify({'error': 'Session not found'}), 404

    session_data = active_sessions[session_id]

    return jsonify({
        'session_id': session_id,
        'problem_description': session_data['problem_description'],
        'status': session_data.get('status', 'pending'),
        'created_at': session_data['created_at']
    })


@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f"Client connected: {request.sid}")
    emit('connected', {'message': 'Connected to solver server'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f"Client disconnected: {request.sid}")


if __name__ == '__main__':
    print("=" * 70)
    print("LLM BRKGA Solver - Web Interface")
    print("=" * 70)
    print("\nStarting server...")
    print("\nAccess the application at: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 70)

    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
