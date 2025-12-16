// Global state
let sessionId = null;
let questions = [];
let answers = {};
let hyperparameters = null;
let socket = null;
let uploadedFiles = {};  // Track uploaded files: {file_name: {filename, metadata}}
let fileMapping = {};     // Track file-to-parameter mapping: {param_name: file_name}
let convergenceChart = null;  // Chart.js instance for convergence graph
let resultsConvergenceChart = null;  // Chart.js instance for results page
let convergenceData = {
    generations: [],
    bestFitness: [],
    avgFitness: [],
    worstFitness: []
};
let analysisData = null;  // Store analysis data for results page

// Initialize Socket.IO
function initSocket() {
    socket = io();

    socket.on('connect', () => {
        console.log('Connected to server');
        addLog('Connected to solver server', 'success');
    });

    socket.on('disconnect', () => {
        console.log('Disconnected from server');
    });

    socket.on('progress_update', (data) => {
        if (data.session_id === sessionId) {
            updateProgress(data.progress, data.stage, data.message);
        }
    });

    socket.on('log_message', (data) => {
        if (data.session_id === sessionId) {
            addLog(data.message, data.level);
        }
    });

    socket.on('solver_complete', (data) => {
        if (data.session_id === sessionId) {
            showResults(data);
        }
    });

    socket.on('solver_error', (data) => {
        if (data.session_id === sessionId) {
            hideLoading();
            alert('Error: ' + data.error);
        }
    });

    socket.on('analysis_complete', (data) => {
        if (data.session_id === sessionId) {
            displayAnalysisResults(data.analysis);
        }
    });

    socket.on('code_generated', (data) => {
        if (data.session_id === sessionId) {
            displayGeneratedCode(data.code, data.config_path);
        }
    });

    socket.on('convergence_update', (data) => {
        if (data.session_id === sessionId) {
            updateConvergenceGraph(data);
        }
    });
}

// DOM Ready
document.addEventListener('DOMContentLoaded', () => {
    initSocket();
    initEventListeners();
});

// Event Listeners
function initEventListeners() {
    // Step 1: Analyze problem
    document.getElementById('analyze-btn').addEventListener('click', analyzeProblem);

    // Step 3: Hyperparameters
    document.getElementById('skip-hyperparams-btn').addEventListener('click', skipHyperparameters);
    document.getElementById('apply-hyperparams-btn').addEventListener('click', applyHyperparameters);

    // Step 4: File upload
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');

    browseBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    uploadArea.addEventListener('click', () => fileInput.click());

    document.getElementById('skip-file-btn').addEventListener('click', skipFile);
    document.getElementById('proceed-solve-btn').addEventListener('click', checkFileMapping);
    document.getElementById('confirm-mapping-btn').addEventListener('click', confirmMapping);

    // Step 5: Results
    document.getElementById('export-btn').addEventListener('click', exportSolution);
    document.getElementById('copy-solution-btn').addEventListener('click', copySolution);
    document.getElementById('new-problem-btn').addEventListener('click', resetApp);

    // Expand code button
    document.getElementById('expand-code-btn').addEventListener('click', toggleCodeExpansion);
}

// Step 1: Analyze Problem
async function analyzeProblem() {
    const problemDescription = document.getElementById('problem-description').value.trim();

    if (!problemDescription) {
        alert('Please enter a problem description');
        return;
    }

    showLoading('Analyzing your problem...');

    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ problem_description: problemDescription })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Analysis failed');
        }

        sessionId = data.session_id;
        questions = data.questions;

        hideLoading();
        showQuestions();

    } catch (error) {
        hideLoading();
        alert('Error: ' + error.message);
    }
}

// Step 2: Show Questions
function showQuestions() {
    const container = document.getElementById('questions-container');

    if (!questions || questions.length === 0) {
        container.innerHTML = '<div class="no-questions">No clarifying questions needed! Your problem description is clear.</div>';
        document.getElementById('skip-questions-btn').style.display = 'block';
    } else {
        container.innerHTML = '';
        questions.forEach((question, index) => {
            const questionDiv = document.createElement('div');
            questionDiv.className = 'question-item';
            questionDiv.innerHTML = `
                <div class="question-text">${index + 1}. ${question}</div>
                <input type="text" id="answer-${index}" class="answer-input" placeholder="Your answer...">
            `;
            container.appendChild(questionDiv);
        });

        const submitBtn = document.createElement('button');
        submitBtn.className = 'btn btn-primary';
        submitBtn.textContent = 'Submit Answers';
        submitBtn.addEventListener('click', submitAnswers);
        container.appendChild(submitBtn);
    }

    showStep(2);
}

function submitAnswers() {
    answers = {};
    let allAnswered = true;

    questions.forEach((question, index) => {
        const answerInput = document.getElementById(`answer-${index}`);
        const answer = answerInput.value.trim();

        if (!answer) {
            allAnswered = false;
            answerInput.style.borderColor = 'var(--danger-color)';
        } else {
            answerInput.style.borderColor = '';
            answers[question] = answer;
        }
    });

    if (!allAnswered && questions.length > 0) {
        alert('Please answer all questions');
        return;
    }

    showStep(3);
}

// Step 3: Hyperparameters
function skipHyperparameters() {
    hyperparameters = null;  // Use defaults
    showStep(4);
}

function applyHyperparameters() {
    // Collect hyperparameter values
    const popSize = document.getElementById('population-size').value;
    const elitePct = document.getElementById('elite-percentage').value;
    const mutantPct = document.getElementById('mutant-percentage').value;
    const eliteProb = document.getElementById('elite-prob').value;
    const maxGens = document.getElementById('max-generations').value;

    // Build hyperparameters object (only include non-empty values)
    hyperparameters = {};
    if (popSize) hyperparameters.population_size = parseInt(popSize);
    if (elitePct) hyperparameters.elite_percentage = parseFloat(elitePct);
    if (mutantPct) hyperparameters.mutant_percentage = parseFloat(mutantPct);
    if (eliteProb) hyperparameters.elite_prob = parseFloat(eliteProb);
    if (maxGens) hyperparameters.max_generations = parseInt(maxGens);

    // If no values were entered, set to null (use defaults)
    if (Object.keys(hyperparameters).length === 0) {
        hyperparameters = null;
    }

    showStep(4);
}

// Step 4: File Upload
function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

async function handleFile(file) {
    // Check max files limit
    if (Object.keys(uploadedFiles).length >= 10) {
        alert('Maximum 10 files allowed per problem');
        return;
    }

    // Prompt user for a file label/name
    const fileName = prompt(`Enter a descriptive name for this file (e.g., "cities", "distances", "demands"):\n\nFilename: ${file.name}`,
                            file.name.replace(/\.[^/.]+$/, "")); // Default: filename without extension

    if (!fileName) {
        return; // User cancelled
    }

    // Check if name already exists
    if (uploadedFiles[fileName]) {
        alert(`A file with the name "${fileName}" already exists. Please choose a different name.`);
        return;
    }

    showLoading('Uploading and parsing file...');

    const formData = new FormData();
    formData.append('file', file);
    formData.append('session_id', sessionId);
    formData.append('file_name', fileName);

    try {
        const response = await fetch('/api/upload_data', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Upload failed');
        }

        hideLoading();

        // Store file info
        uploadedFiles[fileName] = {
            filename: file.name,
            metadata: data.metadata
        };

        // Update file list display
        updateFilesList();

        // Reset file input
        document.getElementById('file-input').value = '';

        // Show proceed button if at least one file uploaded
        document.getElementById('proceed-solve-btn').style.display = 'inline-block';

    } catch (error) {
        hideLoading();
        alert('Error: ' + error.message);
    }
}

function updateFilesList() {
    const filesList = document.getElementById('files-list');

    if (Object.keys(uploadedFiles).length === 0) {
        filesList.innerHTML = '';
        document.getElementById('proceed-solve-btn').style.display = 'none';
        return;
    }

    filesList.innerHTML = '<h3>Uploaded Files:</h3>';

    for (const [fileName, fileInfo] of Object.entries(uploadedFiles)) {
        const fileCard = document.createElement('div');
        fileCard.className = 'file-card';
        fileCard.innerHTML = `
            <div class="file-card-header">
                <div class="file-card-title">
                    <strong>${fileName}</strong>
                    <span class="file-card-filename">(${fileInfo.filename})</span>
                </div>
                <button class="btn-remove" onclick="removeFile('${fileName}')" title="Remove file">✕</button>
            </div>
            <div class="file-card-info">
                <span><strong>Format:</strong> ${fileInfo.metadata.format}</span>
                <span><strong>Size:</strong> ${fileInfo.metadata.size} items</span>
                ${fileInfo.metadata.dimension_info ? `<span><strong>Dimensions:</strong> ${fileInfo.metadata.dimension_info}</span>` : ''}
            </div>
            <div class="file-card-preview">
                <details>
                    <summary>Preview</summary>
                    <pre>${fileInfo.metadata.preview.join('\n')}</pre>
                </details>
            </div>
        `;
        filesList.appendChild(fileCard);
    }
}

function removeFile(fileName) {
    if (confirm(`Remove file "${fileName}"?`)) {
        delete uploadedFiles[fileName];
        updateFilesList();
    }
}

function skipFile() {
    showStep(5);
    startSolver();
}

// File Mapping Functions
async function checkFileMapping() {
    const fileCount = Object.keys(uploadedFiles).length;

    // If no files or only one file, no mapping needed
    if (fileCount <= 1) {
        startSolver();
        return;
    }

    // Request parameter names from backend
    try {
        const response = await fetch('/api/get_required_parameters', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                problem_description: document.getElementById('problem-description').value
            })
        });

        const data = await response.json();

        if (!data.parameters || data.parameters.length === 0) {
            // No parameters defined, just proceed
            startSolver();
            return;
        }

        if (data.parameters.length === 1 && fileCount === 1) {
            // Auto-map single file to single parameter
            const fileNames = Object.keys(uploadedFiles);
            fileMapping[data.parameters[0]] = fileNames[0];
            startSolver();
            return;
        }

        // Show mapping interface
        showMappingInterface(data.parameters);

    } catch (error) {
        console.error('Error getting parameters:', error);
        // On error, proceed without mapping
        startSolver();
    }
}

function showMappingInterface(parameters) {
    const mappingSection = document.getElementById('file-mapping-section');
    const mappingControls = document.getElementById('mapping-controls');
    const proceedBtn = document.getElementById('proceed-solve-btn');

    // Clear previous mappings
    mappingControls.innerHTML = '';
    fileMapping = {};

    // Create dropdown for each parameter
    const fileNames = Object.keys(uploadedFiles);

    parameters.forEach(param => {
        const row = document.createElement('div');
        row.className = 'mapping-row';

        const label = document.createElement('div');
        label.className = 'mapping-label';
        label.textContent = formatParameterName(param);

        const select = document.createElement('select');
        select.className = 'mapping-select';
        select.dataset.parameter = param;

        // Add placeholder option
        const placeholder = document.createElement('option');
        placeholder.value = '';
        placeholder.textContent = '-- Select file --';
        select.appendChild(placeholder);

        // Add file options
        fileNames.forEach(fileName => {
            const option = document.createElement('option');
            option.value = fileName;
            option.textContent = `${fileName} (${uploadedFiles[fileName].filename})`;
            select.appendChild(option);
        });

        // Try to auto-suggest based on parameter name
        const suggested = suggestFileForParameter(param, fileNames);
        if (suggested) {
            select.value = suggested;
            select.style.borderColor = '#10b981'; // Green border for suggestion
        }

        row.appendChild(label);
        row.appendChild(select);
        mappingControls.appendChild(row);
    });

    // Show mapping section, hide proceed button
    mappingSection.style.display = 'block';
    proceedBtn.style.display = 'none';
}

function formatParameterName(param) {
    // Convert snake_case to Title Case
    return param
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

function suggestFileForParameter(param, fileNames) {
    const paramLower = param.toLowerCase();

    // Simple keyword matching
    const keywords = {
        'travel': ['travel', 'tt', 'distance'],
        'job': ['job', 'jt', 'task'],
        'time': ['time', 'duration']
    };

    for (const fileName of fileNames) {
        const fileLower = fileName.toLowerCase();

        // Check for keyword matches
        for (const [key, synonyms] of Object.entries(keywords)) {
            if (paramLower.includes(key)) {
                for (const syn of synonyms) {
                    if (fileLower.includes(syn)) {
                        return fileName;
                    }
                }
            }
        }
    }

    return null;
}

function confirmMapping() {
    const selects = document.querySelectorAll('.mapping-select');
    const usedFiles = new Set();
    fileMapping = {};

    // Validate mapping
    for (const select of selects) {
        const param = select.dataset.parameter;
        const file = select.value;

        if (!file) {
            alert(`Please select a file for "${formatParameterName(param)}"`);
            return;
        }

        if (usedFiles.has(file)) {
            alert(`File "${file}" is already mapped to another parameter. Each file can only be used once.`);
            return;
        }

        usedFiles.add(file);
        fileMapping[param] = file;
    }

    // Hide mapping interface
    document.getElementById('file-mapping-section').style.display = 'none';

    // Proceed to solve
    startSolver();
}

// Step 5: Solve
async function startSolver() {
    showStep(5);
    clearLog();

    try {
        const response = await fetch('/api/solve', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                answers: answers,
                hyperparameters: hyperparameters,
                file_mapping: fileMapping  // Send explicit file-to-parameter mapping
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Failed to start solver');
        }

        addLog('Solver started successfully', 'info');

    } catch (error) {
        alert('Error: ' + error.message);
    }
}

// Progress Updates
function updateProgress(percent, stage, message) {
    const progressFill = document.getElementById('progress-fill');
    const stageName = document.getElementById('progress-stage-name');
    const progressMessage = document.getElementById('progress-message');

    progressFill.style.width = percent + '%';
    progressFill.textContent = percent + '%';

    const stageNames = {
        'initialization': 'Initializing Solver',
        'analysis': 'Analyzing Problem',
        'generation': 'Generating Code',
        'compilation': 'Compiling Solver',
        'execution': 'Running Genetic Algorithm',
        'finalizing': 'Finalizing Results',
        'complete': 'Complete!'
    };

    stageName.textContent = stageNames[stage] || stage;
    progressMessage.textContent = message;
}

// Log Management
function addLog(message, level = 'info') {
    const logOutput = document.getElementById('log-output');
    const logEntry = document.createElement('div');
    logEntry.className = `log-entry ${level}`;

    const timestamp = new Date().toLocaleTimeString();
    logEntry.innerHTML = `<span class="log-timestamp">[${timestamp}]</span>${message}`;

    logOutput.appendChild(logEntry);
    logOutput.scrollTop = logOutput.scrollHeight;
}

function clearLog() {
    document.getElementById('log-output').innerHTML = '';
}

// Step 6: Results
function showResults(data) {
    showStep(6);

    const resultSummary = document.getElementById('result-summary');
    const solutionPreview = document.getElementById('solution-preview');
    const solutionContent = document.getElementById('solution-content');

    if (data.success) {
        resultSummary.className = 'result-box success';
        resultSummary.innerHTML = `
            <h3>Success! Solver Completed</h3>
            <div class="result-stats">
                <div class="stat-item">
                    <div class="stat-label">Best Fitness</div>
                    <div class="stat-value">${data.result.best_fitness ? data.result.best_fitness.toFixed(2) : 'N/A'}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Generations</div>
                    <div class="stat-value">${data.result.generations || 'N/A'}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Execution Time</div>
                    <div class="stat-value">${data.result.execution_time ? data.result.execution_time.toFixed(2) + 's' : 'N/A'}</div>
                </div>
            </div>
            <p style="margin-top: 1rem;"><strong>Generated Files:</strong></p>
            <p>Config: <code>${data.result.config_path}</code></p>
            <p>Executable: <code>${data.result.executable_path}</code></p>
        `;

        // Display analysis in results page
        if (analysisData) {
            const analysisWidget = document.getElementById('results-analysis-widget');
            const analysisContent = document.getElementById('results-analysis-content');
            analysisContent.innerHTML = buildAnalysisHTML(analysisData);
            analysisWidget.style.display = 'block';
        }

        // Display convergence graph in results page
        if (convergenceData.generations.length > 0) {
            displayResultsConvergenceGraph();
        }

        if (data.result.solution) {
            solutionContent.textContent = data.result.solution;
            solutionPreview.style.display = 'block';
        }
    } else {
        resultSummary.className = 'result-box error';
        resultSummary.innerHTML = `
            <h3>Solver Failed</h3>
            <p>The solver could not complete successfully. Please check the problem description and try again.</p>
            <p style="margin-top: 1rem;">Generated config: <code>${data.result.config_path}</code></p>
        `;
    }
}

async function exportSolution() {
    window.location.href = `/api/export/${sessionId}`;
}

function copySolution() {
    const solutionContent = document.getElementById('solution-content').textContent;
    navigator.clipboard.writeText(solutionContent).then(() => {
        alert('Solution copied to clipboard!');
    }).catch(err => {
        alert('Failed to copy: ' + err);
    });
}

// Utility Functions
function showStep(stepNumber) {
    document.querySelectorAll('.step').forEach(step => {
        step.classList.remove('active');
    });
    document.getElementById(`step-${stepNumber}`).classList.add('active');
}

function showLoading(message = 'Processing...') {
    document.getElementById('loading-text').textContent = message;
    document.getElementById('loading-overlay').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loading-overlay').style.display = 'none';
}

function resetApp() {
    sessionId = null;
    questions = [];
    answers = {};
    uploadedFiles = {};
    analysisData = null;

    document.getElementById('problem-description').value = '';
    document.getElementById('questions-container').innerHTML = '';
    updateFilesList();
    clearLog();

    // Reset Step 5 widgets
    document.getElementById('analysis-widget').style.display = 'none';
    document.getElementById('code-widget').style.display = 'none';
    document.getElementById('convergence-widget').style.display = 'none';

    // Reset Step 6 widgets
    document.getElementById('results-analysis-widget').style.display = 'none';
    document.getElementById('results-convergence-widget').style.display = 'none';

    // Reset convergence data and charts
    convergenceData = { generations: [], bestFitness: [], avgFitness: [], worstFitness: [] };
    if (convergenceChart) {
        convergenceChart.destroy();
        convergenceChart = null;
    }
    if (resultsConvergenceChart) {
        resultsConvergenceChart.destroy();
        resultsConvergenceChart = null;
    }

    showStep(1);
}

// Display Analysis Results
function displayAnalysisResults(analysis) {
    // Store globally for results page
    analysisData = analysis;

    // Display in Step 5
    const widget = document.getElementById('analysis-widget');
    const content = document.getElementById('analysis-content');

    const html = buildAnalysisHTML(analysis);
    content.innerHTML = html;
    widget.style.display = 'block';

    addLog('Problem analysis complete', 'success');
}

// Build Analysis HTML (reusable)
function buildAnalysisHTML(analysis) {
    let html = '<div class="analysis-results">';

    html += `<div class="analysis-item">
        <strong>Problem Type:</strong> <span class="badge">${analysis.problem_type || 'Unknown'}</span>
    </div>`;

    html += `<div class="analysis-item">
        <strong>Chromosome Length:</strong> <span>${analysis.chromosome_length || 'Auto'}</span>
    </div>`;

    if (analysis.objectives && analysis.objectives.length > 0) {
        html += `<div class="analysis-item">
            <strong>Objectives:</strong>
            <ul>`;
        analysis.objectives.forEach(obj => {
            html += `<li>${obj.name}: ${obj.type} (weight: ${obj.weight || 1.0})</li>`;
        });
        html += `</ul></div>`;
    }

    if (analysis.constraints && analysis.constraints.length > 0) {
        html += `<div class="analysis-item">
            <strong>Constraints:</strong>
            <ul>`;
        analysis.constraints.forEach(constraint => {
            html += `<li>${constraint}</li>`;
        });
        html += `</ul></div>`;
    }

    if (analysis.decoder_strategy) {
        html += `<div class="analysis-item">
            <strong>Decoder Strategy:</strong> <span>${analysis.decoder_strategy}</span>
        </div>`;
    }

    if (analysis.estimated_complexity) {
        html += `<div class="analysis-item">
            <strong>Estimated Complexity:</strong> <span class="badge">${analysis.estimated_complexity}</span>
        </div>`;
    }

    html += '</div>';
    return html;
}

// Display Generated Code
function displayGeneratedCode(code, configPath) {
    const widget = document.getElementById('code-widget');
    const content = document.getElementById('code-content');

    content.innerHTML = `<pre><code>${escapeHtml(code)}</code></pre>`;
    widget.style.display = 'block';

    addLog(`Code generated: ${configPath}`, 'success');
}

// Toggle code expansion
function toggleCodeExpansion() {
    const content = document.getElementById('code-content');
    const btn = document.getElementById('expand-code-btn');

    if (content.style.maxHeight === '300px') {
        content.style.maxHeight = 'none';
        btn.textContent = 'Collapse';
    } else {
        content.style.maxHeight = '300px';
        btn.textContent = 'Expand';
    }
}

// Initialize Convergence Graph
function initializeConvergenceGraph() {
    const canvas = document.getElementById('convergence-chart');
    const ctx = canvas.getContext('2d');

    // Show widget
    document.getElementById('convergence-widget').style.display = 'block';

    // Reset data
    convergenceData = { generations: [], bestFitness: [], avgFitness: [], worstFitness: [] };

    // Create chart
    convergenceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: convergenceData.generations,
            datasets: [
                {
                    label: 'Best Fitness',
                    data: convergenceData.bestFitness,
                    borderColor: 'rgb(16, 185, 129)',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    borderWidth: 2,
                    tension: 0.1,
                    fill: true
                },
                {
                    label: 'Average Fitness',
                    data: convergenceData.avgFitness,
                    borderColor: 'rgb(59, 130, 246)',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 1,
                    tension: 0.1,
                    fill: false
                },
                {
                    label: 'Worst Fitness',
                    data: convergenceData.worstFitness,
                    borderColor: 'rgb(239, 68, 68)',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    borderWidth: 1,
                    tension: 0.1,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 2,
            animation: {
                duration: 0 // Disable animations for real-time updates
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Generation'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Fitness'
                    },
                    beginAtZero: false
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            }
        }
    });
}

// Update Convergence Graph
function updateConvergenceGraph(data) {
    if (!convergenceChart) {
        initializeConvergenceGraph();
    }

    // Add new data point
    convergenceData.generations.push(data.generation);
    convergenceData.bestFitness.push(data.best);
    convergenceData.avgFitness.push(data.avg);
    convergenceData.worstFitness.push(data.worst);

    // Update chart
    convergenceChart.data.labels = convergenceData.generations;
    convergenceChart.data.datasets[0].data = convergenceData.bestFitness;
    convergenceChart.data.datasets[1].data = convergenceData.avgFitness;
    convergenceChart.data.datasets[2].data = convergenceData.worstFitness;
    convergenceChart.update('none'); // 'none' mode for no animation
}

// Display convergence graph in results page
function displayResultsConvergenceGraph() {
    const widget = document.getElementById('results-convergence-widget');
    const canvas = document.getElementById('results-convergence-chart');
    const ctx = canvas.getContext('2d');

    // Show widget
    widget.style.display = 'block';

    // Destroy existing chart if any
    if (resultsConvergenceChart) {
        resultsConvergenceChart.destroy();
    }

    // Create new chart with the stored data
    resultsConvergenceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: convergenceData.generations,
            datasets: [
                {
                    label: 'Best Fitness',
                    data: convergenceData.bestFitness,
                    borderColor: 'rgb(16, 185, 129)',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    borderWidth: 2,
                    tension: 0.1,
                    fill: true
                },
                {
                    label: 'Average Fitness',
                    data: convergenceData.avgFitness,
                    borderColor: 'rgb(59, 130, 246)',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 1,
                    tension: 0.1,
                    fill: false
                },
                {
                    label: 'Worst Fitness',
                    data: convergenceData.worstFitness,
                    borderColor: 'rgb(239, 68, 68)',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    borderWidth: 1,
                    tension: 0.1,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 2,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Generation'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Fitness'
                    },
                    beginAtZero: false
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            }
        }
    });
}

// Utility function to escape HTML
function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}
