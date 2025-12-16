#!/bin/bash

# LLM BRKGA Solver - Web Interface Startup Script

echo "=========================================="
echo "LLM BRKGA Solver - Web Interface"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements_web.txt

# Check for ANTHROPIC_API_KEY
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo ""
    echo "WARNING: ANTHROPIC_API_KEY environment variable is not set!"
    echo "Please set it with: export ANTHROPIC_API_KEY='your-key-here'"
    echo ""
    read -p "Do you want to continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create necessary directories
echo "Setting up directories..."
mkdir -p uploads
mkdir -p llm_solver/generated
mkdir -p llm_solver/results

# Start the web server
echo ""
echo "=========================================="
echo "Starting web server..."
echo "=========================================="
echo ""
python3 web_app.py
