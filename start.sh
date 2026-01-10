#!/bin/bash

# AI Resume Screening System - Quick Start Script
# This script pulls latest changes and starts the server

echo "🚀 AI Resume Screening System - Quick Start"
echo "==========================================="
echo ""

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Pull latest changes
echo "📥 Pulling latest changes from GitHub..."
git pull origin main

# Run setup to ensure dependencies are updated
echo "🔧 Ensuring dependencies are up to date..."
bash setup.sh

echo ""
echo "🌐 Starting backend server..."
cd backend
source venv/bin/activate
uvicorn main:app --reload --port 8000
