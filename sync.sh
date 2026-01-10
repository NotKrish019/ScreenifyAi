#!/bin/bash

# Sync Script - For your friend to pull and update automatically
# This script pulls latest code and updates everything automatically

echo "🔄 Syncing AI Resume Screening System..."
echo "========================================"
echo ""

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Stash any local changes
echo "💾 Saving any local changes..."
git stash

# Pull latest changes
echo "📥 Pulling latest code from GitHub..."
git pull origin main

if [ $? -eq 0 ]; then
    echo "✅ Code updated successfully!"
else
    echo "❌ Failed to pull changes. Please check your internet connection."
    exit 1
fi

# Navigate to backend
cd backend

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Update dependencies
echo "📚 Updating dependencies..."
pip install -r requirements.txt --quiet --upgrade

echo ""
echo "✅ Sync complete! Everything is up to date."
echo ""
echo "🎯 To start the server, run:"
echo "   ./start.sh"
echo ""
echo "   Or manually:"
echo "   cd backend && source venv/bin/activate && uvicorn main:app --reload --port 8000"
echo ""
